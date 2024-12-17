import os

import numpy as np

import scipy.spatial.transform
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.ndimage
import scipy.linalg
import scipy.stats

from lmfit import Minimizer, Parameters

from mantid.simpleapi import mtd
from mantid import config
config['Q.convention'] = 'Crystallography'

config['MultiThreaded.MaxCores'] == '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_THREAD_ENABLED'] = '0'

from garnet.plots.peaks import RadiusPlot, PeakPlot
from garnet.config.instruments import beamlines
from garnet.reduction.ub import UBModel, Optimization, lattice_group
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.data import DataModel
from garnet.reduction.plan import SubPlan
from garnet.reduction.parallel import ParallelProcessor

class Integration(SubPlan):

    def __init__(self, plan):

        super(Integration, self).__init__(plan)

        self.params = plan['Integration']
        self.output = plan['OutputName']+'_integration'

        self.validate_params()

    def validate_params(self):

        assert self.params['Cell'] in lattice_group.keys()
        assert self.params['Centering'] in centering_reflection.keys()
        assert self.params['MinD'] > 0
        assert self.params['Radius'] > 0

        if self.params.get('ModVec1') is None:
            self.params['ModVec1'] = [0, 0, 0]
        if self.params.get('ModVec2') is None:
            self.params['ModVec2'] = [0, 0, 0]
        if self.params.get('ModVec3') is None:
            self.params['ModVec3'] = [0, 0, 0]

        if self.params.get('MaxOrder') is None:
            self.params['MaxOrder'] = 0
        if self.params.get('CrossTerms') is None:
            self.params['CrossTerms'] = False

        assert len(self.params['ModVec1']) == 3
        assert len(self.params['ModVec2']) == 3
        assert len(self.params['ModVec3']) == 3

        assert self.params['MaxOrder'] >= 0
        assert type(self.params['CrossTerms']) is bool

    def integrate(self, n_proc=1):

        data = DataModel(beamlines[self.plan['Instrument']])

        instance = Integration(self.plan)
        instance.n_proc = n_proc

        if data.laue:
            return instance.laue_integrate()
        else:
            return instance.monochromatic_integrate()

    def integrate_peaks(self, data):

        pp = ParallelProcessor(n_proc=self.n_proc)
        return pp.process_dict(data, self.fit_peaks)

    def laue_integrate(self):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        runs = self.plan['Runs']

        self.run = 0
        self.runs = len(runs)

        for run in runs:

            self.run += 1

            print('{:3}/{:3}'.format(self.run, len(runs)))

            data.load_data('data',
                           self.plan['IPTS'],
                           run,
                           self.plan.get('Grouping'))

            data.apply_calibration('data',
                                   self.plan.get('DetectorCalibration'),
                                   self.plan.get('TubeCalibration'))

            data.preprocess_detectors('data')

            data.load_efficiency_file(self.plan['EfficiencyFile'])

            data.load_spectra_file(self.plan['SpectraFile'])

            data.crop_for_normalization('data')

            data.apply_mask('data', self.plan.get('MaskFile'))

            data.load_background(self.plan['BackgroundFile'], 'data')

            data.calculate_correction_factor()

            data.normalize_data('data')

            data.convert_to_Q_sample('data', 'md')

            data.load_clear_UB(self.plan['UBFile'], 'data', run)

            lamda_min, lamda_max = data.wavelength_band

            peaks.predict_peaks('data',
                                'peaks',
                                self.params['Centering'],
                                self.params['MinD'],
                                lamda_min,
                                lamda_max)

            r_cut = self.params['Radius']

            peaks.integrate_peaks('md', 'peaks', r_cut)

            peaks.remove_weak_peaks('peaks', 10)

            self.peaks, self.data = peaks, data

            params = self.estimate_peak_size('peaks', 'md', r_cut)

            peaks.predict_peaks('data',
                                'peaks',
                                self.params['Centering'],
                                self.params['MinD'],
                                lamda_min,
                                lamda_max)

            if self.params['MaxOrder'] > 0:

                peaks.predict_satellite_peaks('peaks',
                                              'md',
                                              self.params['MinD'],
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

            data.delete_workspace('data')

            md_file = self.get_diagnostic_file('run#{}_data'.format(run))

            data.save_histograms(md_file, 'md', sample_logs=True)

            peak_dict = self.extract_peak_info('peaks', params)

            results = self.integrate_peaks(peak_dict)

            peak_dict = dict(results)

            self.update_peak_info('peaks', peak_dict)

            peaks.remove_weak_peaks('peaks')

            peaks.combine_peaks('peaks', 'combine')

            pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))

            peaks.save_peaks(pk_file, 'peaks')

            data.delete_workspace('peaks')

            data.delete_workspace('md')

        result_file = self.get_file(output_file, '')

        peaks.save_peaks(result_file, 'combine')

        # ---

        if mtd.doesExist('combine'):

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(output_file)[0]+'.mat'

            ub = UBModel('combine')
            ub.save_UB(ub_file)

        mtd.clear()



    def monochromatic_integrate(self):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        runs = self.plan['Runs']

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        self.run = 0
        self.runs = len(runs)

        if self.plan['Instrument'] == 'WAND²':

            self.runs = 1
            self.run += 1

            data.load_data('data',
                           self.plan['IPTS'],
                           runs,
                           self.plan.get('Grouping'))

            data.load_generate_normalization(self.plan['VanadiumFile'], 'data')

            data.convert_to_Q_sample('data', 'md', lorentz_corr=True)

            md_file = self.get_diagnostic_file('run#{}_data'.format(self.run))
            data.save_histograms(md_file, 'md', sample_logs=True)

        else:

            for run in runs:

                self.run += 1

                data.load_data('data',
                               self.plan['IPTS'],
                               run,
                               self.plan.get('Grouping'))

                data.load_generate_normalization(self.plan['VanadiumFile'],
                                                 'data')

                data.convert_to_Q_sample('data',
                                         'md',
                                         lorentz_corr=True)

                if self.plan.get('UBFile') is None:
                    UB_file = output_file.replace('.nxs', '.mat')
                    data.save_UB(UB_file, 'md_data')
                    self.plan['UBFile'] = UB_file

                data.load_clear_UB(self.plan['UBFile'], 'md')

                peaks.predict_peaks('md',
                                    'peaks',
                                    self.params['Centering'],
                                    self.params['MinD'],
                                    lamda_min,
                                    lamda_max)

                if self.params['MaxOrder'] > 0:

                    peaks.predict_satellite_peaks('peaks',
                                                  'md',
                                                  self.params['MinD'],
                                                  lamda_min,
                                                  lamda_max,
                                                  self.params['ModVec1'],
                                                  self.params['ModVec2'],
                                                  self.params['ModVec3'],
                                                  self.params['MaxOrder'],
                                                  self.params['CrossTerms'])

                self.peaks, self.data = peaks, data

                params = self.estimate_peak_size('peaks', 'md')

                self.fit_peaks('peaks', params)

                peaks.combine_peaks('peaks', 'combine')

                md_file = self.get_diagnostic_file('run#{}_data'.format(run))
                data.save_histograms(md_file, 'md', sample_logs=True)

                pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))
                peaks.save_peaks(pk_file, 'peaks')

        if self.plan['Instrument'] != 'WAND²':

            peaks.remove_weak_peaks('combine')

            peaks.save_peaks(output_file, 'combine')

        mtd.clear()

        return output_file

    def monochromatic_combine(self, files):

        output_file = self.get_output_file()
        result_file = self.get_file(output_file, '')

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        if self.plan['Instrument'] == 'WAND²':

            merge = []
            for file in files:

                peaks.load_peaks(file, 'peaks')
                peaks.combine_peaks('peaks', 'combine')

                md_file = file.replace('_peaks', '_data')
                data.load_histograms(md_file, md_file)

                merge.append(md_file)
                os.remove(md_file)

            data.combine_Q_sample(merge, 'md')

            if self.plan.get('UBFile') is None:
                UB_file = output_file.replace('.nxs', '.mat')
                data.save_UB(UB_file, 'md')
                self.plan['UBFile'] = UB_file

            data.load_clear_UB(self.plan['UBFile'], 'md')

            peaks.predict_peaks('md',
                                'peaks',
                                self.params['Centering'],
                                self.params['MinD'],
                                lamda_min,
                                lamda_max)

            if self.params['MaxOrder'] > 0:

                peaks.predict_satellite_peaks('peaks',
                                              'md',
                                              self.params['MinD'],
                                              lamda_min,
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

            self.peaks, self.data = peaks, data

            params = self.estimate_peak_size('peaks', 'md')

            self.fit_peaks('peaks', params)

            md_file = self.get_diagnostic_file('data')
            data.save_histograms(md_file, 'md', sample_logs=True)

            pk_file = self.get_diagnostic_file('peaks')
            peaks.save_peaks(pk_file, 'peaks')

        else:

            for file in files:

                peaks.load_peaks(file, 'tmp')
                peaks.combine_peaks('tmp', 'combine')

            for file in files:
                os.remove(file)

        if mtd.doesExist('combine'):

            peaks.save_peaks(result_file, 'combine')

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(result_file)[0]+'.mat'

            ub = UBModel('combine')
            ub.save_UB(ub_file)

        mtd.clear()

    def get_file(self, file, ws=''):
        """
        Update filename with identifier name and optional workspace name.

        Parameters
        ----------
        file : str
            Original file name.
        ws : str, optional
            Name of workspace. The default is ''.

        Returns
        -------
        output_file : str
            File with updated name for identifier and workspace name.

        """

        if len(ws) > 0:
            ws = '_'+ws

        return self.append_name(file).replace('.nxs', ws+'.nxs')

    def append_name(self, file):
        """
        Update filename with identifier name.

        Parameters
        ----------
        file : str
            Original file name.

        Returns
        -------
        output_file : str
            File with updated name for identifier name.

        """

        append = self.cell_centering_name() \
               + self.modulation_name() \
               + self.resolution_name()

        name, ext = os.path.splitext(file)

        return name+append+ext

    def cell_centering_name(self):
        """
        Lattice and reflection condition.

        Returns
        -------
        lat_ref : str
            Underscore separated strings.

        """

        cell = self.params['Cell']
        centering = self.params['Centering']

        return '_'+cell+'_'+centering

    def modulation_name(self):
        """
        Modulation vectors.

        Returns
        -------
        mod : str
            Underscore separated vectors and max order

        """

        mod = ''

        max_order = self.params.get('MaxOrder')
        mod_vec_1 = self.params.get('ModVec1')
        mod_vec_2 = self.params.get('ModVec1')
        mod_vec_3 = self.params.get('ModVec3')
        cross_terms = self.params.get('CrossTerms')

        if max_order > 0:
            for vec in [mod_vec_1, mod_vec_2, mod_vec_3]:
                if np.linalg.norm(vec) > 0:
                   mod += '_({},{},{})'.format(*vec)
            if cross_terms:
                mod += '_mix'

        return mod

    def resolution_name(self):
        """
        Minimum d-spacing and starting radii

        Returns
        -------
        res_rad : str
            Underscore separated strings.

        """

        min_d = self.params['MinD']
        max_r = self.params['Radius']

        return '_d(min)={:.2f}'.format(min_d)+'_r(max)={:.2f}'.format(max_r)

    def estimate_peak_size(self, peaks_ws, data_ws, r_cut):
        """
        Integrate peaks with spherical envelope up to cutoff size.
        Estimates spherical envelope radius parameters.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data.
        r_cut : float
            Cutoff radius.

        Returns
        -------
        lo : float
            Nominal profile radius.
        lc : float
            Scale profile radius.
        to : float
            Nominal projection radius.
        tc : float
            Scale projection radius.

        """

        peaks = self.peaks

        peaks_name = peaks.get_peaks_name(peaks_ws)

        params = peaks.intensity_vs_radius(data_ws, peaks_ws, r_cut)

        r, sig_noise, x, y, Q = params

        sphere = PeakSphere(r_cut)

        r_cut = sphere.fit(r, sig_noise)

        sig_noise_fit, *vals = sphere.best_fit(r)

        plot = RadiusPlot(r, sig_noise, sig_noise_fit)

        plot.add_sphere(r_cut, *vals)

        profile = PeakProfile(r_cut)

        x, y, lamda = peaks.intensity_profile(data_ws, peaks_ws, r_cut)

        lo, lc, hist, r_bins, l_bins = profile.fit(x, y, lamda)

        plot.add_profile(hist, r_bins, l_bins)

        projection = PeakProjection(r_cut)

        x, y, theta = peaks.intensity_projection(data_ws, peaks_ws, r_cut)

        to, tc, hist, r_bins, t_bins = projection.fit(x, y, theta)

        plot.add_projection(hist, r_bins, t_bins)

        plot.save_plot(self.get_plot_file(peaks_name))

        return lo, lc, to, tc

    def fit_peaks(self, key_value, make_plot=True):

        if make_plot:

            plot = PeakPlot()

        key, value = key_value

        data_info, peak_info = value

        Q0, Q1, Q2, counts, y, e, dQ, projections = data_info

        peak_name, wavelength, angles, goniometer = peak_info

        ellipsoid = PeakEllipsoid()

        params = ellipsoid.fit(Q0, Q1, Q2, counts, y, e, dQ)

        value = None

        if params is not None:

            c, S, *fitting = ellipsoid.best_fit

            shape = self.revert_ellipsoid_parameters(params, projections)

            norm_params = Q0, Q1, Q2, y, e, counts, c, S

            I, sigma = ellipsoid.integrate_norm(*norm_params)

            if make_plot:

                plot.add_fitting(*fitting)

                plot.add_profile_fit(*ellipsoid.best_prof)

                plot.add_projection_fit(*ellipsoid.best_proj)

                plot.add_ellipsoid(c, S)

                plot.add_peak_info(wavelength, angles, goniometer)

                plot.add_peak_stats(ellipsoid.redchi2)

                plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

                plot.save_plot(self.get_plot_file(peak_name))

            value = I, sigma, shape, ellipsoid.info

        return key, value

    def extract_peak_info(self, peaks_ws, params):
        """
        Obtain peak information for envelope determination.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        params : list
            Cutoff radius parameters.

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        UB = self.peaks.get_UB(peaks_ws)

        lo, lc, to, tc = params

        peak_dict = {}

        for i in range(n_peak):

            # d = peak.get_d_spacing(i)

            h, k, l = peak.get_hkl(i)

            wavelength = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            l_cut = lo+lc*wavelength
            t_cut = to+tc*two_theta/2

            params = peak.get_peak_shape(i, l_cut)

            peak.set_peak_intensity(i, 0, 0)

            goniometer = peak.get_goniometer_angles(i)

            peak_name = peak.get_peak_name(i)

            # det_id = peak.get_detector_id(i)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            R = peak.get_goniometer_matrix(i)

            bin_params = l_cut, t_cut, dQ, R, two_theta, az_phi, UB

            # ---

            bins, extents, projections = self.bin_extent(*params, *bin_params)

            y, e, Q0, Q1, Q2 = data.bin_in_Q('md', extents, bins, projections)

            counts = data.extract_counts('md_bin')

            data_info = (Q0, Q1, Q2, counts, y, e, dQ, projections)

            peak_info = (peak_name, wavelength, angles, goniometer)

            peak_dict[i] = data_info, peak_info

        return peak_dict

    def update_peak_info(self, peaks_ws, peak_dict):

        peak = PeakModel(peaks_ws)

        for i, value in peak_dict.items():

            if value is not None:

                I, sigma, shape, info = value

                peak.set_peak_intensity(i, I, sigma)

                peak.set_peak_shape(i, *shape)

                peak.add_diagonstic_info(i, info)

    def bin_axes(self, R, two_theta, az_phi):

        two_theta = np.deg2rad(two_theta)
        az_phi = np.deg2rad(az_phi)

        kf_hat = np.array([np.sin(two_theta)*np.cos(az_phi),
                           np.sin(two_theta)*np.sin(az_phi),
                           np.cos(two_theta)])

        ki_hat = np.array([0, 0, 1])

        n = kf_hat-ki_hat
        n /= np.linalg.norm(n)

        v = np.cross(ki_hat, kf_hat)
        v /= np.linalg.norm(v)

        u = np.cross(v, n)
        u /= np.linalg.norm(u)

        return R.T @ n, R.T @ u, R.T @ v

    def project_ellipsoid_parameters(self, params, projections):

        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W.T, [c0, c1, c2]), r0, r1, r2, *np.dot(W.T, V).T

    def revert_ellipsoid_parameters(self, params, projections):

        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W, [c0, c1, c2]), r0, r1, r2, *np.dot(W, V).T

    def trasform_Q(self, Q0, Q1, Q2, projections):

        W = np.column_stack(projections)

        return np.einsum('ij,j...->i...', W, [Q0, Q1, Q2])

    def bin_extent(self, Q0, Q1, Q2,
                         r0, r1, r2,
                         v0, v1, v2, l_cut, t_cut, bin_size,
                         R, two_theta, az_phi, UB):

        n, u, v = self.bin_axes(R, two_theta, az_phi)

        projections = [n, u, v]

        params = Q0, Q1, Q2, r0, r1, r2, v0, v1, v2

        params = self.project_ellipsoid_parameters(params, projections)

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        dQ = 2*np.array([l_cut, t_cut, t_cut])

        W = np.column_stack([v0, v1, v2])
        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(W, V), W.T)

        dQ = np.column_stack([2*np.sqrt(np.diag(S)), dQ]).min(axis=1)

        W = np.column_stack(projections)

        am = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [-0.5, 0, 0]))
        bm = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [0, -0.5, 0]))
        cm = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [0, 0, -0.5]))

        ap = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [0.5, 0, 0]))
        bp = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [0, 0.5, 0]))
        cp = np.dot(W.T, np.einsum('ij,j...->i...', 2*np.pi*UB, [0, 0, 0.5]))

        Q0_min = np.min([am[0], bm[0], cm[0], ap[0], bp[0], cp[0]])
        Q1_min = np.min([am[1], bm[1], cm[1], ap[1], bp[1], cp[1]])
        Q2_min = np.min([am[2], bm[2], cm[2], ap[2], bp[2], cp[2]])

        Q0_max = np.max([am[0], bm[0], cm[0], ap[0], bp[0], cp[0]])
        Q1_max = np.max([am[1], bm[1], cm[1], ap[1], bp[1], cp[1]])
        Q2_max = np.max([am[2], bm[2], cm[2], ap[2], bp[2], cp[2]])

        dQ0, dQ1, dQ2 = dQ

        dQ0 = np.min(np.abs([dQ0, Q0-Q0_min, Q0_max-Q0]))
        dQ1 = np.min(np.abs([dQ1, Q1-Q1_min, Q1_max-Q1]))
        dQ2 = np.min(np.abs([dQ2, Q2-Q2_min, Q2_max-Q2]))

        extents = np.array([[Q0-dQ0, Q0+dQ0],
                            [Q1-dQ1, Q1+dQ1],
                            [Q2-dQ2, Q2+dQ2]])

        bin_sizes = np.array(dQ)/10
        bin_sizes[bin_sizes < bin_size/2] = bin_size/2

        min_adjusted = np.floor(extents[:,0]/bin_sizes)*bin_sizes
        max_adjusted = np.ceil(extents[:,1]/bin_sizes)*bin_sizes

        bins = ((max_adjusted-min_adjusted)/bin_sizes).astype(int)
        bin_sizes = (max_adjusted-min_adjusted)/bins

        bins = np.where(bins % 2 == 0, bins, bins+1)

        max_adjusted = min_adjusted+bins*bin_sizes

        extents = np.vstack((min_adjusted, max_adjusted)).T

        return bins, extents, projections

    @staticmethod
    def combine_parallel(plan, files):

        instance = Integration(plan)

        data = DataModel(beamlines[plan['Instrument']])

        instance = Integration(plan)

        if data.laue:
            return instance.laue_combine(files)
        else:
            return instance.monochromatic_combine(files)


class PeakSphere:

    def __init__(self, r_cut):

        self.params = Parameters()

        if np.isclose(r_cut, 0.04) or r_cut < 0.04:
            r_cut = 0.2

        self.params.add('sigma', value=r_cut/6, min=0.01, max=r_cut/4)

    def model(self, x, A, sigma):

        z = x/sigma

        return A*(scipy.special.erf(z/np.sqrt(2)) -
                  np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2))

    def residual(self, params, x, y):

        A = params['A']
        sigma = params['sigma']

        y_fit = self.model(x, A, sigma)

        diff = y_fit-y
        diff[~np.isfinite(diff)] = 1e9

        return diff

    def fit(self, x, y):

        y_max = np.max(y)

        y[y < 0] = 0

        if np.isclose(y_max, 0):
            y_max = np.inf

        self.params.add('A', value=y_max, min=0, max=100*y_max, vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(x, y),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        return 4*result.params['sigma'].value

    def best_fit(self, r):

        A = self.params['A'].value
        sigma = self.params['sigma'].value

        return self.model(r, A, sigma), A, sigma


class PeakProfile:

    def __init__(self, r_cut):

        self.params = Parameters()

        self.params.add('ro', value=r_cut/2, min=r_cut/8, max=r_cut, vary=True)
        self.params.add('rc', value=0, min=-r_cut, max=r_cut, vary=True)

        self.r_cut = r_cut

    def residual(self, params, r, l):

        ro = params['ro'].value
        rc = params['rc'].value

        r_fit = ro+rc*l

        diff = r_fit-r

        return diff.flatten()

    def fit(self, x, y, l):

        w = y.copy()
        w -= np.nanmin(w, axis=1)[:,np.newaxis]

        c = np.nansum(x*w, axis=1)/np.nansum(w, axis=1)
        x -= c[:,np.newaxis]

        l = np.repeat(l, x.shape[1]).reshape(*x.shape)

        r = x[0]

        dr = np.diff(r).mean()

        r_bins = np.concatenate(([r[0]-0.5*dr],
                                 0.5*(r[1:]+r[:-1]),
                                 [r[-1]+0.5*dr]))

        r, w, l = [array.flatten() for array in [x, w, l]]

        l_bins = np.arange(np.min(l), np.max(l), 0.25)

        hist, _, _ = np.histogram2d(r, l, bins=[r_bins, l_bins], weights=w)
        hist = hist.T

        r = 0.5*(r_bins[:-1]+r_bins[1:])
        l = 0.5*(l_bins[:-1]+l_bins[1:])

        hist.T[np.abs(r) > self.r_cut] = np.nan

        hist -= np.nanmin(hist, axis=1)[:,np.newaxis]

        hist /= np.nansum(hist, axis=1)[:,np.newaxis]
        hist[hist == 0] = np.nan

        c = np.nansum(hist*r, axis=1)
        s = np.sqrt(np.nansum(hist*(r-c[:,np.newaxis])**2, axis=1))*3

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(s, l),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        ro = result.params['ro'].value
        rc = result.params['rc'].value

        return ro, rc, hist, r, l


class PeakProjection:

    def __init__(self, r_cut):

        self.params = Parameters()

        self.params.add('ro', value=r_cut/2, min=r_cut/8, max=r_cut, vary=True)
        self.params.add('rc', value=0, min=-r_cut, max=r_cut, vary=True)

        self.r_cut = r_cut

    def residual(self, params, r, t):

        ro = params['ro'].value
        rc = params['rc'].value

        r_fit = ro+rc*t

        diff = r_fit-r

        return diff.flatten()

    def fit(self, x, y, t):

        w = y.copy()

        r_bins = np.histogram_bin_edges(x, bins='auto')
        t_bins = np.arange(np.min(t), np.max(t), 20)

        hist, _, _ = np.histogram2d(x, t, bins=[r_bins, t_bins], weights=w)
        hist = hist.T

        r = 0.5*(r_bins[:-1]+r_bins[1:])
        t = 0.5*(t_bins[:-1]+t_bins[1:])

        hist -= np.nanmin(hist, axis=1)[:,np.newaxis]

        hist /= np.nansum(hist, axis=1)[:,np.newaxis]

        s = np.nansum(hist*r, axis=1)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(s, t),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        ro = result.params['ro'].value
        rc = result.params['rc'].value

        return ro, rc, hist, r, t


class PeakEllipsoid:

    def __init__(self):

        self.params = Parameters()

        t = np.linspace(0, np.pi, 1024)
        cdf = (t-np.sin(t))/np.pi

        self._angle = scipy.interpolate.interp1d(cdf, t, kind='linear')

    def update_constraints(self, x0, x1, x2, dx):

        r0 = (x0[:,0,0][-1]-x0[:,0,0][0])/4
        r1 = (x1[0,:,0][-1]-x1[0,:,0][0])/4
        r2 = (x2[0,0,:][-1]-x2[0,0,:][0])/4

        r0_max = (x0[:,0,0][-1]-x0[:,0,0][0])#/2
        r1_max = (x1[0,:,0][-1]-x1[0,:,0][0])#/2
        r2_max = (x2[0,0,:][-1]-x2[0,0,:][0])#/2

        c0, c1, c2 = x0[:,0,0].mean(), x1[0,:,0].mean(), x2[0,0,:].mean()

        c0_min, c1_min, c2_min = x0[0,0,0], x1[0,0,0], x2[0,0,0]
        c0_max, c1_max, c2_max = x0[-1,0,0], x1[0,-1,0], x2[0,0,-1]

        self.params.add('c0', value=c0, min=c0_min, max=c0_max)
        self.params.add('c1', value=c1, min=c1_min, max=c1_max)
        self.params.add('c2', value=c2, min=c2_min, max=c2_max)

        self.params.add('r0', value=r0, min=dx, max=r0_max)
        self.params.add('r1', value=r1, min=dx, max=r1_max)
        self.params.add('r2', value=r2, min=dx, max=r2_max)

        self.params.add('u0', value=0.0, min=0, max=1)
        self.params.add('u1', value=0.0, min=0, max=1)
        self.params.add('u2', value=0.0, min=0, max=1)

    def angles(self, u0, u1, u2):

        theta = np.arccos(1-2*u0)
        phi = 2*np.pi*u1

        omega = self._angle(u2)

        return phi, theta, omega

    def eigenvectors(self, W):

        w = scipy.spatial.transform.Rotation.from_matrix(W).as_rotvec()

        omega = np.linalg.norm(w)

        u0, u1, u2 = (0, 0, 1) if np.isclose(omega, 0) else w/omega

        return u0, u1, u2, omega

    def S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        U = self.U_matrix(phi, theta, omega)

        V = np.diag([sigma0**2, sigma1**2, sigma2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        U = self.U_matrix(phi, theta, omega)

        V = np.diag([1/sigma0**2, 1/sigma1**2, 1/sigma2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, phi, theta, omega):

        v0 = np.cos(phi)*np.sin(theta)
        v1 = np.sin(phi)*np.sin(theta)
        v2 = np.cos(theta)

        w = omega*np.array([v0, v1, v2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def centroid_inverse_covariance(self, c0, c1, c2,
                                          r0, r1, r2,
                                          phi, theta, omega):

        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, phi, theta, omega)

        return c, inv_S

    def residual(self, params, x0, x1, x2, ys, es, vs, ws, lamda=0.01):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        y1, y2, y3, y11, y12, y21, y22 = ys
        e1, e2, e3, e11, e12, e21, e22 = es
        v1, v2, v3, v11, v12, v21, v22 = vs
        w1, w2, w3, w11, w12, w21, w22 = ws

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        phi, theta, omega = self.angles(u0, u1, u2)

        # C1 = params['C1']
        # C2 = params['C2']
        # C3 = params['C3']

        B1 = params['B1']
        B2 = params['B2']
        B3 = params['B3']

        # B11 = params['B11']
        # B21 = params['B21']

        # B12 = params['B12']
        # B22 = params['B22']

        A1 = params['A1']
        A2 = params['A2']
        A3 = params['A3']

        # A11 = params['A11']
        # A21 = params['A21']

        # A12 = params['A12']
        # A22 = params['A22']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    phi, theta, omega)

        args = x0, x1, x2, 1, 0, c, inv_S

        y1_gauss = self.gaussian(*args, '1d')
        y2_gauss = self.gaussian(*args, '2d')
        y3_gauss = self.gaussian(*args, '3d')

        # y11_gauss = self.gaussian(*args, '1d1')
        # y21_gauss = self.gaussian(*args, '2d1')

        # y12_gauss = self.gaussian(*args, '1d2')
        # y22_gauss = self.gaussian(*args, '2d2')

        diff = []

        # b1 = B1#+C1*x0[:,0,0]
        # b2 = B2#+C2*x1[0,:,:]+C3*x2[0,:,:]
        # b3 = B3

        y1_fit = A1*y1_gauss+B1
        y2_fit = A2*y2_gauss+B2
        y3_fit = A3*y3_gauss+B3

        # y11_fit = A11*y11_gauss+B11
        # y21_fit = A21*y21_gauss+B21

        # y12_fit = A12*y12_gauss+B12
        # y22_fit = A22*y22_gauss+B22

        u1 = np.sqrt(1+y1_fit**2)
        u2 = np.sqrt(1+y2_fit**2)
        u3 = np.sqrt(1+y3_fit**2)

        # u11 = np.sqrt(1+y11_fit**2)
        # u21 = np.sqrt(1+y21_fit**2)

        # u12 = np.sqrt(1+y12_fit**2)
        # u22 = np.sqrt(1+y22_fit**2)

        res = np.arcsinh(y1*u1-y1_fit*v1)*w1

        diff += res.flatten().tolist()

        res = np.arcsinh(y2*u2-y2_fit*v2)*w2

        diff += res.flatten().tolist()

        res = np.arcsinh(y3*u3-y3_fit*v3)*w3

        diff += res.flatten().tolist()

        # ---

        # res = np.arcsinh(y11*u11-y11_fit*v11)*w11

        # diff += res.flatten().tolist()

        # res = np.arcsinh(y21*u21-y21_fit*v21)*w21

        # diff += res.flatten().tolist()

        # res = np.arcsinh(y12*u12-y12_fit*v12)*w12

        # diff += res.flatten().tolist()

        # res = np.arcsinh(y22*u22-y22_fit*v22)*w22

        # diff += res.flatten().tolist()

        args = x0, x1, x2, c, inv_S

        pk1 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '1d')
        pk2 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '2d')
        pk3 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '3d')

        # pk11 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '1d1')
        # pk21 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '2d1')

        # pk12 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '1d2')
        # pk22 = self.ellipsoid_mask(x0, x1, x2, c, inv_S, '2d2')

        bkg1 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '1d') & (~pk1)
        bkg2 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '2d') & (~pk2)
        bkg3 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '3d') & (~pk3)

        # bkg11 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '1d1') & (~pk11)
        # bkg21 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '2d1') & (~pk21)

        # bkg12 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '1d2') & (~pk12)
        # bkg22 = self.ellipsoid_mask(x0, x1, x2, c, 0.25*inv_S, '2d2') & (~pk22)

        b1 = np.nanmean(y1[bkg1])
        b2 = np.nanmean(y2[bkg2])
        b3 = np.nanmean(y3[bkg3])

        # b11 = np.nanmean(y11[bkg11])
        # b21 = np.nanmean(y21[bkg21])

        # b12 = np.nanmean(y12[bkg12])
        # b22 = np.nanmean(y22[bkg22])

        b1_err = np.sqrt(np.nanmean(y1[bkg1]**2))
        b2_err = np.sqrt(np.nanmean(y2[bkg2]**2))
        b3_err = np.sqrt(np.nanmean(y3[bkg3]**2))

        # b11_err = np.sqrt(np.nanmean(y11[bkg11]**2))
        # b21_err = np.sqrt(np.nanmean(y21[bkg21]**2))

        # b12_err = np.sqrt(np.nanmean(y12[bkg12]**2))
        # b22_err = np.sqrt(np.nanmean(y22[bkg22]**2))

        I1 = np.nansum(y1[pk1]-b1)
        I2 = np.nansum(y2[pk2]-b2)
        I3 = np.nansum(y3[pk3]-b3)

        # I11 = np.nansum(y11[pk11]-b11)
        # I21 = np.nansum(y21[pk21]-b21)

        # I12 = np.nansum(y12[pk12]-b12)
        # I22 = np.nansum(y22[pk22]-b22)

        sig1 = np.sqrt(np.nansum(e1[pk1]**2+b1_err**2))
        sig2 = np.sqrt(np.nansum(e2[pk2]**2+b2_err**2))
        sig3 = np.sqrt(np.nansum(e3[pk3]**2+b3_err**2))

        # sig11 = np.sqrt(np.nansum(e11[pk11]**2+b11_err**2))
        # sig21 = np.sqrt(np.nansum(e21[pk21]**2+b21_err**2))

        # sig12 = np.sqrt(np.nansum(e12[pk12]**2+b12_err**2))
        # sig22 = np.sqrt(np.nansum(e22[pk22]**2+b22_err**2))

        sig = np.array([sig1,sig2,sig3])
        I = np.array([I1,I2,I3])

        penalty = lamda*sig/I
        penalty[~np.isfinite(penalty)] = lamda
        penalty[np.isclose(sig, 0)] = lamda

        diff += penalty.tolist()

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def integrate(self, x0, x1, x2, counts, y, e, mode='1d'):

        if mode == '1d':
            c_int = np.nansum(counts, axis=(1,2))
            n_int = c_int/np.nansum(y, axis=(1,2))
            m_int = c_int/np.nansum(e**2, axis=(1,2))
        elif mode == '2d':
            c_int = np.nansum(counts, axis=0)
            n_int = c_int/np.nansum(y, axis=0)
            m_int = c_int/np.nansum(e**2, axis=0)
        elif mode == '3d':
            c_int = counts.copy()
            n_int = c_int/y
            m_int = c_int/e**2
        elif mode == '1d1':
            c_int = np.nansum(counts, axis=(0,2))
            n_int = c_int/np.nansum(y, axis=(0,2))
            m_int = c_int/np.nansum(e**2, axis=(0,2))
        elif mode == '1d2':
            c_int = np.nansum(counts, axis=(0,1))
            n_int = c_int/np.nansum(y, axis=(0,1))
            m_int = c_int/np.nansum(e**2, axis=(0,1))
        elif mode == '2d1':
            c_int = np.nansum(counts, axis=1)
            n_int = c_int/np.nansum(y, axis=1)
            m_int = c_int/np.nansum(e**2, axis=1)
        elif mode == '2d2':
            c_int = np.nansum(counts, axis=2)
            n_int = c_int/np.nansum(y, axis=2)
            m_int = c_int/np.nansum(e**2, axis=2)

        mask = (c_int > 0) & np.isfinite(c_int) \
             & (n_int > 0) & np.isfinite(n_int) \
             & (m_int > 0) & np.isfinite(m_int)

        y_int = c_int/n_int
        e_int = np.sqrt(c_int/m_int)

        y_int[~mask] = np.nan
        e_int[~mask] = np.nan

        return y_int, e_int

    def ellipsoid_mask(self, x0, x1, x2, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        if mode == '3d':
            inv_s = inv_S
        elif mode == '2d':
            inv_s = inv_S[1:,1:]
        elif mode == '1d':
            inv_s = inv_S[0,0]
        elif mode == '2d1':
            inv_s = inv_S[0::2,0::2]
        elif mode == '2d2':
            inv_s = inv_S[:2,:2]
        elif mode == '1d1':
            inv_s = inv_S[1,1]
        elif mode == '1d2':
            inv_s = inv_S[2,2]

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
        elif mode == '2d':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
        elif mode == '1d':
            dx = dx0[:,0,0]
            d2 = inv_s*dx**2
        elif mode == '2d1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
        elif mode == '2d2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
        elif mode == '1d1':
            dx = dx1[0,:,0]
            d2 = inv_s*dx**2
        elif mode == '1d2':
            dx = dx2[0,0,:]
            d2 = inv_s*dx**2

        return d2 < 1

    # def sphere_mask(self, x0, x1, x2, c, inv_S, mode='3d'):

    #     c0, c1, c2 = c

    #     dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

    #     # r = 1.2*np.max(1/np.sqrt(np.linalg.eigvalsh(inv_S)))
    #     r = np.cbrt(2*np.sqrt(1/np.linalg.det(inv_S)))

    #     if mode == '3d':
    #         inv_s = np.diag([1/r**2]*3)
    #     elif '2d' in mode:
    #         inv_s = np.diag([1/r**2]*2)
    #     elif '1d' in mode :
    #         inv_s = 1/r**2

    #     if mode == '3d':
    #         dx = [dx0, dx1, dx2]
    #         d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
    #     elif mode == '2d':
    #         dx = [dx1[0,:,:], dx2[0,:,:]]
    #         d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
    #     elif mode == '1d':
    #         dx = dx0[:,0,0]
    #         d2 = inv_s*dx**2
    #     elif mode == '2d1':
    #         dx = [dx0[:,0,:], dx2[:,0,:]]
    #         d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
    #     elif mode == '2d2':
    #         dx = [dx0[:,:,0], dx1[:,:,0]]
    #         d2 = np.einsum('i...,ij,j...->...', dx, inv_s, dx)
    #     elif mode == '1d1':
    #         dx = dx1[0,:,0]
    #         d2 = inv_s*dx**2
    #     elif mode == '1d2':
    #         dx = dx2[0,0,:]
    #         d2 = inv_s*dx**2

    #     return d2 < 1

    def ellipsoid_covariance(self, inv_S, mode='3d', perc=99.7):

        if mode == '3d':
            scale = scipy.stats.chi2.ppf(perc/100, df=3)
            inv_var = scale*inv_S
        elif mode == '2d':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[1:,1:]*scale
        elif mode == '1d':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[0,0]*scale
        elif mode == '2d1':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[0::2,0::2]*scale
        elif mode == '2d2':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[:2,:2]*scale
        elif mode == '1d1':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[1,1]*scale
        elif mode == '1d2':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[2,2]*scale

        return inv_var

    def gaussian(self, x0, x1, x2, A, B, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '2d':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '1d':
            dx = dx0[:,0,0]
            d2 = inv_var*dx**2
        elif mode == '2d1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '2d2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '1d1':
            dx = dx1[0,:,0]
            d2 = inv_var*dx**2
        elif mode == '1d2':
            dx = dx2[0,0,:]
            d2 = inv_var*dx**2

        return A*np.exp(-0.5*d2)+B

    def estimate_weights(self, x0, x1, x2, counts, y, e):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        y1, e1 = self.integrate(x0, x1, x2, counts, y, e, mode='1d')
        y2, e2 = self.integrate(x0, x1, x2, counts, y, e, mode='2d')
        y3, e3 = self.integrate(x0, x1, x2, counts, y, e, mode='3d')

        y11, e11 = self.integrate(x0, x1, x2, counts, y, e, mode='1d1')
        y12, e12 = self.integrate(x0, x1, x2, counts, y, e, mode='1d2')

        y21, e21 = self.integrate(x0, x1, x2, counts, y, e, mode='2d1')
        y22, e22 = self.integrate(x0, x1, x2, counts, y, e, mode='2d2')

        y1_min = np.nanmin(y1)
        y2_min = np.nanmin(y2)
        y3_min = np.nanmin(y3)

        y1_max = np.nanmax(y1)
        y2_max = np.nanmax(y2)
        y3_max = np.nanmax(y3)

        y11_min = np.nanmin(y11)
        y21_min = np.nanmin(y21)

        y12_min = np.nanmin(y12)
        y22_min = np.nanmin(y22)

        y11_max = np.nanmax(y11)
        y21_max = np.nanmax(y21)

        y12_max = np.nanmax(y12)
        y22_max = np.nanmax(y22)

        if np.isclose(y1_max, y1_min):
            return None

        if np.isclose(y2_max, y2_min):
            return None

        if np.isclose(y3_max, y3_min):
            return None

        if np.isclose(y11_max, y11_min):
            return None

        if np.isclose(y21_max, y21_min):
            return None

        if np.isclose(y12_max, y12_min):
            return None

        if np.isclose(y22_max, y22_min):
            return None

        w1 = y1-y1_min
        w11 = y11-y11_min
        w12 = y12-y12_min

        c0 = np.nansum(x0[:,0,0]*w1)/np.nansum(w1)
        c1 = np.nansum(x1[0,:,0]*w11)/np.nansum(w11)
        c2 = np.nansum(x2[0,0,:]*w12)/np.nansum(w12)

        r0 = 3*np.sqrt(np.nansum((x0[:,0,0]-c0)**2*w1)/np.nansum(w1))
        r1 = 3*np.sqrt(np.nansum((x1[0,:,0]-c1)**2*w11)/np.nansum(w11))
        r2 = 3*np.sqrt(np.nansum((x2[0,0,:]-c2)**2*w12)/np.nansum(w12))

        for param in ['c0', 'c1', 'c2', 'r0', 'r1', 'r2']:
            value = eval(param)
            if np.isfinite(value):
                self.params[param].set(value=value)

        # C1_max = (y1_max-y1_min)/dx0
        # C2_max = (y2_max-y2_min)/np.min([dx1,dx2])

        # self.params.add('C1', value=0, min=-5*C1_max, max=5*C1_max, vary=True)
        # self.params.add('C2', value=0, min=-5*C2_max, max=5*C2_max, vary=True)
        # self.params.add('C3', value=0, min=-5*C2_max, max=5*C2_max, vary=True)

        self.params.add('A1', value=y1_max, min=0, max=5*y1_max)
        self.params.add('A2', value=y2_max, min=0, max=5*y2_max)
        self.params.add('A3', value=y3_max, min=0, max=5*y3_max)

        # self.params.add('A11', value=y11_max, min=0, max=5*y11_max)
        # self.params.add('A21', value=y21_max, min=0, max=5*y21_max)

        # self.params.add('A12', value=y12_max, min=0, max=5*y12_max)
        # self.params.add('A22', value=y22_max, min=0, max=5*y22_max)

        self.params.add('B1', value=y1_min, min=-5*y1_max, max=5*y1_max)
        self.params.add('B2', value=y2_min, min=-5*y2_max, max=5*y2_max)
        self.params.add('B3', value=y3_min, min=-5*y3_max, max=5*y3_max)

        # self.params.add('B11', value=y11_min, min=-5*y11_max, max=5*y11_max)
        # self.params.add('B21', value=y21_min, min=-5*y21_max, max=5*y21_max)

        # self.params.add('B12', value=y12_min, min=-5*y12_max, max=5*y12_max)
        # self.params.add('B22', value=y22_min, min=-5*y22_max, max=5*y22_max)

        v1 = np.sqrt(1+y1**2)
        v2 = np.sqrt(1+y2**2)
        v3 = np.sqrt(1+y3**2)

        v11 = np.sqrt(1+y11**2)
        v21 = np.sqrt(1+y21**2)

        v12 = np.sqrt(1+y12**2)
        v22 = np.sqrt(1+y22**2)

        w1 = v1/e1/np.sqrt(e1.size)
        w2 = v2/e2/np.sqrt(e2.size)
        w3 = v3/e3/np.sqrt(e3.size)

        w11 = v11/e11/np.sqrt(e11.size)
        w21 = v21/e21/np.sqrt(e21.size)

        w12 = v12/e12/np.sqrt(e12.size)
        w22 = v22/e22/np.sqrt(e22.size)

        ys = (y1, y2, y3, y11, y12, y21, y22)
        es = (e1, e2, e3, e11, e12, e21, e22)
        vs = (v1, v2, v3, v11, v12, v21, v22)
        ws = (w1, w2, w3, w11, w12, w21, w22)

        args = [x0, x1, x2, ys, es, vs, ws]

        # ---

        self.params['A1'].set(value=y1_max)
        self.params['A2'].set(value=y2_max)
        self.params['A3'].set(value=y3_max)

        # self.params['A11'].set(value=y11_max)
        # self.params['A21'].set(value=y21_max)

        # self.params['A12'].set(value=y12_max)
        # self.params['A22'].set(value=y22_max)

        self.params['B1'].set(value=y1_min)
        self.params['B2'].set(value=y2_min)
        self.params['B3'].set(value=y3_min)

        # self.params['B11'].set(value=y11_min)
        # self.params['B21'].set(value=y21_min)

        # self.params['B12'].set(value=y12_min)
        # self.params['B22'].set(value=y22_min)

        self.params['c0'].set(vary=True)
        self.params['c1'].set(vary=True)
        self.params['c2'].set(vary=True)

        self.params['r0'].set(vary=True)
        self.params['r1'].set(vary=True)
        self.params['r2'].set(vary=True)

        self.params['u0'].set(vary=False)
        self.params['u1'].set(vary=False)
        self.params['u2'].set(vary=False)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=args,
                        nan_policy='omit')

        result = out.minimize(method='least_squares')

        self.params = result.params

        self.params['A1'].set(value=y1_max)
        self.params['A2'].set(value=y2_max)
        self.params['A3'].set(value=y3_max)

        # self.params['A11'].set(value=y11_max)
        # self.params['A21'].set(value=y21_max)

        # self.params['A12'].set(value=y12_max)
        # self.params['A22'].set(value=y22_max)

        self.params['B1'].set(value=y1_min)
        self.params['B2'].set(value=y2_min)
        self.params['B3'].set(value=y3_min)

        # self.params['B11'].set(value=y11_min)
        # self.params['B21'].set(value=y21_min)

        # self.params['B12'].set(value=y12_min)
        # self.params['B22'].set(value=y22_min)

        self.params['c0'].set(vary=False)
        self.params['c1'].set(vary=False)
        self.params['c2'].set(vary=False)

        self.params['r0'].set(vary=True)
        self.params['r1'].set(vary=True)
        self.params['r2'].set(vary=True)

        self.params['u0'].set(vary=True)
        self.params['u1'].set(vary=True)
        self.params['u2'].set(vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=args,
                        nan_policy='omit')

        result = out.minimize(method='least_squares')

        # self.params = result.params

        self.params['A1'].set(value=y1_max)
        self.params['A2'].set(value=y2_max)
        self.params['A3'].set(value=y3_max)

        # self.params['A11'].set(value=y11_max)
        # self.params['A21'].set(value=y21_max)

        # self.params['A12'].set(value=y12_max)
        # self.params['A22'].set(value=y22_max)

        self.params['B1'].set(value=y1_min)
        self.params['B2'].set(value=y2_min)
        self.params['B3'].set(value=y3_min)

        # self.params['B11'].set(value=y11_min)
        # self.params['B21'].set(value=y21_min)

        # self.params['B12'].set(value=y12_min)
        # self.params['B22'].set(value=y22_min)

        self.params['c0'].set(vary=True)
        self.params['c1'].set(vary=True)
        self.params['c2'].set(vary=True)

        self.params['r0'].set(vary=True)
        self.params['r1'].set(vary=True)
        self.params['r2'].set(vary=True)

        self.params['u0'].set(vary=True)
        self.params['u1'].set(vary=True)
        self.params['u2'].set(vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=args,
                        nan_policy='omit')

        result = out.minimize(method='least_squares')

        self.params = result.params

        c0 = self.params['c0'].value
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value

        r0 = self.params['r0'].value
        r1 = self.params['r1'].value
        r2 = self.params['r2'].value

        u0 = self.params['u0'].value
        u1 = self.params['u1'].value
        u2 = self.params['u2'].value

        phi, theta, omega = self.angles(u0, u1, u2)

        # C1 = self.params['C1'].value
        # C2 = self.params['C2'].value
        # C3 = self.params['C3'].value

        B1 = self.params['B1'].value
        B2 = self.params['B2'].value
        B3 = self.params['B3'].value

        A1 = self.params['A1'].value
        A2 = self.params['A2'].value
        A3 = self.params['A3'].value

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    phi, theta, omega)

        args = x0, x1, x2, 1, 0, c, inv_S

        y1_gauss = self.gaussian(*args, '1d')
        y2_gauss = self.gaussian(*args, '2d')
        y3_gauss = self.gaussian(*args, '3d')

        y1_fit = A1*y1_gauss+B1#+C1*x0[:,0,0]
        y2_fit = A2*y2_gauss+B2#+C2*x1[0,:,:]+C3*x2[0,:,:]
        y3_fit = A3*y3_gauss+B3

        self.redchi2 = np.nanmean((y1_fit-y1)**2/e1**2),\
                       np.nanmean((y2_fit-y2)**2/e2**2),\
                       np.nanmean((y3_fit-y3)**2/e3**2)

        self.error_scale = np.sqrt(self.redchi2[2])

        inv_S = self.inv_S_matrix(r0, r1, r2, phi, theta, omega)

        return c, inv_S, (y1_fit, y1, e1), (y2_fit, y2, e2), (y3_fit, y3, e3)

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def fit(self, x0, x1, x2, c, y_norm, e_norm, dx):

        counts = c.copy()
        y = y_norm.copy()
        e = e_norm.copy()

        self.update_constraints(x0, x1, x2, dx)

        mask = (counts > 0) & (e > 0) & np.isfinite(counts) & np.isfinite(e)

        w = 0*counts.copy()+1
        v = e**2

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        scale = np.sqrt(scipy.stats.chi2.ppf(99.7/100, df=3))

        sigma = np.floor(dx/np.array([dx0, dx1, dx2])/scale).astype(int)+1

        counts[~mask] = 0
        counts = scipy.ndimage.gaussian_filter(counts, sigma=sigma)

        y[~mask] = 0
        y = scipy.ndimage.gaussian_filter(y, sigma=sigma)

        v[~mask] = 0
        v = scipy.ndimage.gaussian_filter(v, sigma=sigma)

        w[~mask] = 0
        w = scipy.ndimage.gaussian_filter(w, sigma=sigma)

        counts /= w
        y /= w
        v /= w

        e = np.sqrt(v)

        counts[~mask] = np.nan
        y[~mask] = np.nan
        e[~mask] = np.nan

        y_max = np.nanmax(y)

        if mask.sum() < 20 or (np.array(mask.shape) <= 5).any() or y_max <= 0:
            return None

        coords = np.argwhere(mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0)+1

        y = y[i0:j0,i1:j1,i2:j2].copy()
        e = e[i0:j0,i1:j1,i2:j2].copy()
        counts = counts[i0:j0,i1:j1,i2:j2].copy()

        # y_bin = y_bin[i0:j0,i1:j1,i2:j2].copy()
        # e_bin = e_bin[i0:j0,i1:j1,i2:j2].copy()

        if (np.array(y.shape) <= 3).any():
            return None

        x0 = x0[i0:j0,i1:j1,i2:j2].copy()
        x1 = x1[i0:j0,i1:j1,i2:j2].copy()
        x2 = x2[i0:j0,i1:j1,i2:j2].copy()

        # self.counts = self.counts[i0:j0,i1:j1,i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print('Invalid data')
            return None

        # size = np.floor(dx/np.array([dx0, dx1, dx2])).astype(int)+1

        # d_val = scipy.ndimage.median_filter(d_val, size=size, mode='nearest')
        # n_val = scipy.ndimage.median_filter(n_val, size=size, mode='nearest')

        weights = self.estimate_weights(x0, x1, x2, counts, y, e)

        if weights is None:
            print('Invalid weight estimate')
            return None

        c, inv_S, vals1d, vals2d, vals3d = weights

        y_prof_fit, y_prof, e_prof = vals1d

        y_proj_fit, y_proj, e_proj = vals2d

        y_fit, y, e = vals3d

        if not np.linalg.det(inv_S) > 0:
            print('Improper optimal covariance')
            return None

        S = np.linalg.inv(inv_S)

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        dxv = [dx0, dx1, dx2]

        threshold = np.einsum('i...,ij,j...->...', dxv, inv_S, dxv) <= 1

        if threshold.sum() < 13:
            print('Low counts')
            return None

        V, W = np.linalg.eigh(S)

        c0, c1, c2 = c

        r0, r1, r2 = np.sqrt(V)

        v0, v1, v2 = W.T

        binning = (x0, x1, x2), y, e

        fitting = binning, y_fit

        self.best_fit = c, S, *fitting

        self.best_prof = (x0[:,0,0], y_prof, e_prof), y_prof_fit

        self.best_proj = (x1[0,:,:], x2[0,:,:], y_proj, e_proj), y_proj_fit

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_norm(self, x0, x1, x2, y, e, counts, c, S):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        # r = 1.2*np.max(np.sqrt(np.linalg.eigvalsh(S)))
        # r = np.cbrt(2*np.sqrt(np.linalg.det(S)))

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)
        # sphere = np.einsum('ij,jklm,iklm->klm', np.diag([1/r**2]*3), x, x)

        pk = (ellipsoid <= 1.1**2) & (e > 0)
        bkg = (ellipsoid > 1.1**2) & (ellipsoid < 2**2) & (e > 0)

        d3x = dx0*dx1*dx2

        y_pk = y[pk].copy()
        e_pk = e[pk].copy()

        y_bkg = y[bkg].copy()
        e_bkg = e[bkg].copy()

        b = np.nanmean(y_bkg)
        b_err = np.sqrt(np.nanmean(e_bkg**2))

        intens = np.nansum(y_pk-b)
        sig = np.sqrt(np.nansum(e_pk**2+b_err**2))

        # *(1+self.error_scale**2)

        self.weights = (x0[pk], x1[pk], x2[pk]), counts[pk].copy()

        self.info = [d3x, b, b_err]

        freq = y-b
        freq[~(pk | bkg)] = np.nan

        c_pk = counts[pk].copy()
        c_bkg = counts[bkg].copy()

        b_raw = np.nanmean(c_bkg)
        b_raw_err = np.sqrt(np.nanmean(c_bkg))

        intens_raw = np.nansum(c_pk-b_raw)
        sig_raw = np.sqrt(np.nansum(c_pk+b_raw_err**2))

        self.info += [intens_raw, sig_raw]

        if not np.isfinite(sig):
            sig = intens

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        return intens, sig