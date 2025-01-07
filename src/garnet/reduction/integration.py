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

from garnet.plots.peaks import PeakPlot
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

    @staticmethod
    def integrate_parallel(plan, runs, proc):

        plan['Runs'] = runs
        plan['ProcName'] = '_p{}'.format(proc)

        data = DataModel(beamlines[plan['Instrument']])

        instance = Integration(plan)
        instance.proc = proc
        instance.n_proc = 1

        if data.laue:
            return instance.laue_integrate()
        else:
            return instance.monochromatic_integrate()

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

        self.make_plot = True
        self.peak_plot = PeakPlot()

        runs = self.plan['Runs']

        self.run = 0
        self.runs = len(runs)

        for run in runs:

            self.run += 1

            print('{}: {:}/{:}'.format(self.proc, self.run, len(runs)))

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

            data.load_background(self.plan.get('BackgroundFile'), 'data')

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

            # peaks.integrate_peaks('md', 'peaks', r_cut)

            # peaks.remove_weak_peaks('peaks', 10)

            self.peaks, self.data = peaks, data

            params = self.estimate_peak_size('peaks', 'md', r_cut)

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

            self.update_peak_info('peaks', results)

            # peaks.remove_weak_peaks('peaks')

            peaks.combine_peaks('peaks', 'combine')

            pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))

            peaks.save_peaks(pk_file, 'peaks')

            data.delete_workspace('peaks')

            data.delete_workspace('md')

        # result_file = self.get_file(output_file, '')

        peaks.save_peaks(output_file, 'combine')

        # ---

        if mtd.doesExist('combine'):

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(output_file)[0]+'.mat'

            ub = UBModel('combine')
            ub.save_UB(ub_file)

        mtd.clear()

        return output_file

    # def laue_combine(self, files):

    #     output_file = self.get_output_file()
    #     result_file = self.get_file(output_file, '')

    #     peaks = PeaksModel()

    #     for file in files:

    #         peaks.load_peaks(file, 'tmp')
    #         peaks.combine_peaks('tmp', 'combine')

    #     for file in files:
    #         os.remove(file)

    #     if mtd.doesExist('combine'):

    #         peaks.save_peaks(result_file, 'combine')

    #         opt = Optimization('combine')
    #         opt.optimize_lattice(self.params['Cell'])

    #         ub_file = os.path.splitext(result_file)[0]+'.mat'

    #         ub = UBModel('combine')
    #         ub.save_UB(ub_file)

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

        params = self.peaks.intensity_vs_radius(data_ws, peaks_ws, r_cut)

        r, sig_noise, x, y, e, lamda = params

        sphere = PeakSphere(r_cut)

        r_cut = sphere.fit(r, sig_noise)

        sig_noise_fit, *vals = sphere.best_fit(r)

        # values = self.peaks.extract_peaks_roi(data_ws, peaks_ws, r_cut)

        roi = PeakRegionOfInterest(r_cut)

        r = roi.fit(x, y, e, lamda)

        return r

    def fit_peaks(self, key_value):

        key, value = key_value

        data_info, peak_info = value

        Q0, Q1, Q2, counts, y, e, dQ, Qmod, projections = data_info

        peak_name, wavelength, angles, goniometer = peak_info
        # print(key, peak_name)

        ellipsoid = PeakEllipsoid()

        params = ellipsoid.fit(Q0, Q1, Q2, counts, y, e, dQ, Qmod)

        value = None

        if params is not None:

            c, S, *fitting = ellipsoid.best_fit

            shape = self.revert_ellipsoid_parameters(params, projections)

            norm_params = Q0, Q1, Q2, y, e, counts, c, S

            I, sigma = ellipsoid.integrate(*norm_params)

            if self.make_plot:

                self.peak_plot.add_fitting(*fitting)

                self.peak_plot.add_profile_fit(*ellipsoid.best_prof)

                self.peak_plot.add_projection_fit(*ellipsoid.best_proj)

                self.peak_plot.add_ellipsoid(c, S)

                self.peak_plot.add_peak_info(wavelength, angles, goniometer)

                self.peak_plot.add_peak_stats(ellipsoid.redchi2)

                self.peak_plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

                self.peak_plot.save_plot(self.get_plot_file(peak_name))

            value = I, sigma, shape, ellipsoid.info

        return key, value

    def extract_peak_info(self, peaks_ws, r):
        """
        Obtain peak information for envelope determination.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r : list
            Cutoff radius parameters.

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        UB = self.peaks.get_UB(peaks_ws)

        peak_dict = {}

        for i in range(n_peak):
            # print(i)

            Qmod = 2*np.pi/peak.get_d_spacing(i)

            h, k, l = peak.get_hkl(i)

            wavelength = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            params = peak.get_peak_shape(i, r[0])

            peak.set_peak_intensity(i, 0, 0)

            goniometer = peak.get_goniometer_angles(i)

            peak_name = peak.get_peak_name(i)

            # det_id = peak.get_detector_id(i)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            R = peak.get_goniometer_matrix(i)

            bin_params = r, wavelength, dQ, R, two_theta, az_phi, UB

            # ---

            bins, extents, projections = self.bin_extent(*params, *bin_params)

            y, e, Q0, Q1, Q2 = data.bin_in_Q('md', extents, bins, projections)

            counts = data.extract_counts('md_bin')

            data_info = (Q0, Q1, Q2, counts, y, e, dQ, Qmod, projections)

            peak_info = (peak_name, wavelength, angles, goniometer)

            peak_dict[i] = data_info, peak_info

        return peak_dict

    def update_peak_info(self, peaks_ws, peak_dict):

        peak = PeakModel(peaks_ws)

        for i, value in peak_dict.items():

            if value is not None:

                I, sigma, shape, info = value
                # print(i, I, sigma)

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
                         v0, v1, v2, roi, lamda, bin_size,
                         R, two_theta, az_phi, UB):

        n, u, v = self.bin_axes(R, two_theta, az_phi)

        projections = [n, u, v]

        params = Q0, Q1, Q2, r0, r1, r2, v0, v1, v2

        params = self.project_ellipsoid_parameters(params, projections)

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        r = roi[0]+roi[1]*lamda

        dQ = 2*np.array([r]*3)

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

        bin_sizes = np.array(dQ)/15
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


class PeakRegionOfInterest:

    def __init__(self, r_cut):

        self.params = Parameters()

        self.params.add('r0', value=r_cut/2, min=0.001, max=2*r_cut)
        self.params.add('r1', value=0, min=-r_cut, max=r_cut)

    def objective(self, params, x, y, e, lamda):

        r0 = params['r0']
        r1 = params['r1']

        sigma = (r0+r1*lamda[:,np.newaxis])/3.76205

        z = x/sigma

        y_hat = scipy.special.erf(z/np.sqrt(2))\
              - np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2)

        num = np.nansum(y_hat*y, axis=1)
        den = np.nansum(y_hat**2, axis=1)
        # wgt = np.nanmax(y, axis=1)

        A = num/den

        residuals = A[:,np.newaxis]*y_hat-y

        return residuals#*wgt[:,np.newaxis]

        # cost = np.nansum((weight*residuals**2)/(fit.size-2)*wgt)

    def fit(self, x, y, e, lamda):

        if np.max(lamda)-np.min(lamda) < 0.2:
            self.params['r1'].set(vary=False)

        out = Minimizer(self.objective,
                        self.params,
                        fcn_args=(x, y, e, lamda),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        return result.params['r0'].value, result.params['r1'].value


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

        return 3.76205*result.params['sigma'].value

    def best_fit(self, r):

        A = self.params['A'].value
        sigma = self.params['sigma'].value

        return self.model(r, A, sigma), A, sigma

class PeakEllipsoid:

    def __init__(self):

        self.params = Parameters()

    def update_constraints(self, x0, x1, x2, dx):

        r0 = (x0[:,0,0][-1]-x0[:,0,0][0])/8
        r1 = (x1[0,:,0][-1]-x1[0,:,0][0])/8
        r2 = (x2[0,0,:][-1]-x2[0,0,:][0])/8

        r0_max = (x0[:,0,0][-1]-x0[:,0,0][0])/2
        r1_max = (x1[0,:,0][-1]-x1[0,:,0][0])/2
        r2_max = (x2[0,0,:][-1]-x2[0,0,:][0])/2

        c0 = (x0[:,0,0][-1]+x0[:,0,0][0])/2
        c1 = (x1[0,:,0][-1]+x1[0,:,0][0])/2
        c2 = (x2[0,0,:][-1]+x2[0,0,:][0])/2

        c0_min, c1_min, c2_min = c0-r0_max/2, c1-r1_max/2, c2-r2_max/2
        c0_max, c1_max, c2_max = c0+r0_max/2, c1+r1_max/2, c2+r2_max/2

        self.params.add('c0', value=c0, min=c0_min, max=c0_max)
        self.params.add('c1', value=c1, min=c1_min, max=c1_max)
        self.params.add('c2', value=c2, min=c2_min, max=c2_max)

        self.params.add('r0', value=r0, min=dx, max=r0_max)
        self.params.add('r1', value=r1, min=dx, max=r1_max)
        self.params.add('r2', value=r2, min=dx, max=r2_max)

        self.params.add('u0', value=0.0, min=-np.pi/6, max=np.pi/6)
        self.params.add('u1', value=0.0, min=-np.pi/6, max=np.pi/6)
        self.params.add('u2', value=0.0, min=-np.pi/6, max=np.pi/6)

    def S_matrix(self, r0, r1, r2, u0, u1, u2):

        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):

        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1/r0**2, 1/r1**2, 1/r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):

        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):

        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S

    def normalize(self, x0, x1, x2, counts, y, e, mode='3d'):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == '1d_0':
            c_int = dx0*np.mean(counts > 0, axis=(1,2))
            y_int = np.nansum(y, axis=(1,2))/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=(1,2)))/c_int
        elif mode == '1d_1':
            c_int = dx1*np.mean(counts > 0, axis=(0,2))
            y_int = np.nansum(y, axis=(0,2))/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=(0,2)))/c_int
        elif mode == '1d_2':
            c_int = dx2*np.mean(counts > 0, axis=(0,1))
            y_int = np.nansum(y, axis=(0,1))/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=(0,1)))/c_int
        elif mode == '2d_0':
            c_int = dx1*dx2*np.mean(counts > 0, axis=0)
            y_int = np.nansum(y, axis=0)/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=0))/c_int
        elif mode == '2d_1':
            c_int = dx0*dx2*np.mean(counts > 0, axis=1)
            y_int = np.nansum(y, axis=1)/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=1))/c_int
        elif mode == '2d_2':
            c_int = dx0*dx1*np.mean(counts > 0, axis=2)
            y_int = np.nansum(y, axis=2)/c_int
            e_int = np.sqrt(np.nansum(e**2, axis=2))/c_int
        elif mode == '3d':
            c_int = dx0*dx1*dx2
            y_int = y.copy()/c_int
            e_int = e.copy()/c_int

        mask = (y_int > 0) & np.isfinite(y_int) \
             & (e_int > 0) & np.isfinite(e_int)

        y_int[~mask] = np.nan
        e_int[~mask] = np.nan

        return y_int, e_int

    def ellipsoid_covariance(self, inv_S, mode='3d', perc=99.7):

        if mode == '3d':
            scale = scipy.stats.chi2.ppf(perc/100, df=3)
            inv_var = inv_S*scale
        elif mode == '2d_0':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[1:,1:]*scale
        elif mode == '2d_1':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[0::2,0::2]*scale
        elif mode == '2d_2':
            scale = scipy.stats.chi2.ppf(perc/100, df=2)
            inv_var = inv_S[:2,:2]*scale
        elif mode == '1d_0':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[0,0]*scale
        elif mode == '1d_1':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[1,1]*scale
        elif mode == '1d_2':
            scale = scipy.stats.chi2.ppf(perc/100, df=1)
            inv_var = inv_S[2,2]*scale

        return inv_var

    def chi_2_fit(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S, dx)
            m = 11
        elif mode == '2d_0':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S[1:,1:], dx)
            m = 7
        elif mode == '2d_1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S[0::2,0::2], dx)
            m = 9
        elif mode == '2d_2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S[:2,:2], dx)
            m = 9
        elif mode == '1d_0':
            dx = dx0[:,0,0]
            d2 = inv_S[0,0]*dx**2
            m = 5
        elif mode == '1d_1':
            dx = dx1[0,:,0]
            d2 = inv_S[1,1]*dx**2
            m = 5
        elif mode == '1d_2':
            dx = dx2[0,0,:]
            d2 = inv_S[2,2]*dx**2
            m = 5

        mask = (d2 < 2**2) & np.isfinite(y) & (e > 0)

        n = np.sum(mask)

        if n <= m:
            return np.inf
        else:
            return np.nansum(((y_fit[mask]-y[mask])/e[mask])**2)/(n-m)

    def gaussian(self, x0, x1, x2, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '2d_0':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '2d_1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '2d_2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
        elif mode == '1d_0':
            dx = dx0[:,0,0]
            d2 = inv_var*dx**2
        elif mode == '1d_1':
            dx = dx1[0,:,0]
            d2 = inv_var*dx**2
        elif mode == '1d_2':
            dx = dx2[0,0,:]
            d2 = inv_var*dx**2

        return np.exp(-0.5*d2)

    def inv_S_deriv_r(self, r0, r1, r2, u0, u1, u2):

        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2/r0**3,0,0]) @ U.T
        dinv_S1 = U @ np.diag([0,-2/r1**3,0]) @ U.T
        dinv_S2 = U @ np.diag([0,0,-2/r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_u(self, r0, r1, r2, u0, u1, u2):

        V = np.diag([1/r0**2, 1/r1**2, 1/r2**2])

        U = self.U_matrix(u0, u1, u2)
        dU0, dU1, dU2 = self.U_deriv_u(u0, u1, u2)

        dinv_S0 = dU0 @ V @ U.T + U @ V @ dU0.T
        dinv_S1 = dU1 @ V @ U.T + U @ V @ dU1.T
        dinv_S2 = dU2 @ V @ U.T + U @ V @ dU2.T

        return dinv_S0, dinv_S1, dinv_S2
    
    def U_deriv_u(self, u0, u1, u2, delta=1e-6):

        dU0 = self.U_matrix(u0+delta, u1, u2)-self.U_matrix(u0-delta, u1, u2)
        dU1 = self.U_matrix(u0, u1+delta, u2)-self.U_matrix(u0, u1-delta, u2)
        dU2 = self.U_matrix(u0, u1, u2+delta)-self.U_matrix(u0, u1, u2-delta)

        return 0.5*dU0/delta, 0.5*dU1/delta, 0.5*dU2/delta

    def gaussian_jac_c(self, x0, x1, x2, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0, g1, g2 = np.einsum('ij,j...->i...', inv_var, dx)
        elif mode == '2d_0':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g1, g2 = np.einsum('ij,j...->i...', inv_var, dx)
            g0 = g1*0
        elif mode == '2d_1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0, g2 = np.einsum('ij,j...->i...', inv_var, dx)
            g1 = g2*0
        elif mode == '2d_2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0, g1 = np.einsum('ij,j...->i...', inv_var, dx)
            g2 = g0*0
        elif mode == '1d_0':
            dx = dx0[:,0,0]
            d2 = inv_var*dx**2
            g0 = inv_var*dx
            g1 = g2 = g0*0
        elif mode == '1d_1':
            dx = dx1[0,:,0]
            d2 = inv_var*dx**2
            g1 = inv_var*dx
            g2 = g0 = g1*0
        elif mode == '1d_2':
            dx = dx2[0,0,:]
            d2 = inv_var*dx**2
            g2 = inv_var*dx
            g0 = g1 = g2*0

        g = np.exp(-0.5*d2)

        return g*np.array([g0, g1, g2])

    def gaussian_jac_S(self, x0, x1, x2, c, inv_S, d_inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0 = np.einsum('i...,ij,j...->...', dx, d_inv_var[0], dx)
            g1 = np.einsum('i...,ij,j...->...', dx, d_inv_var[1], dx)
            g2 = np.einsum('i...,ij,j...->...', dx, d_inv_var[2], dx)
        elif mode == '2d_0':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g1 = np.einsum('i...,ij,j...->...', dx, d_inv_var[1], dx)
            g2 = np.einsum('i...,ij,j...->...', dx, d_inv_var[2], dx)
            g0 = g1*0
        elif mode == '2d_1':
            dx = [dx0[:,0,:], dx2[:,0,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0 = np.einsum('i...,ij,j...->...', dx, d_inv_var[0], dx)
            g2 = np.einsum('i...,ij,j...->...', dx, d_inv_var[2], dx)
            g1 = g2*0
        elif mode == '2d_2':
            dx = [dx0[:,:,0], dx1[:,:,0]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_var, dx)
            g0 = np.einsum('i...,ij,j...->...', dx, d_inv_var[0], dx)
            g1 = np.einsum('i...,ij,j...->...', dx, d_inv_var[1], dx)
            g2 = g0*0
        elif mode == '1d_0':
            dx = dx0[:,0,0]
            d2 = inv_var*dx**2
            g0 = d_inv_var[0]*dx**2
            g1 = g2 = g0*0
        elif mode == '1d_1':
            dx = dx1[0,:,0]
            d2 = inv_var*dx**2
            g1 = d_inv_var[1]*dx**2
            g2 = g0 = g1*0
        elif mode == '1d_2':
            dx = dx2[0,0,:]
            d2 = inv_var*dx**2
            g2 = d_inv_var[2]*dx**2
            g0 = g1 = g2*0

        g = np.exp(-0.5*d2)

        return -0.5*g*np.array([g0, g1, g2])

    def residual_1d(self, params, x0, x1, x2, ys, es):

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A0 = params['A1d_0']
        A1 = params['A1d_1']
        A2 = params['A1d_2']

        B0 = params['B1d_0']
        B1 = params['B1d_1']
        B2 = params['B1d_2']

        C0 = params['C1d_0']
        C1 = params['C1d_1']
        C2 = params['C1d_2']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, '1d_0')
        y1_gauss = self.gaussian(*args, '1d_1')
        y2_gauss = self.gaussian(*args, '1d_2')

        diff = []

        y0_fit = A0*y0_gauss+B0+C0*(x0[:,0,0]-c0)
        y1_fit = A1*y1_gauss+B1+C1*(x1[0,:,0]-c1)
        y2_fit = A2*y2_gauss+B2+C2*(x2[0,0,:]-c2)

        res = (y0_fit-y0)/e0

        diff += res.flatten().tolist()

        res = (y1_fit-y1)/e1

        diff += res.flatten().tolist()

        res = (y2_fit-y2)/e2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_1d(self, params, x0, x1, x2, ys, es):

        params_list = list(params.keys())

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A0 = params['A1d_0']
        A1 = params['A1d_1']
        A2 = params['A1d_2']

        # B0 = params['B1d_0']
        # B1 = params['B1d_1']
        # B2 = params['B1d_2']

        C0 = params['C1d_0']
        C1 = params['C1d_1']
        C2 = params['C1d_2']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, '1d_0')
        y1_gauss = self.gaussian(*args, '1d_1')
        y2_gauss = self.gaussian(*args, '1d_2')

        dA0 = y0_gauss/e0
        dA1 = y1_gauss/e1
        dA2 = y2_gauss/e2

        dB0 = 1/e0
        dB1 = 1/e1
        dB2 = 1/e2

        dC0 = (x0[:,0,0]-c0)/e0
        dC1 = (x1[0,:,0]-c1)/e1
        dC2 = (x2[0,0,:]-c2)/e2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='1d_0')
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='1d_1')
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='1d_2')

        dc0_0, dc1_0, dc2_0 = A0*yc0_gauss/e0
        dc0_1, dc1_1, dc2_1 = A1*yc1_gauss/e1
        dc0_2, dc1_2, dc2_2 = A2*yc2_gauss/e2

        dc0_0 -= C0/e0
        dc1_1 -= C1/e1
        dc2_2 -= C2/e2

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='1d_0')
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='1d_1')
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='1d_2')

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='1d_0')
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='1d_1')
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='1d_2')

        dr0_0, dr1_0, dr2_0 = A0*yr0_gauss/e0
        dr0_1, dr1_1, dr2_1 = A1*yr1_gauss/e1
        dr0_2, dr1_2, dr2_2 = A2*yr2_gauss/e2

        du0_0, du1_0, du2_0 = A0*yu0_gauss/e0
        du0_1, du1_1, du2_1 = A1*yu1_gauss/e1
        du0_2, du1_2, du2_2 = A2*yu2_gauss/e2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)
        jac = np.zeros((n_params,n0+n1+n2))

        jac[params_list.index('A1d_0'),:n0] = dA0.flatten()
        jac[params_list.index('B1d_0'),:n0] = dB0.flatten()
        jac[params_list.index('C1d_0'),:n0] = dC0.flatten()
        jac[params_list.index('c0'),:n0] = dc0_0.flatten()
        jac[params_list.index('c1'),:n0] = dc1_0.flatten()
        jac[params_list.index('c2'),:n0] = dc2_0.flatten()
        jac[params_list.index('r0'),:n0] = dr0_0.flatten()
        jac[params_list.index('r1'),:n0] = dr1_0.flatten()
        jac[params_list.index('r2'),:n0] = dr2_0.flatten()
        jac[params_list.index('u0'),:n0] = du0_0.flatten()
        jac[params_list.index('u1'),:n0] = du1_0.flatten()
        jac[params_list.index('u2'),:n0] = du2_0.flatten()

        jac[params_list.index('A1d_1'),n0:n0+n1] = dA1.flatten()
        jac[params_list.index('B1d_1'),n0:n0+n1] = dB1.flatten()
        jac[params_list.index('C1d_1'),n0:n0+n1] = dC1.flatten()
        jac[params_list.index('c0'),n0:n0+n1] = dc0_1.flatten()
        jac[params_list.index('c1'),n0:n0+n1] = dc1_1.flatten()
        jac[params_list.index('c2'),n0:n0+n1] = dc2_1.flatten()
        jac[params_list.index('r0'),n0:n0+n1] = dr0_1.flatten()
        jac[params_list.index('r1'),n0:n0+n1] = dr1_1.flatten()
        jac[params_list.index('r2'),n0:n0+n1] = dr2_1.flatten()
        jac[params_list.index('u0'),n0:n0+n1] = du0_1.flatten()
        jac[params_list.index('u1'),n0:n0+n1] = du1_1.flatten()
        jac[params_list.index('u2'),n0:n0+n1] = du2_1.flatten()

        jac[params_list.index('A1d_2'),n0+n1:n0+n1+n2] = dA2.flatten()
        jac[params_list.index('B1d_2'),n0+n1:n0+n1+n2] = dB2.flatten()
        jac[params_list.index('C1d_2'),n0+n1:n0+n1+n2] = dC2.flatten()
        jac[params_list.index('c0'),n0+n1:n0+n1+n2] = dc0_2.flatten()
        jac[params_list.index('c1'),n0+n1:n0+n1+n2] = dc1_2.flatten()
        jac[params_list.index('c2'),n0+n1:n0+n1+n2] = dc2_2.flatten()
        jac[params_list.index('r0'),n0+n1:n0+n1+n2] = dr0_2.flatten()
        jac[params_list.index('r1'),n0+n1:n0+n1+n2] = dr1_2.flatten()
        jac[params_list.index('r2'),n0+n1:n0+n1+n2] = dr2_2.flatten()
        jac[params_list.index('u0'),n0+n1:n0+n1+n2] = du0_2.flatten()
        jac[params_list.index('u1'),n0+n1:n0+n1+n2] = du1_2.flatten()
        jac[params_list.index('u2'),n0+n1:n0+n1+n2] = du2_2.flatten()

        # ---

        diff = np.concatenate([e.flatten() for e in es])

        mask = np.isfinite(diff)

        return jac[:,mask]

    def residual_2d(self, params, x0, x1, x2, ys, es):

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A0 = params['A2d_0']
        A1 = params['A2d_1']
        A2 = params['A2d_2']

        B0 = params['B2d_0']
        B1 = params['B2d_1']
        B2 = params['B2d_2']

        C01 = params['C2d_01']
        C02 = params['C2d_02']

        C10 = params['C2d_10']
        C12 = params['C2d_12']

        C20 = params['C2d_20']
        C21 = params['C2d_21']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, '2d_0')
        y1_gauss = self.gaussian(*args, '2d_1')
        y2_gauss = self.gaussian(*args, '2d_2')

        diff = []

        y0_fit = A0*y0_gauss+B0+C01*(x1[0,:,:]-c1)+C02*(x2[0,:,:]-c2)
        y1_fit = A1*y1_gauss+B1+C10*(x0[:,0,:]-c0)+C12*(x2[:,0,:]-c2)
        y2_fit = A2*y2_gauss+B2+C20*(x0[:,:,0]-c0)+C21*(x1[:,:,0]-c1)

        res = (y0_fit-y0)/e0

        diff += res.flatten().tolist()

        res = (y1_fit-y1)/e1

        diff += res.flatten().tolist()

        res = (y2_fit-y2)/e2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_2d(self, params, x0, x1, x2, ys, es):

        params_list = list(params.keys())

        y0, y1, y2 = ys
        e0, e1, e2 = es

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A0 = params['A2d_0']
        A1 = params['A2d_1']
        A2 = params['A2d_2']

        # B0 = params['B2d_0']
        # B1 = params['B2d_1']
        # B2 = params['B2d_2']

        C01 = params['C2d_01']
        C02 = params['C2d_02']

        C10 = params['C2d_10']
        C12 = params['C2d_12']

        C20 = params['C2d_20']
        C21 = params['C2d_21']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, '2d_0')
        y1_gauss = self.gaussian(*args, '2d_1')
        y2_gauss = self.gaussian(*args, '2d_2')

        dA0 = y0_gauss/e0
        dA1 = y1_gauss/e1
        dA2 = y2_gauss/e2

        dB0 = 1/e0
        dB1 = 1/e1
        dB2 = 1/e2

        dC01 = (x1[0,:,:]-c1)/e0
        dC02 = (x2[0,:,:]-c2)/e0

        dC10 = (x0[:,0,:]-c0)/e1
        dC12 = (x2[:,0,:]-c2)/e1

        dC20 = (x0[:,:,0]-c0)/e2
        dC21 = (x1[:,:,0]-c1)/e2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='2d_0')
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='2d_1')
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='2d_2')

        dc0_0, dc1_0, dc2_0 = A0*yc0_gauss/e0
        dc0_1, dc1_1, dc2_1 = A1*yc1_gauss/e1
        dc0_2, dc1_2, dc2_2 = A2*yc2_gauss/e2

        dc1_0 -= C01/e0
        dc2_0 -= C02/e0

        dc0_1 -= C10/e1
        dc2_1 -= C12/e1

        dc0_2 -= C20/e2
        dc1_2 -= C21/e2

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='2d_0')
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='2d_1')
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='2d_2')

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='2d_0')
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='2d_1')
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='2d_2')

        dr0_0, dr1_0, dr2_0 = A0*yr0_gauss/e0
        dr0_1, dr1_1, dr2_1 = A1*yr1_gauss/e1
        dr0_2, dr1_2, dr2_2 = A2*yr2_gauss/e2

        du0_0, du1_0, du2_0 = A0*yu0_gauss/e0
        du0_1, du1_1, du2_1 = A1*yu1_gauss/e1
        du0_2, du1_2, du2_2 = A2*yu2_gauss/e2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)
        jac = np.zeros((n_params,n0+n1+n2))

        jac[params_list.index('A2d_0'),:n0] = dA0.flatten()
        jac[params_list.index('B2d_0'),:n0] = dB0.flatten()
        jac[params_list.index('C2d_01'),:n0] = dC01.flatten()
        jac[params_list.index('C2d_02'),:n0] = dC02.flatten()
        jac[params_list.index('c0'),:n0] = dc0_0.flatten()
        jac[params_list.index('c1'),:n0] = dc1_0.flatten()
        jac[params_list.index('c2'),:n0] = dc2_0.flatten()
        jac[params_list.index('r0'),:n0] = dr0_0.flatten()
        jac[params_list.index('r1'),:n0] = dr1_0.flatten()
        jac[params_list.index('r2'),:n0] = dr2_0.flatten()
        jac[params_list.index('u0'),:n0] = du0_0.flatten()
        jac[params_list.index('u1'),:n0] = du1_0.flatten()
        jac[params_list.index('u2'),:n0] = du2_0.flatten()

        jac[params_list.index('A2d_1'),n0:n0+n1] = dA1.flatten()
        jac[params_list.index('B2d_1'),n0:n0+n1] = dB1.flatten()
        jac[params_list.index('C2d_10'),n0:n0+n1] = dC10.flatten()
        jac[params_list.index('C2d_12'),n0:n0+n1] = dC12.flatten()
        jac[params_list.index('c0'),n0:n0+n1] = dc0_1.flatten()
        jac[params_list.index('c1'),n0:n0+n1] = dc1_1.flatten()
        jac[params_list.index('c2'),n0:n0+n1] = dc2_1.flatten()
        jac[params_list.index('r0'),n0:n0+n1] = dr0_1.flatten()
        jac[params_list.index('r1'),n0:n0+n1] = dr1_1.flatten()
        jac[params_list.index('r2'),n0:n0+n1] = dr2_1.flatten()
        jac[params_list.index('u0'),n0:n0+n1] = du0_1.flatten()
        jac[params_list.index('u1'),n0:n0+n1] = du1_1.flatten()
        jac[params_list.index('u2'),n0:n0+n1] = du2_1.flatten()

        jac[params_list.index('A2d_2'),n0+n1:n0+n1+n2] = dA2.flatten()
        jac[params_list.index('B2d_2'),n0+n1:n0+n1+n2] = dB2.flatten()
        jac[params_list.index('C2d_20'),n0+n1:n0+n1+n2] = dC20.flatten()
        jac[params_list.index('C2d_21'),n0+n1:n0+n1+n2] = dC21.flatten()
        jac[params_list.index('c0'),n0+n1:n0+n1+n2] = dc0_2.flatten()
        jac[params_list.index('c1'),n0+n1:n0+n1+n2] = dc1_2.flatten()
        jac[params_list.index('c2'),n0+n1:n0+n1+n2] = dc2_2.flatten()
        jac[params_list.index('r0'),n0+n1:n0+n1+n2] = dr0_2.flatten()
        jac[params_list.index('r1'),n0+n1:n0+n1+n2] = dr1_2.flatten()
        jac[params_list.index('r2'),n0+n1:n0+n1+n2] = dr2_2.flatten()
        jac[params_list.index('u0'),n0+n1:n0+n1+n2] = du0_2.flatten()
        jac[params_list.index('u1'),n0+n1:n0+n1+n2] = du1_2.flatten()
        jac[params_list.index('u2'),n0+n1:n0+n1+n2] = du2_2.flatten()

        # ---

        diff = np.concatenate([e.flatten() for e in es])

        mask = np.isfinite(diff)

        return jac[:,mask]

    def residual_3d(self, params, x0, x1, x2, y, e):

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A = params['A3d']
        B = params['B3d']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, '3d')

        diff = []

        y_fit = A*y_gauss+B

        res = (y_fit-y)/e

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_3d(self, params, x0, x1, x2, y, e):

        params_list = list(params.keys())

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        u0 = params['u0']
        u1 = params['u1']
        u2 = params['u2']

        A = params['A3d']
        # B = params['B3d']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, '3d')

        dA = y_gauss/e

        dB = 1/e

        yc_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode='3d')

        dc0, dc1, dc2 = A*yc_gauss/e

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)

        yr_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode='3d')

        yu_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode='3d')

        dr0, dr1, dr2 = A*yr_gauss/e

        du0, du1, du2 = A*yu_gauss/e

        n, n_params = y.size, len(params)
        jac = np.zeros((n_params,n))

        jac[params_list.index('A3d'),:n] = dA.flatten()
        jac[params_list.index('B3d'),:n] = dB.flatten()
        jac[params_list.index('c0'),:n] = dc0.flatten()
        jac[params_list.index('c1'),:n] = dc1.flatten()
        jac[params_list.index('c2'),:n] = dc2.flatten()
        jac[params_list.index('r0'),:n] = dr0.flatten()
        jac[params_list.index('r1'),:n] = dr1.flatten()
        jac[params_list.index('r2'),:n] = dr2.flatten()
        jac[params_list.index('u0'),:n] = du0.flatten()
        jac[params_list.index('u1'),:n] = du1.flatten()
        jac[params_list.index('u2'),:n] = du2.flatten()

        # ---

        mask = np.isfinite(e.flatten())

        return jac[:,mask]

    def residual(self, params, args_1d, args_2d, args_3d):

        cost_1d = self.residual_1d(params, *args_1d)
        cost_2d = self.residual_2d(params, *args_2d)
        cost_3d = self.residual_3d(params, *args_3d)

        reg = [params[key] for key in params.keys()]

        cost = np.concatenate([cost_1d, cost_2d, cost_3d, reg])

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):

        params_list = list(params.keys())        

        jac_1d = self.jacobian_1d(params, *args_1d)
        jac_2d = self.jacobian_2d(params, *args_2d)
        jac_3d = self.jacobian_3d(params, *args_3d)

        reg = np.eye(len(params_list))

        jac = np.column_stack([jac_1d, jac_2d, jac_3d, reg])

        return jac

    def estimate_envelope(self, x0, x1, x2, counts, y, e):

        y1d_0, e1d_0 = self.normalize(x0, x1, x2, counts, y, e, mode='1d_0')
        y1d_1, e1d_1 = self.normalize(x0, x1, x2, counts, y, e, mode='1d_1')
        y1d_2, e1d_2 = self.normalize(x0, x1, x2, counts, y, e, mode='1d_2')

        y0, y1, y2 = y1d_0, y1d_1, y1d_2

        y0_min = np.nanmin(y0)
        y0_max = np.nanmax(y0)

        if np.isclose(y0_max, y0_min) or (y0 > 0).sum() <= 13:
            return None

        y1_min = np.nanmin(y1)
        y1_max = np.nanmax(y1)

        if np.isclose(y1_max, y1_min) or (y1 > 0).sum() <= 13:
            return None

        y2_min = np.nanmin(y2)
        y2_max = np.nanmax(y2)

        if np.isclose(y2_max, y2_min) or (y2 > 0).sum() <= 13:
            return None

        self.params.add('A1d_0', value=y0_max, min=0, max=2*y0_max)
        self.params.add('A1d_1', value=y1_max, min=0, max=2*y1_max)
        self.params.add('A1d_2', value=y2_max, min=0, max=2*y2_max)

        self.params.add('B1d_0', value=y0_min, min=-y0_max, max=5*y0_max)
        self.params.add('B1d_1', value=y1_min, min=-y1_max, max=5*y1_max)
        self.params.add('B1d_2', value=y2_min, min=-y2_max, max=5*y2_max)

        C0_max = (y0_max-y0_min)/(x0[-1,0,0]-x0[0,0,0])
        C1_max = (y1_max-y1_min)/(x1[0,-1,0]-x1[0,0,0])
        C2_max = (y2_max-y2_min)/(x2[0,0,-1]-x2[0,0,0])

        self.params.add('C1d_0', value=0, min=-2*C0_max, max=2*C0_max)
        self.params.add('C1d_1', value=0, min=-2*C1_max, max=2*C1_max)
        self.params.add('C1d_2', value=0, min=-2*C2_max, max=2*C2_max)

        y1d = [y1d_0, y1d_1, y1d_2]
        e1d = [e1d_0, e1d_1, e1d_2]

        args_1d = [x0, x1, x2, y1d, e1d]

        y2d_0, e2d_0 = self.normalize(x0, x1, x2, counts, y, e, mode='2d_0')
        y2d_1, e2d_1 = self.normalize(x0, x1, x2, counts, y, e, mode='2d_1')
        y2d_2, e2d_2 = self.normalize(x0, x1, x2, counts, y, e, mode='2d_2')

        y0, y1, y2 = y2d_0, y2d_1, y2d_2

        y0_min = np.nanmin(y0)
        y0_max = np.nanmax(y0)

        if np.isclose(y0_max, y0_min) or (y0 > 0).sum() <= 13:
            return None

        y1_min = np.nanmin(y1)
        y1_max = np.nanmax(y1)

        if np.isclose(y1_max, y1_min) or (y1 > 0).sum() <= 13:
            return None

        y2_min = np.nanmin(y2)
        y2_max = np.nanmax(y2)

        if np.isclose(y2_max, y2_min) or (y2 > 0).sum() <= 13:
            return None

        self.params.add('A2d_0', value=y0_max, min=0, max=2*y0_max)
        self.params.add('A2d_1', value=y1_max, min=0, max=2*y1_max)
        self.params.add('A2d_2', value=y2_max, min=0, max=2*y2_max)

        self.params.add('B2d_0', value=y0_min, min=-y0_max, max=5*y0_max)
        self.params.add('B2d_1', value=y1_min, min=-y1_max, max=5*y1_max)
        self.params.add('B2d_2', value=y2_min, min=-y2_max, max=5*y2_max)

        C01_max = (y0_max-y0_min)/(x1[0,-1,0]-x1[0,0,0])
        C02_max = (y0_max-y0_min)/(x2[0,0,-1]-x2[0,0,0])

        C10_max = (y1_max-y1_min)/(x0[-1,0,0]-x0[0,0,0])
        C12_max = (y1_max-y1_min)/(x2[0,0,-1]-x2[0,0,0])

        C20_max = (y2_max-y2_min)/(x0[-1,0,0]-x0[0,0,0])
        C21_max = (y2_max-y2_min)/(x1[0,-1,0]-x1[0,0,0])

        self.params.add('C2d_01', value=0, min=-2*C01_max, max=2*C01_max)
        self.params.add('C2d_02', value=0, min=-2*C02_max, max=2*C02_max)

        self.params.add('C2d_10', value=0, min=-2*C10_max, max=2*C10_max)
        self.params.add('C2d_12', value=0, min=-2*C12_max, max=2*C12_max)

        self.params.add('C2d_20', value=0, min=-2*C20_max, max=2*C20_max)
        self.params.add('C2d_21', value=0, min=-2*C21_max, max=2*C21_max)

        y2d = [y2d_0, y2d_1, y2d_2]
        e2d = [e2d_0, e2d_1, e2d_2]

        args_2d = [x0, x1, x2, y2d, e2d]

        y3d, e3d = self.normalize(x0, x1, x2, counts, y, e, mode='3d')

        y_min = np.nanmin(y3d)
        y_max = np.nanmax(y3d)

        if np.isclose(y_max, y_min) or (y > 0).sum() <= 13:
            return None

        self.params.add('A3d', value=y_max, min=0, max=2*y_max)

        self.params.add('B3d', value=y_min, min=-2*y_max, max=2*y_max)

        args_3d = [x0, x1, x2, y3d, e3d]

        self.redchi2 = []

        y1, y2, y3 = y1d_0, y2d_0, y3d
        e1, e2, e3 = e1d_0, e2d_0, e3d

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(args_1d, args_2d, args_3d),
                        nan_policy='omit')

        result = out.minimize(method='leastsq',
                              Dfun=self.jacobian,
                              ftol=1e-6,
                              gtol=1e-6,
                              xtol=1e-6,
                              max_nfev=100,
                              col_deriv=True)

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

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    u0, u1, u2)

        args = x0, x1, x2, c, inv_S

        C1_0 = self.params['C1d_0'].value

        B1 = self.params['B1d_0'].value

        A1 = self.params['A1d_0'].value

        y1_gauss = self.gaussian(*args, '1d_0')

        y1_fit = A1*y1_gauss+B1+C1_0*(x0[:,0,0]-c0)

        self.ellipsoid_covariance(inv_S, mode='1d_0')

        chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, y1_fit, y1, e1, '1d_0')

        self.redchi2.append(chi2)

        # ---

        C2_1 = self.params['C2d_01'].value
        C2_2 = self.params['C2d_02'].value

        B2 = self.params['B2d_0'].value

        A2 = self.params['A2d_0'].value

        y2_gauss = self.gaussian(*args, '2d_0')

        y2_fit = A2*y2_gauss+B2+C2_1*(x1[0,:,:]-c1)+C2_2*(x2[0,:,:]-c2)

        chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, y2_fit, y2, e2, '2d_0')

        self.redchi2.append(chi2)
    
        # ---

        B3 = self.params['B3d'].value

        A3 = self.params['A3d'].value

        y3_gauss = self.gaussian(*args, '3d')

        y3_fit = A3*y3_gauss+B3

        chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, y3_fit, y3, e3, '3d')

        self.redchi2.append(chi2)

        # ---

        self.bkg = B3

        B3_err = self.params['B3d'].stderr
        if B3_err is None:
            B3_err = B3

        self.bkg_err = B3_err

        self.error_scale = np.sqrt(self.redchi2[2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S, (y1_fit, y1, e1), (y2_fit, y2, e2), (y3_fit, y3, e3)

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def fit(self, x0_prof, x1_proj, x2_proj, c, y_norm, e_norm, dx, xmod):

        counts = c.copy()
        y = y_norm.copy()
        e = e_norm.copy()

        x0 = x0_prof-xmod
        x1 = x1_proj.copy()
        x2 = x2_proj.copy()

        self.update_constraints(x0, x1, x2, dx)

        mask = (counts >= 0) & (e >= 0) & np.isfinite(counts) & np.isfinite(e)

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

        if (np.array(y.shape) <= 3).any():
            return None

        x0 = x0[i0:j0,i1:j1,i2:j2].copy()
        x1 = x1[i0:j0,i1:j1,i2:j2].copy()
        x2 = x2[i0:j0,i1:j1,i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print('Invalid data')
            return None

        weights = self.estimate_envelope(x0, x1, x2, counts, y, e)

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
        c0 += xmod
        c = c0, c1, c2

        r0, r1, r2 = np.sqrt(V)

        v0, v1, v2 = W.T

        binning = (x0+xmod, x1, x2), y, e

        fitting = binning, y_fit

        self.best_fit = c, S, *fitting

        self.best_prof = (x0[:,0,0]+xmod, y_prof, e_prof), y_prof_fit

        self.best_proj = (x1[0,:,:], x2[0,:,:], y_proj, e_proj), y_proj_fit

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate(self, x0, x1, x2, y, e, counts, c, S):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d3x = dx0*dx1*dx2

        y /= d3x
        e /= d3x

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        # r = 1.2*np.max(np.sqrt(np.linalg.eigvalsh(S)))
        # r = np.cbrt(2*np.sqrt(np.linalg.det(S)))

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)
        # sphere = np.einsum('ij,jklm,iklm->klm', np.diag([1/r**2]*3), x, x)

        pk = (ellipsoid <= 1.1**2) & (e > 0)
        bkg = (ellipsoid > 1.1**2) & (ellipsoid < 2**2) & (e > 0)

        y_pk = y[pk].copy()
        e_pk = e[pk].copy()

        # y_bkg = y[bkg].copy()
        # e_bkg = e[bkg].copy()

        #b = np.nanmean(y_bkg)
        #b_err = np.sqrt(np.nanmean(e_bkg**2))
        b = self.bkg
        b_err = self.bkg_err

        intens = np.nansum(y_pk-b)*d3x
        sig = np.sqrt(np.nansum(e_pk**2+b_err**2))*d3x

        sig *= np.sqrt(1+self.error_scale**2)

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