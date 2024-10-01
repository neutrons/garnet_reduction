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

        if data.laue:
            return instance.laue_integrate()
        else:
            return instance.monochromatic_integrate()

    def laue_integrate(self):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        runs = self.plan['Runs']

        # grouping_file = self.plan['GroupingFile']

        self.run = 0
        self.runs = len(runs)

        for run in runs:

            self.run += 1

            data.load_data('data', self.plan['IPTS'], run)

            data.load_generate_normalization(self.plan['VanadiumFile'],
                                             self.plan['FluxFile'])

            data.load_spectra_file(self.plan['SpectraFile'])

            data.apply_calibration('data',
                                   self.plan.get('DetectorCalibration'),
                                   self.plan.get('TubeCalibration'))

            data.apply_mask('data', self.plan.get('MaskFile'))

            data.preprocess_detectors('data')

            data.crop_for_normalization('data')

            data.normalize_data('data')

            data.load_background(self.plan['BackgroundFile'], 'data')

            data.convert_to_Q_sample('data', 'md', lorentz_corr=True)

            data.load_clear_UB(self.plan['UBFile'], 'data')

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

            self.fit_peaks('peaks', params)

            peaks.remove_weak_peaks('peaks')

            peaks.combine_peaks('peaks', 'combine')

            md_file = self.get_diagnostic_file('run#{}_data'.format(run))
            data.save_histograms(md_file, 'md', sample_logs=True)

            pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))
            peaks.save_peaks(pk_file, 'peaks')

            data.delete_workspace('peaks')
            data.delete_workspace('md')

        peaks.save_peaks(output_file, 'combine')

        mtd.clear()

        return output_file

    def laue_combine(self, files):

        output_file = self.get_output_file()
        result_file = self.get_file(output_file, '')

        peaks = PeaksModel()

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

    def fit_peaks(self, peaks_ws, params, make_plot=True):
        """
        Integrate peaks.

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

        if make_plot:

            plot = PeakPlot()

        UB = self.peaks.get_UB(peaks_ws)

        lo, lc, to, tc = params

        for i in range(n_peak):

            comp = '{:3.0f}%'.format(i/n_peak*100)
            iters = '({:}/{:})'.format(self.run, self.runs)
            proc = 'Proc {:2}:'.format(self.proc)

            print(proc+' '+iters+' '+comp)

            # d = peak.get_d_spacing(i)
            h, k, l = peak.get_hkl(i)

            wavelength = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            l_cut = lo+lc*wavelength
            t_cut = to+tc*two_theta/2

            params = peak.get_peak_shape(i, l_cut)

            peak.set_peak_intensity(i, 0, 0)

            det_id = peak.get_detector_id(i)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            T = np.eye(3)

            params = self.project_ellipsoid_parameters(params, T)

            bin_params = l_cut, t_cut, dQ, UB

            bins, extents, projections = self.bin_extent(*params, *bin_params)

            data_norm = data.normalize_in_Q('md', extents, bins, projections)

            y, e, Q0, Q1, Q2 = data_norm

            counts = data.extract_counts('md_data')

            ellipsoid = PeakEllipsoid(counts)

            params = ellipsoid.fit(Q0, Q1, Q2, y, e, dQ, l_cut, t_cut)

            if params is not None and det_id > 0:

                c, S, *fitting = ellipsoid.best_fit

                params = self.revert_ellipsoid_parameters(params, projections)

                params = self.revert_ellipsoid_parameters(params, T)

                peak.set_peak_shape(i, *params)

                bin_data = ellipsoid.bin_data

                I, sigma = ellipsoid.integrate_norm(bin_data, c, S)

                peak.set_peak_intensity(i, I, sigma)

                if make_plot:

                    plot.add_fitting(*fitting)

                    plot.add_profile_fit(*ellipsoid.best_prof)

                    plot.add_projection_fit(*ellipsoid.best_proj)

                    plot.add_ellipsoid(c, S)

                    goniometer = peak.get_goniometer_angles(i)

                    plot.add_peak_info(wavelength, angles, goniometer)

                    plot.add_peak_stats(ellipsoid.redchi2)

                    plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

                    peak_name = peak.get_peak_name(i)

                    plot.save_plot(self.get_plot_file(peak_name))

    def bin_axes(self, c0, c1, c2):

        n = np.array([c0, c1, c2])

        n /= np.linalg.norm(n)

        v = np.array([0, 1, 0])

        u = np.cross(v, n)
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        return n, u, v

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
                         v0, v1, v2, l_cut, t_cut, bin_size, UB):

        n, u, v = self.bin_axes(Q0, Q1, Q2)

        projections = [n, u, v]

        params = Q0, Q1, Q2, r0, r1, r2, v0, v1, v2

        params = self.project_ellipsoid_parameters(params, projections)

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        dQ = 2*np.array([l_cut, t_cut, t_cut])

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

        # bin_sizes = np.array([bin_size, bin_size, bin_size])
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

    def __init__(self, counts):

        self.counts = counts.copy()

        self.params = Parameters()

    def update_constraints(self, x0, x1, x2, y, dx, l_cut, t_cut):

        r0 = (x0[:,0,0][-1]-x0[:,0,0][0])/4
        r1 = (x1[0,:,0][-1]-x1[0,:,0][0])/4
        r2 = (x2[0,0,:][-1]-x2[0,0,:][0])/4

        r0_max = (x0[:,0,0][-1]-x0[:,0,0][0])/2
        r1_max = (x1[0,:,0][-1]-x1[0,:,0][0])/2
        r2_max = (x2[0,0,:][-1]-x2[0,0,:][0])/2

        c0, c1, c2 = x0[:,0,0].mean(), x1[0,:,0].mean(), x2[0,0,:].mean()

        c0_min, c1_min, c2_min = x0[0,0,0], x1[0,0,0], x2[0,0,0]
        c0_max, c1_max, c2_max = x0[-1,0,0], x1[0,-1,0], x2[0,0,-1]

        phi = omega = 0
        theta = np.pi/2

        self.params.add('c0', value=c0, min=c0_min, max=c0_max)
        self.params.add('c1', value=c1, min=c1_min, max=c1_max)
        self.params.add('c2', value=c2, min=c2_min, max=c2_max)

        self.params.add('r0', value=r0, min=2*dx, max=r0_max)
        self.params.add('r1', value=r1, min=2*dx, max=r1_max)
        self.params.add('r2', value=r2, min=2*dx, max=r2_max)

        self.params.add('phi', value=phi, min=-np.pi, max=np.pi)
        self.params.add('theta', value=theta, min=0, max=np.pi)
        self.params.add('omega', value=omega, min=-np.pi, max=np.pi)

        # self.params.add('phi', value=phi)
        # self.params.add('theta', value=theta)
        # self.params.add('omega', value=omega)

    def eigenvectors(self, W):

        w = scipy.spatial.transform.Rotation.from_matrix(W).as_rotvec()

        omega = np.linalg.norm(w)

        u0, u1, u2 = (0, 0, 1) if np.isclose(omega, 0) else w/omega

        return u0, u1, u2, omega

    def angles(self, v0, v1, v2):

        W = np.column_stack([v0, v1, v2])

        u0, u1, u2, omega = self.eigenvectors(W)

        theta = np.arccos(u2)
        phi = np.arctan2(u1, u0)

        return phi, theta, omega

    def scale(self, r0, r1, r2):

        return 0.25*r0, 0.25*r1, 0.25*r2

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

        u0 = np.cos(phi)*np.sin(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(theta)

        w = omega*np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def centroid_inverse_covariance(self, c0, c1, c2,
                                          r0, r1, r2,
                                          phi, theta, omega):

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        c = np.array([c0, c1, c2])
        inv_S = self.inv_S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        return c, inv_S

    def residual(self, params, x0, x1, x2, y, e, lamda=1e-2):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        # c0, c1, c2, r0, r1, r2, phi, theta, omega = params

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    phi, theta, omega)

        diff = []

        args = x0, x1, x2, 1, 0, c, inv_S

        y1_gauss = self.gaussian(*args, '1d')
        y2_gauss = self.gaussian(*args, '2d')
        y3_gauss = self.gaussian(*args, '3d')

        y1_lorentz = self.lorentzian(*args, '1d')
        y2_lorentz = self.lorentzian(*args, '2d')
        y3_lorentz = self.lorentzian(*args, '3d')

        y1_int, e1_int = self.integrate(x0, x1, x2, y, e, '1d')
        y2_int, e2_int = self.integrate(x0, x1, x2, y, e, '2d')
        y3_int, e3_int = y.copy(), e.copy()

        p1d = y1_gauss, y1_lorentz, y1_int, e1_int
        p2d = y2_gauss, y2_lorentz, y2_int, e2_int
        p3d = y3_gauss, y3_lorentz, y3_int, e3_int

        result = self.intensity_background_shared(x0, x1, x2, p1d, p2d, p3d)

        intens1, intens2, intens3, bkg1, bkg2, bkg3, global_bkg = result

        G1, L1, G1_err, L1_err = intens1
        G2, L2, G2_err, L2_err = intens2
        G3, L3, G3_err, L3_err = intens3

        B1, B1_err = bkg1
        B2, B2_err = bkg2
        B3, B3_err = bkg3

        res = (G1*y1_gauss+L1*y1_lorentz+B1-y1_int)/e1_int
        res /= np.sqrt(res.size)

        diff += res.flatten().tolist()

        res = (G2*y2_gauss+L2*y2_lorentz+B2-y2_int)/e2_int
        res /= np.sqrt(res.size)

        diff += res.flatten().tolist()

        res = (G3*y3_gauss+L3*y3_lorentz+B3-y3_int)/e3_int
        res /= np.sqrt(res.size)

        diff += res.flatten().tolist()

        I_sig_1 = np.nansum(y1_int-B1)/np.sqrt(np.nansum(e1_int**2+B1_err**2))
        I_sig_2 = np.nansum(y2_int-B2)/np.sqrt(np.nansum(e2_int**2+B2_err**2))
        I_sig_3 = np.nansum(y3_int-B3)/np.sqrt(np.nansum(e3_int**2+B3_err**2))

        if not np.isfinite(I_sig_1):
            I_sig_1 = 1
        if not np.isfinite(I_sig_2):
            I_sig_2 = 1 
        if not np.isfinite(I_sig_3):
            I_sig_3 = 1 

        diff.append(lamda/I_sig_1)
        diff.append(lamda/I_sig_2)
        diff.append(lamda/I_sig_3)

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def integrate(self, x0, x1, x2, y, e, mode='1d'):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == '1d':
            n = np.nansum(y/e**2, axis=(1,2))
            d = np.nansum(y**2/e**2, axis=(1,2))
        else:
            n = np.nansum(y/e**2, axis=0)
            d = np.nansum(y**2/e**2, axis=0)

        y_int = d/n
        e_int = np.sqrt(d)/n

        mask = (y_int > 0) & np.isfinite(y_int) \
             & (e_int > 0) & np.isfinite(e_int)

        y_int[~mask] = np.nan
        e_int[~mask] = np.nan

        return y_int, e_int

    def gaussian(self, x0, x1, x2, A, B, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S, dx)
            factor = np.sqrt(np.linalg.det(inv_S)/(2*np.pi)**3)
        elif mode == '2d':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            mat = inv_S[1:,1:]
            d2 = np.einsum('i...,ij,j...->...', dx, mat, dx)
            factor = np.sqrt(np.linalg.det(mat)/(2*np.pi)**2)
        else: # mode == '1d'
            dx = dx0[:,0,0]
            inv_var = inv_S[0,0]
            d2 = inv_var*dx**2
            factor = np.sqrt(inv_var/(2*np.pi))

        return A*np.exp(-0.5*d2)*factor+B

    def lorentzian(self, x0, x1, x2, A, B, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        inv_G = inv_S/np.log(4) # FWHM = 2*sqrt(2*ln(2))*sigma = 2*gamma

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_G, dx)
            factor = np.sqrt(np.linalg.det(inv_G))/np.pi**2
            power = 2
        elif mode == '2d':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            mat = inv_G[1:,1:]
            d2 = np.einsum('i...,ij,j...->...', dx, mat, dx)
            factor = np.sqrt(np.linalg.det(mat))/(2*np.pi)
            power = 1.5
        else: # mode == '1d'
            dx = dx0[:,0,0]
            inv_var = inv_G[0,0]
            d2 = inv_var*dx**2
            factor = np.sqrt(inv_var)/np.pi
            power = 1

        return A/(1+d2)**power*factor+B

    def intensity_background_shared(self, x0, x1, x2, p1d, p2d, p3d):

        y1_gauss, y1_lorentz, y1, e1 = p1d
        y2_gauss, y2_lorentz, y2, e2 = p2d
        y3_gauss, y3_lorentz, y3, e3 = p3d

        mask1 = (y1 > 0) & (e1 > 0)
        mask2 = (y2 > 0) & (e2 > 0)
        mask3 = (y3 > 0) & (e3 > 0)

        n1, n2, n3 = np.sum(mask1), np.sum(mask2), np.sum(mask3)

        d1 = [np.ones(n1), x0[:,0,0][mask1], np.zeros(n1), np.zeros(n1)]
        d2 = [np.ones(n2), np.zeros(n2), x1[0,:,:][mask2], x2[0,:,:][mask2]]
        d3 = [np.ones(n3), x0[mask3]*0, x1[mask3]*0, x2[mask3]*0]

        w1 = 1/e1[mask1]#/np.sqrt(n1)
        w2 = 1/e2[mask2]#/np.sqrt(n2)
        w3 = 1/e3[mask3]#/np.sqrt(n3)

        A1 = (np.vstack([y1_gauss[mask1], y1_lorentz[mask1], *d1])*w1).T
        A2 = (np.vstack([y2_gauss[mask2], y2_lorentz[mask2], *d2])*w2).T
        A3 = (np.vstack([y3_gauss[mask3], y3_lorentz[mask3], *d3])*w3).T

        A = np.zeros((n1+n2+n3, 6+4))

        A[:n1,0] = A1[:,0]
        A[:n1,1] = A1[:,1]
        A[n1:n1+n2,2] = A2[:,0]
        A[n1:n1+n2,3] = A2[:,1]
        A[n1+n2:,4] = A3[:,0]
        A[n1+n2:,5] = A3[:,1]

        A[:n1,6:] = A1[:,2:]
        A[n1:n1+n2,6:] = A2[:,2:]
        A[n1+n2:,6:] = A3[:,2:]

        b = np.hstack([y1[mask1]*w1, y2[mask2]*w2, y3[mask3]*w3])

        # reg_matrix = alpha*np.eye(A.shape[1])

        # A, b = A.T @ A+reg_matrix, A.T @ b

        #result, residuals, *_ = np.linalg.lstsq(A, b, rcond=None)

        y1_max = np.nanmax(y1)
        y2_max = np.nanmax(y2)
        y3_max = np.nanmax(y3)

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        y1_int = np.nansum(y1)*dx0
        y2_int = np.nansum(y2)*dx1*dx2
        y3_int = np.nansum(y3)*dx0*dx1*dx2

        y1x0_slope = (y1_max-y1_int)/dx0
        y2x1_slope = (y2_max-y2_int)/dx1
        y2x2_slope = (y2_max-y2_int)/dx2

        y_max = np.nanmax([y1_max, y2_max, y3_max])

        lb = [0]*7+[-y1x0_slope, -y2x1_slope, -y2x2_slope]
        ub = [y1_int]*2+[y2_int]*2+[y3_int]*2\
           + [y_max, y1x0_slope, y2x1_slope, y2x2_slope]

        result = scipy.optimize.lsq_linear(A, 
                                           b,
                                           bounds=(lb, ub),
                                           method='bvls',
                                           lsmr_tol='auto')

        G1, L1, G2, L2, G3, L3, *params = result.x

        residuals = result.fun

        res_sq = np.sum(residuals**2)/(b.size-A.shape[1])
        inv_cov = np.dot(A.T, A)
        if np.linalg.det(inv_cov) > 0:
            cov = res_sq*np.linalg.inv(inv_cov)
        else:
            cov = res_sq*np.full_like(inv_cov, np.inf)

        G1_err, L1_err, \
        G2_err, L2_err, \
        G3_err, L3_err, *uncert = np.sqrt(np.diag(cov))

        uncert = np.array(uncert)

        d1 = np.array([np.ones_like(mask1), x0[:,0,0], x1[:,0,0]*0, x2[:,0,0]*0])
        d2 = np.array([np.ones_like(mask2), x0[0,:,:]*0, x1[0,:,:], x2[0,:,:]])
        d3 = np.array([np.ones_like(mask3), x0*0, x1*0, x2*0])

        B1 = np.einsum('i...,i->...', d1, params)
        B2 = np.einsum('i...,i->...', d2, params)
        B3 = np.einsum('i...,i->...', d3, params)

        B1_err = np.sqrt(np.einsum('i...,i->...', d1**2, uncert**2))
        B2_err = np.sqrt(np.einsum('i...,i->...', d2**2, uncert**2))
        B3_err = np.sqrt(np.einsum('i...,i->...', d3**2, uncert**2))

        global_bkg = params[0], uncert[0]

        intens1 = G1, L1, G1_err, L1_err
        intens2 = G2, L2, G2_err, L2_err
        intens3 = G3, L3, G3_err, L3_err

        bkg1 = B1, B1_err
        bkg2 = B2, B2_err
        bkg3 = B3, B3_err

        return intens1, intens2, intens3, bkg1, bkg2, bkg3, global_bkg

    # def fun(self, x, A, b):

    #     return A @ x-b

    # def intensity_background_shared(self, x0, x1, x2, p1d, p2d, p3d):

    #     y1_fit, y1, e1 = p1d
    #     y2_fit, y2, e2 = p2d
    #     y3_fit, y3, e3 = p3d

    #     mask1 = (y1 > 0) & (e1 > 0)
    #     mask2 = (y2 > 0) & (e2 > 0)
    #     mask3 = (y3 > 0) & (e3 > 0)

    #     n1, n2, n3 = np.sum(mask1), np.sum(mask2), np.sum(mask3)

    #     d1 = [np.ones(n1), x0[:,0,0][mask1]]
    #     d2 = [np.ones(n2), x1[0,:,:][mask2], x2[0,:,:][mask2]]
    #     d3 = [np.ones(n3), x0[mask3], x1[mask3], x2[mask3]]

    #     A1 = (np.vstack([y1_fit[mask1], *d1])/e1[mask1]).T
    #     A2 = (np.vstack([y2_fit[mask2], *d2])/e2[mask2]).T
    #     A3 = (np.vstack([y3_fit[mask3], *d3])/e3[mask3]).T

    #     m1, m2, m3 = len(d1), len(d2), len(d3)

    #     A = np.zeros((n1+n2+n3, 1+m1+m2+m3))

    #     A[:n1,0] = A1[:,0]
    #     A[n1:n1+n2,0] = A2[:,0]
    #     A[n1+n2:,0] = A3[:,0]

    #     A[:n1,1:1+m1] = A1[:,1:]
    #     A[n1:n1+n2,1+m1:1+m1+m2] = A2[:,1:]
    #     A[n1+n2:,1+m1+m2:] = A3[:,1:]

    #     b = np.hstack([y1[mask1]/e1[mask1],
    #                    y2[mask2]/e2[mask2],
    #                    y3[mask3]/e3[mask3]])

    #     result, residuals, *_ = np.linalg.lstsq(A, b, rcond=None)

    #     I = result[0]

    #     params1 = result[1:1+m1]
    #     params2 = result[1+m1:1+m1+m2]
    #     params3 = result[1+m1+m2:]

    #     d1 = [np.ones_like(mask1), x0[:,0,0]]
    #     d2 = [np.ones_like(mask2), x1[0,:,:], x2[0,:,:]]
    #     d3 = [np.ones_like(mask3), x0, x1, x2]

    #     B1 = np.einsum('i...,i->...', d1, params1)
    #     B2 = np.einsum('i...,i->...', d2, params2)
    #     B3 = np.einsum('i...,i->...', d3, params3)

    #     if np.isclose(I, 0):
    #         I += 1e-12

    #     if len(residuals) > 0:
    #         dof = b.size-len(result)
    #         res_var = residuals/dof
    #         cov_matrix = np.linalg.inv(np.dot(A.T, A))*res_var
    #         I_err = np.sqrt(cov_matrix[0, 0])
    #     else:
    #         I_err = I

    #     return I, I_err, B1, B2, B3

    def estimate_weights(self, x0, x1, x2, y, e):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        args = [x0, x1, x2, y, e]

        # ---

        self.params['c0'].set(vary=True)
        self.params['c1'].set(vary=True)
        self.params['c2'].set(vary=True)

        self.params['r0'].set(vary=True)
        self.params['r1'].set(vary=True)
        self.params['r2'].set(vary=True)

        self.params['phi'].set(vary=True)
        self.params['theta'].set(vary=True)
        self.params['omega'].set(vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=args,
                        reduce_fcn=np.abs,
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        # out = Minimizer(self.residual,
        #                 self.params,
        #                 fcn_args=args,
        #                 nan_policy='omit')

        # result = out.minimize(method='powell')

        # self.params = result.params

        # c0 = self.params['c0'].value
        # c1 = self.params['c1'].value
        # c2 = self.params['c2'].value

        # r0 = self.params['r0'].value
        # r1 = self.params['r1'].value
        # r2 = self.params['r2'].value

        # phi = self.params['phi'].value
        # theta = self.params['theta'].value
        # omega = self.params['omega'].value

        # params = [c0, c1, c2, r0, r1, r2, phi, theta, omega]

        # result = out.minimize(method='dual_annealing',
        #                       no_local_search=True,
        #                       maxfun=10000,
        #                       x0=params,
        #                       minimizer_kwargs={'method': 'powell'})

        # self.params = result.params

        # ---

        c0 = self.params['c0'].value
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value

        r0 = self.params['r0'].value
        r1 = self.params['r1'].value
        r2 = self.params['r2'].value

        phi = self.params['phi'].value
        theta = self.params['theta'].value
        omega = self.params['omega'].value

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    phi, theta, omega)

        args = x0, x1, x2, 1, 0, c, inv_S

        y1_gauss = self.gaussian(*args, '1d')
        y2_gauss = self.gaussian(*args, '2d')
        y3_gauss = self.gaussian(*args, '3d')

        y1_lorentz = self.lorentzian(*args, '1d')
        y2_lorentz = self.lorentzian(*args, '2d')
        y3_lorentz = self.lorentzian(*args, '3d')

        y1, e1 = self.integrate(x0, x1, x2, y, e, mode='1d')
        y2, e2 = self.integrate(x0, x1, x2, y, e, mode='2d')
        y3, e3 = y.copy(), e.copy()

        p1d = y1_gauss, y1_lorentz, y1, e1
        p2d = y2_gauss, y2_lorentz, y2, e2
        p3d = y3_gauss, y3_lorentz, y3, e3

        result = self.intensity_background_shared(x0, x1, x2, p1d, p2d, p3d)

        intens1, intens2, intens3, bkg1, bkg2, bkg3, global_bkg = result

        G1, L1, G1_err, L1_err = intens1
        G2, L2, G2_err, L2_err = intens2
        G3, L3, G3_err, L3_err = intens3

        B1, B1_err = bkg1
        B2, B2_err = bkg2
        B3, B3_err = bkg3

        B, B_err = global_bkg

        self.B, self.B_err = B, B_err

        y1_fit = G1*y1_gauss+L1*y1_lorentz+B1
        y2_fit = G2*y2_gauss+L2*y2_lorentz+B2
        y3_fit = G3*y3_gauss+L3*y3_lorentz+B3

        self.redchi2 = np.nanmean((y1_fit-y1)**2/e1**2),\
                       np.nanmean((y2_fit-y2)**2/e2**2),\
                       np.nanmean((y3_fit-y3)**2/e3**2)

        inv_S = self.inv_S_matrix(r0, r1, r2, phi, theta, omega)

        return c, inv_S, (y1_fit, y1, e1), (y2_fit, y2, e2), (y3_fit, y3, e3)

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def fit(self, x0, x1, x2, y, e, dx, l_cut, t_cut):

        self.update_constraints(x0, x1, x2, y, dx, l_cut, t_cut)

        mask = (e > 0) & (y > 0) & np.isfinite(e) & np.isfinite(y)

        y[~mask] = np.nan
        e[~mask] = np.nan

        y_max = np.nanmax(y)

        if mask.sum() < 20 or (np.array(mask.shape) <= 5).any() or y_max <= 0:
            return None

        coords = np.argwhere(mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0)+1

        y, e = y[i0:j0,i1:j1,i2:j2].copy(), e[i0:j0,i1:j1,i2:j2].copy()

        if (np.array(y.shape) <= 3).any():
            return None

        x0 = x0[i0:j0,i1:j1,i2:j2].copy()
        x1 = x1[i0:j0,i1:j1,i2:j2].copy()
        x2 = x2[i0:j0,i1:j1,i2:j2].copy()

        self.counts = self.counts[i0:j0,i1:j1,i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print('Invalid data')
            return None

        weights = self.estimate_weights(x0, x1, x2, y, e)

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

        self.bin_data = (x0, x1, x2), (dx0, dx1, dx2), y, e

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_norm(self, bins, c, S, norm=False):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = bins

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)

        pk = (ellipsoid <= 1.1) & (e > 0)
        bkg = (ellipsoid > 1.6) & (ellipsoid <= 2.1) & (e > 0)

        dilate = pk | bkg

        d3x = dx0*dx1*dx2

        scale = d3x if norm else 1#self.counts[pk].sum()

        y_pk = y[pk].copy()
        e_pk = e[pk].copy()

        # w_pk = 1/e_pk**2

        # y_bkg = y[bkg].copy()
        # e_bkg = e[bkg].copy()

        # w_bkg = 1/e_bkg**2

        # if len(w_bkg) > 2:
        #     b = self.weighted_median(y_bkg, w_bkg)
        #     b_err = self.jackknife_uncertainty(y_bkg, w_bkg)
        # else:
        #     b = b_err = 0

        # #if not norm:
        # self.info = [b, b_err]

        # intens = np.nansum(y[pk]-b)*scale
        # sig = np.sqrt(np.nansum(e[pk]**2+b_err**2))*scale

        # n_pk = np.sum(pk)
        # n_bkg = np.sum(bkg)

        #b = self.weighted_median(y_bkg, w_bkg)
        #b_err = self.jackknife_uncertainty(y_bkg, w_bkg)

        # b = np.nansum(y_bkg)/n_bkg
        # b_err = np.nansum(e_bkg**2)/n_bkg

        #b = np.nansum(w_bkg*y_bkg)/np.nansum(w_bkg)
        #b_err = 1/np.sqrt(np.nansum(w_bkg))

        self.info = [scale, scale]

        # b *= n_pk
        # b_err *= n_pk

        # intens = np.nansum(w_pk*(y_pk-b))/np.nansum(w_pk)
        # sig = np.sqrt(np.nansum(w_pk*(e_pk**2+b_err**2))/np.nansum(w_pk))

        b = self.B
        b_err = self.B_err

        intens = np.nansum(y_pk-b)*scale
        sig = np.sqrt(np.nansum(e_pk**2+b_err**2))*scale

        #if not norm:
        self.weights = (x0[pk], x1[pk], x2[pk]), self.counts[pk]

        # bin_count = np.nansum(self.counts[pk])

        self.info += [d3x, intens, sig]

        # if not norm:
        #     self.info += [d3x*np.sum(pk)]
        # else:
        #     self.info += [intens, sig]

        freq = y.copy()
        freq[~dilate] = np.nan

        if not np.isfinite(sig):
            sig = intens

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        return intens, sig

    def weighted_median(self, y, w):

        mask = np.logical_and(np.isfinite(y), np.isfinite(w))

        y_mask = y[mask].copy()
        w_mask = w[mask].copy()

        sort = np.argsort(y_mask)
        y_mask = y_mask[sort]
        w_mask = w_mask[sort]

        cum_wgt = np.cumsum(w_mask)
        tot_wgt_half = np.sum(w_mask)/2

        ind = np.searchsorted(cum_wgt, tot_wgt_half)

        return y_mask[ind]

    def jackknife_uncertainty(self, y, w):

        mask = np.logical_and(np.isfinite(y), np.isfinite(w))

        y_mask = y[mask].copy()
        w_mask = w[mask].copy()

        n = len(y_mask)

        sort = np.argsort(y_mask)
        y_mask = y_mask[sort]
        w_mask = w_mask[sort]

        cum_wgt = np.cumsum(w_mask)

        wgt_med = self.weighted_median(y_mask, w_mask)

        cum_wgt = np.cumsum(w_mask)
        rev_cum_wgt = np.cumsum(w_mask[::-1])[::-1]

        w_left = np.concatenate(([0], cum_wgt[:-1]))
        w_right = np.concatenate((rev_cum_wgt[1:], [0]))

        total_weight_excl = w_left+w_right
        half_weight_excl = total_weight_excl/2

        median_indices = np.where(w_left >= half_weight_excl,
                                  np.searchsorted(cum_wgt,
                                                  half_weight_excl,
                                                  side='left'),
                                  np.searchsorted(rev_cum_wgt[::-1],
                                                  half_weight_excl[::-1],
                                                  side='right'))

        med = y_mask[median_indices]

        dev = med-wgt_med
        return np.sqrt((n-1)*np.sum(dev**2) /n)

    # def jackknife_uncertainty(self, y, w):

    #     n = len(y)
    #     med = np.zeros(n)

    #     sort = np.argsort(y)
    #     y = y[sort]
    #     w = w[sort]

    #     for i in range(n):
    #         jk_y = np.delete(y, i)
    #         jk_w = np.delete(w, i)
    #         med[i] = self.weighted_median(jk_y, jk_w)

    #     wgt_med = self.weighted_median(y, w)

    #     dev = med-wgt_med

    #     return np.sqrt((n-1)*np.sum(dev**2)/n)