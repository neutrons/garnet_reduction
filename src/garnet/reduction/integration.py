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
        self.output = 'integration'

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
        plan['OutputName'] += '_p{}'.format(proc)

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

            self.peaks, self.data = peaks, data

            r_cut = self.params['Radius']

            ro, rc = self.estimate_peak_size('peaks', 'md', r_cut)

            # peaks.integrate_peaks('md',
            #                       'peaks',
            #                       ro,
            #                       rc,
            #                       method='ellipsoid',
            #                       centroid=True)

            self.fit_peaks('peaks', ro, rc)

            peaks.combine_peaks('peaks', 'combine')

            md_file = self.get_diagnostic_file('run#{}_data'.format(run))
            data.save_histograms(md_file, 'md', sample_logs=True)

            pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))
            peaks.save_peaks(pk_file, 'peaks')

            data.delete_workspace('peaks')
            data.delete_workspace('md')

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

        mtd.clear()

        return output_file

    def laue_combine(self, files):

        output_file = self.get_output_file()

        peaks = PeaksModel()

        for file in files:

            peaks.load_peaks(file, 'tmp')
            peaks.combine_peaks('tmp', 'combine')

        for file in files:
            os.remove(file)

        if mtd.doesExist('combine'):

            peaks.save_peaks(output_file, 'combine')

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(output_file)[0]+'.mat'

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

                ro, rc = self.estimate_peak_size('peaks', 'md')

                # peaks.integrate_peaks('md',
                #                       'peaks',
                #                       ro,
                #                       method='sphere')

                self.fit_peaks('peaks', ro, rc)

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

            ro, rc = self.estimate_peak_size('peaks', 'md')

            # peaks.integrate_peaks('md',
            #                       'peaks',
            #                       ro,
            #                       method='sphere')

            self.fit_peaks('peaks', ro, rc)

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

            peaks.save_peaks(output_file, 'combine')

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(output_file)[0]+'.mat'

            ub = UBModel('combine')
            ub.save_UB(ub_file)

        mtd.clear()

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
        ro : float
            Nominal radius.
        rc : float
            Scale radius with Q.

        """

        peaks = self.peaks

        peaks_name = peaks.get_peaks_name(peaks_ws)

        r, sig_noise = peaks.intensity_vs_radius(data_ws, peaks_ws, r_cut)

        sphere = PeakSphere(r_cut)

        r_cut = sphere.fit(r, sig_noise)

        sig_noise_fit, *vals = sphere.best_fit(r)

        plot = RadiusPlot(r, sig_noise, sig_noise_fit)

        plot.add_sphere(r_cut, *vals)

        profile = PeakProfile(r_cut)

        x, y, e, Q = peaks.intensity_Q_profile(data_ws, peaks_ws, r_cut)

        ro, rc = profile.fit(x, y, e, Q)

        x, y, e, I, rQ_min, rQ_max = profile.best_fit()

        plot.add_profile(x, y, e, I, rQ_min, rQ_max)

        plot.save_plot(self.get_plot_file(peaks_name))

        return ro, rc

    def fit_peaks(self, peaks_ws, ro, rc, make_plot=False):
        """
        Integrate peaks.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r_cut : float
            Cutoff radius.

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        if make_plot:
            plot = PeakPlot()

        for i in range(n_peak):

            comp = '{:3.0f}%'.format(i/n_peak*100)
            iters = '({:}/{:})'.format(self.run, self.runs)
            proc = 'Proc {:2}:'.format(self.proc)

            print(proc+' '+iters+' '+comp)

            d = peak.get_d_spacing(i)

            r_cut = ro+rc*2*np.pi/d

            params = peak.get_peak_shape(i, r_cut)

            peak.set_peak_intensity(i, 0, 0)

            wavelength = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            det_id = peak.get_detector_id(i)

            two_theta, az_phi = angles

            # R = peak.get_goniometer_matrix(i)

            T = np.eye(3)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            params = self.project_ellipsoid_parameters(params, T)

            bins, extents, projections = self.bin_extent(*params, bin_size=dQ)

            # params = self.project_ellipsoid_parameters(params, projections)

            data_norm = data.normalize_in_Q('md', extents, bins, projections)

            y, e, Q0, Q1, Q2 = data_norm

            counts = data.extract_counts('md_data')

            ellipsoid = PeakEllipsoid(counts)

            params = ellipsoid.fit(Q0, Q1, Q2, y, e, dQ, r_cut)

            if params is not None and det_id > 0:

                c, S, *fitting = ellipsoid.best_fit

                params = self.revert_ellipsoid_parameters(params, projections)

                params = self.revert_ellipsoid_parameters(params, T)

                peak.set_peak_shape(i, *params)

                bin_data = ellipsoid.bin_data

                I, sigma = ellipsoid.integrate_norm(bin_data, c, S)

                # scale = 1

                # if data.laue:

                #     (Q0, Q1, Q2), weights = ellipsoid.weights

                #     Q0, Q1, Q2 = self.trasform_Q(Q0, Q1, Q2, projections)

                #     scale = data.calculate_norm(det_id, Q0, Q1, Q2, weights)

                peak.set_peak_intensity(i, I, sigma)

                # peak.set_scale_factor(i, scale)

                # peak.add_diagonstic_info(i, ellipsoid.info)

                if make_plot:

                    plot.add_fitting(*fitting)

                    plot.add_ellipsoid(c, S)

                    goniometer = peak.get_goniometer_angles(i)

                    plot.add_peak_info(wavelength, angles, goniometer)

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

        return np.einsum('ij,jk->ik', W, [Q0, Q1, Q2])

    def bin_extent(self, Q0, Q1, Q2,
                         r0, r1, r2,
                         v0, v1, v2, bin_size=0.01):

        n, u, v = self.bin_axes(Q0, Q1, Q2)

        projections = [n, u, v]

        params = Q0, Q1, Q2, r0, r1, r2, v0, v1, v2

        params = self.project_ellipsoid_parameters(params, projections)

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        dQ = self.roi(r0, r1, r2, v0, v1, v2)

        dQ0, dQ1, dQ2 = dQ

        extents = np.array([[Q0-dQ0, Q0+dQ0],
                            [Q1-dQ1, Q1+dQ1],
                            [Q2-dQ2, Q2+dQ2]])

        # bin_sizes = np.array([bin_size, bin_size, bin_size])
        bin_sizes = np.array(dQ)/20
        bin_sizes[bin_sizes < bin_size] = bin_size

        min_adjusted = np.floor(extents[:,0]/bin_sizes)*bin_sizes
        max_adjusted = np.ceil(extents[:,1]/bin_sizes)*bin_sizes

        bins = ((max_adjusted-min_adjusted)/bin_sizes).astype(int)
        bin_sizes = (max_adjusted-min_adjusted)/bins

        bins = np.where(bins % 2 == 0, bins, bins+1)

        max_adjusted = min_adjusted+bins*bin_sizes

        extents = np.vstack((min_adjusted, max_adjusted)).T

        return bins, extents, projections

    def roi(self, r0, r1, r2, v0, v1, v2):

        V = np.diag([r0**2, r1**2, r2**2])
        U = np.column_stack([v0, v1, v2])

        S = np.dot(np.dot(U, V), U.T)

        dQ = 1.5*np.sqrt(np.diag(S))

        return dQ

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

    def __init__(self, r):

        self.params = Parameters()

        self.params.add('ro', value=r/2, min=r/10, max=r, vary=True)
        self.params.add('rc', value=0, min=-r, max=r, vary=False)

        self.r = r

    def model(self, ro, rc, Q):

        return (ro+rc*Q)/3

    def residual(self, params, sigma, w, Q):

        ro = params['ro'].value
        rc = params['rc'].value

        sigma_fit = self.model(ro, rc, Q)

        diff = (sigma-sigma_fit)*w
        diff[~np.isfinite(diff)] = 1e9

        return diff.flatten()

    def fit(self, x, y, e, Q):

        intens = np.nansum(y, axis=1)
        errors = np.sqrt(np.nansum(e**2, axis=1))

        w = np.abs(intens/errors)

        weight = np.nansum(y, axis=1)

        mu = np.nansum(x*y, axis=1)/weight
        sigma = np.sqrt(np.nansum((x-mu[:,np.newaxis])**2*y, axis=1)/weight)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(sigma, w, Q),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        self.sigma = sigma
        self.delta_sigma = 1/w

        self.I = intens
        self.Q = Q
        self.y = y
        self.e = e
        self.x = x-mu[:,np.newaxis]

        self.ro = result.params['ro'].value
        self.rc = result.params['rc'].value

        return self.ro, self.rc

    def best_fit(self):

        rQ_min = self.ro+self.rc*self.Q.min()
        rQ_max = self.ro+self.rc*self.Q.max()

        # r = self.ro+self.rc*self.Q

        return self.x, self.y, self.e, self.I, rQ_min, rQ_max

class PeakEllipsoid:

    def __init__(self, counts):

        self.counts = counts.copy()

        self.params = Parameters()

    def update_constraints(self, c0, c1, c2, dx, r_cut):

        dr = 2*r_cut

        self.params.add('r0', value=r_cut, min=2*dx, max=dr)
        self.params.add('r1', value=r_cut, min=2*dx, max=dr)
        self.params.add('r2', value=r_cut, min=2*dx, max=dr)

        self.params.add('c0', value=c0, min=c0-dr, max=c0+dr)
        self.params.add('c1', value=c1, min=c1-dr, max=c1+dr)
        self.params.add('c2', value=c2, min=c2-dr, max=c2+dr)

        self.params.add('phi', value=0, min=-np.pi, max=np.pi)
        self.params.add('theta', value=np.pi/2, min=0, max=np.pi)
        self.params.add('omega', value=0, min=-np.pi, max=np.pi)

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

    def residual(self, params, x0, x1, x2, y1d, y2d, y3d, w1d, w2d, w3d):

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        c, inv_S = self.centroid_inverse_covariance(c0, c1, c2,
                                                    r0, r1, r2,
                                                    phi, theta, omega)

        A1 = params['A1']
        A2 = params['A2']
        A3 = params['A3']

        B1 = params['B1']
        B2 = params['B2']
        B3 = params['B3']

        args = x0, x1, x2, A1, B1, c, inv_S
        y1d_fit = self.func(*args, mode='1d')

        args = x0, x1, x2, A2, B2, c, inv_S
        y2d_fit = self.func(*args, mode='2d')

        args = x0, x1, x2, A3, B3, c, inv_S
        y3d_fit = self.func(*args, mode='3d')

        diff = ((y1d-y1d_fit)*w1d).ravel().tolist()\
             + ((y2d-y2d_fit)*w2d).ravel().tolist()\
             + ((y3d-y3d_fit)*w3d).ravel().tolist()

        return np.array(diff)

    def func(self, x0, x1, x2, A, B, c, inv_S, mode='3d'):

        c0, c1, c2 = c

        dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        if mode == '3d':
            dx = [dx0, dx1, dx2]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S, dx)
            # factor = np.sqrt(np.linalg.det(inv_S)/(2*np.pi)**3)
        elif mode == '2d':
            dx = [dx1[0,:,:], dx2[0,:,:]]
            d2 = np.einsum('i...,ij,j...->...', dx, inv_S[1:,1:], dx)
            # factor = np.sqrt(np.linalg.det(inv_S[1:,1:])/(2*np.pi)**2)
        else:
            dx = dx0[:,0,0]
            d2 = inv_S[0,0]*dx**2
            # factor = np.sqrt(inv_S[0,0]**2/(2*np.pi))

        return A*np.exp(-0.5*d2)+B

    def estimate_weights(self, x0, x1, x2, y, e, dx, r_cut):

        c0, c1, c2 = x0[:,0,0].mean(), x1[0,:,0].mean(), x2[0,0,:].mean()

        self.update_constraints(c0, c1, c2, dx, r_cut)        

        # dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        y1d = np.nansum(y, axis=(1,2))#*dx1*dx2
        y2d = np.nansum(y, axis=0)#*dx0
        y3d = y.copy()

        e1d = np.sqrt(np.nansum(e**2, axis=(1,2)))#*dx1*dx2
        e2d = np.sqrt(np.nansum(e**2, axis=0))#*dx0
        e3d = e.copy()

        if not np.nansum(y1d) > 0 or not np.nansum(e1d) > 0:
            return None

        if not np.nansum(y2d) > 0 or not np.nansum(e2d) > 0:
            return None

        if not np.nansum(y3d) > 0 or not np.nansum(e3d) > 0:
            return None

        w1d = (1/np.sqrt(e1d.size))/e1d
        w2d = (1/np.sqrt(e2d.size))/e2d
        w3d = (1/np.sqrt(e3d.size))/e3d

        # y_int = np.nansum(y)*dx0*dx1*dx2

        # if y_int <= 1e-9:
        #     return None

        # self.params.add('A', value=y_int, min=0, max=10*y_int)

        y1_max = np.nanmax(y1d)
        y2_max = np.nanmax(y2d)
        y3_max = np.nanmax(y3d)

        y1_min = np.nanmin(y1d)
        y2_min = np.nanmin(y2d)
        y3_min = np.nanmin(y3d)

        A1 = np.nanpercentile(y1d, 99)
        A2 = np.nanpercentile(y2d, 99)
        A3 = np.nanpercentile(y3d, 99)

        B1 = np.nanpercentile(y1d, 15)
        B2 = np.nanpercentile(y2d, 15)
        B3 = np.nanpercentile(y3d, 15)

        if np.isclose(y1_min, y1_max) or not np.isfinite(y1_max):
            return None

        if np.isclose(y2_min, y2_max) or not np.isfinite(y2_max):
            return None

        if np.isclose(y3_min, y3_max) or not np.isfinite(y3_max):
            return None

        self.params.add('A1', value=A1, min=0, max=2*y1_max)
        self.params.add('A2', value=A2, min=0, max=2*y2_max)
        self.params.add('A3', value=A3, min=0, max=2*y3_max)

        self.params.add('B1', value=B1, min=0, max=y1_max)
        self.params.add('B2', value=B2, min=0, max=y2_max)
        self.params.add('B3', value=B3, min=0, max=y3_max)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(x0, x1, x2, y1d, y2d, y3d, w1d, w2d, w3d),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        c0 = self.params['c0'].value
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value

        r0 = self.params['r0'].value
        r1 = self.params['r1'].value
        r2 = self.params['r2'].value

        phi = self.params['phi'].value
        theta = self.params['theta'].value
        omega = self.params['omega'].value

        # dx0, dx1, dx2 = x0-c0, x1-c1, x2-c2

        # dx = [dx0, dx1, dx2]

        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, phi, theta, omega)

        return c, inv_S

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def fit(self, x0, x1, x2, y_norm, e_norm, dx, r_cut):

        y, e = y_norm.copy(), e_norm.copy()

        mask = e > 0

        y_max = np.nanmax(y)

        if mask.sum() > 21 and (np.array(mask.shape) >= 5).all() and y_max > 0:

            dx0, dx1, dx2 = self.voxels(x0, x1, x2)

            y[~mask] = np.nan
            e[~mask] = np.nan

            # mask = e > 0

            # bins = (x0[mask], x1[mask], x2[mask]), y[mask], e[mask]

            if not np.nansum(y) > 0:
                print('Invalid data')
                return None

            weights = self.estimate_weights(x0, x1, x2, y, e, dx, r_cut)

            if weights is None:
                print('Invalid weight estimate')
                return None

            c, inv_S = weights

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

            peak = y.copy()*np.nan
            peak = threshold*1.0

            V, W = np.linalg.eigh(S)

            c0, c1, c2 = c

            r0, r1, r2 = np.sqrt(V)

            v0, v1, v2 = W.T

            binning = (x0, x1, x2), y, e

            fitting = binning, peak

            self.best_fit = c, S, *fitting

            self.bin_data = (x0, x1, x2), (dx0, dx1, dx2), y, e

            return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_norm(self, bins, c, S, norm=False):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = bins

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)

        pk = (ellipsoid <= 1) & (e > 0)
        bkg = (ellipsoid > 1) & (ellipsoid <= 1.5**2) & (e > 0)

        dilate = pk | bkg

        d3x = dx0*dx1*dx2

        scale = d3x if norm else 1

        y_bkg = y[bkg].copy()
        e_bkg = e[bkg].copy()

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
        n_bkg = np.sum(bkg)

        b = np.nansum(y_bkg)/n_bkg
        b_err = np.sqrt(np.nansum(e_bkg**2))/n_bkg

        self.info = [scale, scale]

        # b *= n_pk
        # b_err *= n_pk

        intens = np.nansum(y[pk]-b)*scale
        sig = np.sqrt(np.nansum(e[pk]**2+b_err**2))*scale

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

        sort = np.argsort(y)
        y = y[sort]
        w = w[sort]

        cum_wgt = np.cumsum(w)
        tot_wgt = np.sum(w)

        ind = np.where(cum_wgt >= tot_wgt/2)[0][0]

        return y[ind]

    def jackknife_uncertainty(self, y, w):

        n = len(y)
        med = np.zeros(n)

        sort = np.argsort(y)
        y = y[sort]
        w = w[sort]

        for i in range(n):
            jk_y = np.delete(y, i)
            jk_w = np.delete(w, i)
            med[i] = self.weighted_median(jk_y, jk_w)

        wgt_med = self.weighted_median(y, w)

        dev = med-wgt_med

        return np.sqrt((n-1)*np.sum(dev**2)/n)