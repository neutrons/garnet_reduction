import os

import numpy as np

import scipy.spatial.transform
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.ndimage

from lmfit import Minimizer, Parameters

# import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='garnet.integration.log', level=logging.INFO)

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
                                              lamda_min,
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

            self.peaks, self.data = peaks, data

            r_cut = self.estimate_peak_size('peaks', 'md')

            peaks.integrate_peaks('md',
                                  'peaks',
                                  r_cut/2,
                                  method='sphere')

            peaks.remove_weak_peaks('peaks')

            self.fit_peaks('peaks', r_cut)

            peaks.combine_peaks('peaks', 'combine')

            md_file = self.get_diagnostic_file('run#{}_data'.format(run))
            data.save_histograms(md_file, 'md', sample_logs=True)

            pk_file = self.get_diagnostic_file('run#{}_peaks'.format(run))
            peaks.save_peaks(pk_file, 'peaks')

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

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

            data.convert_to_Q_sample('data', 'md_data', lorentz_corr=False)
            data.convert_to_Q_sample('data', 'md_corr', lorentz_corr=True)

            if self.plan.get('UBFile') is None:
                UB_file = output_file.replace('.nxs', '.mat')
                data.save_UB(UB_file, 'md_data')
                self.plan['UBFile'] = UB_file

            data.load_clear_UB(self.plan['UBFile'], 'md_data')

            peaks.predict_peaks('md_data',
                                'peaks',
                                self.params['Centering'],
                                self.params['MinD'],
                                lamda_min,
                                lamda_max)

            if self.params['MaxOrder'] > 0:

                peaks.predict_satellite_peaks('peaks',
                                              'md_data',
                                              self.params['MinD'],
                                              lamda_min,
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

            peaks.combine_peaks('peaks', 'combine')

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
                                         'tmp_data',
                                         lorentz_corr=False)

                data.convert_to_Q_sample('data',
                                         'tmp_corr',
                                         lorentz_corr=True)

                data.combine_histograms('tmp_data', 'md_data')
                data.combine_histograms('tmp_corr', 'md_corr')

                if self.plan.get('UBFile') is None:
                    UB_file = output_file.replace('.nxs', '.mat')
                    data.save_UB(UB_file, 'md_data')
                    self.plan['UBFile'] = UB_file

                data.load_clear_UB(self.plan['UBFile'], 'md_data')

                peaks.predict_peaks('md_data',
                                    'peaks',
                                    self.params['Centering'],
                                    self.params['MinD'],
                                    lamda_min,
                                    lamda_max)

                if self.params['MaxOrder'] > 0:

                    peaks.predict_satellite_peaks('peaks',
                                                  'md_data',
                                                  self.params['MinD'],
                                                  lamda_min,
                                                  lamda_max,
                                                  self.params['ModVec1'],
                                                  self.params['ModVec2'],
                                                  self.params['ModVec3'],
                                                  self.params['MaxOrder'],
                                                  self.params['CrossTerms'])

                peaks.combine_peaks('peaks', 'combine')

        peaks.convert_peaks('combine')

        peaks.integrate_peaks('md_data',
                              'combine',
                              self.params['Radius'],
                              method='sphere')

        peaks.save_peaks(output_file, 'combine')

        for ws in ['md_data', 'md_corr', 'norm']:
            file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
            data.save_histograms(file, ws, sample_logs=True)

        mtd.clear()

        return output_file

    def monochromatic_combine(self, files):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        for ws in ['md_data', 'md_corr', 'norm']:

            merge = []

            for file in files:
                md_file = file.replace('.nxs', '_{}.nxs'.format(ws))
                data.load_histograms(md_file, md_file)
                merge.append(md_file)
                os.remove(md_file)

            data.combine_Q_sample(merge, ws)

            if ws == 'md_data':
                for file in files:
                    peaks.load_peaks(file, 'peaks')
                    peaks.combine_peaks('peaks', 'combine')
                    os.remove(file)
                md_file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
                data.save_histograms(md_file, ws, sample_logs=True)

        pk_file = output_file.replace('.nxs', '_pk.nxs')
        peaks.save_peaks(pk_file, 'combine')

        peaks.renumber_runs_by_index('md_data', 'combine')

        self.peaks, self.data = peaks, data

        r_cut = self.estimate_peak_size('combine', 'md_corr')

        self.fit_peaks('combine', r_cut)

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

        opt = Optimization('combine')
        opt.optimize_lattice(self.params['Cell'])

        ub_file = os.path.splitext(output_file)[0]+'.mat'

        ub = UBModel('combine')
        ub.save_UB(ub_file)

        mtd.clear()

    def estimate_peak_size(self, peaks_ws, data_ws):
        """
        Integrate peaks with spherical envelope up to cutoff size.
        Estimates spherical envelope radius.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data.

        Returns
        -------
        r_cut : float
            Update cutoff radius.

        """

        peaks = self.peaks

        peaks_name = peaks.get_peaks_name(peaks_ws)

        r_cut = self.params['Radius']

        rad, sig_noise, intens = peaks.intensity_vs_radius(data_ws,
                                                           peaks_ws,
                                                           r_cut)

        sphere = PeakSphere(r_cut)

        r_cut = sphere.fit(rad, sig_noise)

        sig_noise_fit, *vals = sphere.best_fit(rad)

        plot = RadiusPlot(rad, sig_noise, sig_noise_fit)

        plot.add_sphere(r_cut, *vals)

        plot.save_plot(self.get_plot_file(peaks_name))

        return r_cut

    def fit_peaks(self, peaks_ws, r_cut):
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

        plot = PeakPlot()

        for i in range(n_peak):

            comp = '{:3.0f}%'.format(i/n_peak*100)
            iters = '({:}/{:})'.format(self.run, self.runs)
            proc = 'Proc {:2}:'.format(self.proc)

            print(proc+' '+iters+' '+comp)

            params = peak.get_peak_shape(i, r_cut)

            peak.set_peak_intensity(i, 0, 0)

            wavelength = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            R = peak.get_goniometer_matrix(i)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            j, max_iter = 0, 1

            while j < max_iter and params is not None:

                j += 1

                bins, extents, projections = self.bin_extent(*params,
                                                             R,
                                                             two_theta,
                                                             az_phi,
                                                             bin_size=dQ)

                y, e, Q0, Q1, Q2 = data.normalize_to_Q_sample('md',
                                                              extents,
                                                              bins,
                                                              projections)

                params = self.project_ellipsoid_parameters(params, projections)

                ellipsoid = PeakEllipsoid(*params,
                                          r_cut/3,
                                          self.params['Radius'],
                                          dQ)

                c0, c1, c2, *_ = params

                params = ellipsoid.fit(Q0, Q1, Q2, y, e)

                if params is not None:

                    c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

                    dx = 2*self.roi(r0, r1, r2, v0, v1, v2)

                    params = self.revert_ellipsoid_parameters(params,
                                                              projections)

                    if np.isclose(dx, 0).any():
                        params = None
                    elif np.all(np.abs(np.diff(extents, axis=1)/dx-1) < 0.15):
                        j = max_iter

            if params is not None:

                params = self.project_ellipsoid_parameters(params, projections)

                c, S, *fitting = ellipsoid.best_fit

                params = self.revert_ellipsoid_parameters(params, projections)

                peak.set_peak_shape(i, *params)

                int_intens, sig_noise = ellipsoid.intens_fit

                if (sig_noise > 3).all():

                    *_, binning, _ = fitting

                    (x0, x1, x2), dx, y, e = binning

                    bin_data_norm = (x0, x1, x2), dx, y, e

                    I, sigma = ellipsoid.integrate_norm(bin_data_norm, c, S)

                    peak.set_peak_intensity(i, I, sigma)

                    peak.add_diagonstic_info(i, ellipsoid.info)

                    plot.add_fitting(fitting)

                    vals = ellipsoid.interp_fit

                    plot.add_ellipsoid(c, S, vals)

                    plot.add_peak_intensity(int_intens, sig_noise)

                    goniometer = peak.get_goniometer_angles(i)

                    plot.add_peak_info(wavelength, angles, goniometer)

                    plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

                    peak_name = peak.get_peak_name(i)

                    plot.save_plot(self.get_plot_file(peak_name))

    def bin_axes(self, R, two_theta, az_phi):

        two_theta, az_phi = np.deg2rad(two_theta), np.deg2rad(az_phi)

        n = np.array([np.sin(two_theta)*np.cos(az_phi),
                      np.sin(two_theta)*np.sin(az_phi),
                      np.cos(two_theta)-1])

        n /= np.linalg.norm(n)

        v = np.array([0, 0, 1])

        u = np.cross(v, n)
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        return np.dot(R.T, n), np.dot(R.T, u), np.dot(R.T, v)

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

    def bin_extent(self, Q0, Q1, Q2,
                         r0, r1, r2,
                         v0, v1, v2, R, two_theta, az_phi, bin_size=0.01):

        n, u, v = self.bin_axes(R, two_theta, az_phi)

        projections = [n, u, v]

        params = Q0, Q1, Q2, r0, r1, r2, v0, v1, v2

        params = self.project_ellipsoid_parameters(params, projections)

        # if bin_size < 1e-4:
        #     bin_size = 0.001

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        dQ = self.roi(r0, r1, r2, v0, v1, v2)

        dQ0, dQ1, dQ2 = dQ

        extents = np.array([[Q0-dQ0, Q0+dQ0],
                            [Q1-dQ1, Q1+dQ1],
                            [Q2-dQ2, Q2+dQ2]])

        bin_sizes = np.array([bin_size, bin_size, bin_size])/2

        min_adjusted = np.floor(extents[:,0]/bin_sizes)*bin_sizes
        max_adjusted = np.ceil(extents[:,1]/bin_sizes)*bin_sizes

        bins = ((max_adjusted-min_adjusted)/bin_sizes).astype(int)

        bins = np.where(bins % 2 == 0, bins, bins+1)

        max_adjusted = min_adjusted+bins*bin_sizes

        extents = np.vstack((min_adjusted, max_adjusted)).T

        return bins, extents, projections

    def roi(self, r0, r1, r2, v0, v1, v2):

        V = np.diag([r0**2, r1**2, r2**2])
        U = np.column_stack([v0, v1, v2])

        S = np.dot(np.dot(U, V), U.T)

        dQ = 2*np.sqrt(np.diag(S))

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

        self.params.add('sigma', value=r_cut/6, min=0.01, max=r_cut/3)

    def model(self, x, A, sigma):

        z = x/sigma

        return A*(scipy.special.erf(z/np.sqrt(2)) -
                  np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2))

    def residual(self, params, x, y):

        A = params['A']
        sigma = params['sigma']

        y_fit = self.model(x, A, sigma)

        return y_fit-y

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


class PeakEllipsoid:

    def __init__(self, c0, c1, c2,
                       r0, r1, r2,
                       v0, v1, v2, delta, r_cut, bin_size):

        params = Parameters()

        self.params = params

        self.n, self.u, self.v = np.eye(3)

        if np.allclose(np.column_stack([v0, v1, v2]), np.eye(3)):
            v0, v1, v2 = self.n, self.u, self.v

        phi, theta, omega = self.angles(v0, v1, v2)

        self.update_constraints(c0, c1, c2,
                                r0, r1, r2,
                                phi, theta, omega,
                                delta, r_cut, bin_size)

    def update_constraints(self, c0, c1, c2, r0, r1, r2,
                           phi, theta, omega, delta, r_cut, bin_size):

        self.params.add('r0', value=r0, min=bin_size, max=r_cut, vary=True)
        self.params.add('r1', value=r1, min=bin_size, max=r_cut, vary=True)
        self.params.add('r2', value=r2, min=bin_size, max=r_cut, vary=True)

        self.params.add('c0', value=c0, min=c0-delta, max=c0+delta, vary=True)
        self.params.add('c1', value=c1, min=c1-delta, max=c1+delta, vary=True)
        self.params.add('c2', value=c2, min=c2-delta, max=c2+delta, vary=True)

        self.params.add('phi', value=phi, min=-np.pi, max=np.pi, vary=True)
        self.params.add('theta', value=theta, min=0, max=np.pi, vary=True)
        self.params.add('omega', value=omega, min=-np.pi, max=np.pi, vary=True)

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

    def centroid_covariance(self, c0, c1, c2, r0, r1, r2, phi, theta, omega):

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        c = np.array([c0, c1, c2])
        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        return c, S

    def residual(self, params, bin_1d, bin_2d, bin_3d):

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        c, S = self.centroid_covariance(c0, c1, c2,
                                        r0, r1, r2,
                                        phi, theta, omega)

        res = [self.residual_prof(params, c, S, bin_1d),
               self.residual_proj(params, c, S, bin_2d),
               self.residual_func(params, c, S, bin_3d)]

        diff = np.concatenate(res)

        return diff

    def residual_prof(self, params, c, S, xye, integrate=False):

        x, dx, y, e = xye

        A = params['A_1d'].value
        B = params['B_1d'].value

        mu, sigma = self.profile_params(c, S)

        args = x, A, B, mu, sigma, integrate

        y_fit = self.profile(*args)

        w = 1/e

        diff = ((y-y_fit)*w).ravel()

        return diff

    def residual_proj(self, params, c, S, xye, integrate=False):

        (xu, xv), (dxu, dxv), y, e = xye

        A = params['A_2d'].value
        B = params['B_2d'].value

        mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, S)

        args = xu, xv, A, B, mu_u, mu_v, sigma_u, sigma_v, rho, integrate

        y_fit = self.projection(*args)

        w = 1/e

        diff = ((y-y_fit)*w).ravel()

        return diff

    def residual_func(self, params, c, S, xye, integrate=False):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = xye

        A = params['A_3d'].value
        B = params['B_3d'].value

        args = x0, x1, x2, A, B, c, S, integrate

        y_fit = self.func(*args)

        w = 1/e

        diff = ((y-y_fit)*w).ravel()

        return diff

    def residual_integral(self, params, binning, integrate):

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        c, S = self.centroid_covariance(c0, c1, c2,
                                        r0, r1, r2,
                                        phi, theta, omega)

        return self.residual_func(params, c, S, binning, integrate=integrate)

    def func(self, Q0, Q1, Q2, A, B, c, S, integrate=False):

        y = self.generalized3d(Q0, Q1, Q2, c, S, integrate)

        return A*y+B

    def func_grad(self, Q0, Q1, Q2, A, B, c, S, integrate=False):

        y = self.generalized3d(Q0, Q1, Q2, c, S, integrate)

        inv_S = np.linalg.inv(S)

        coeff = np.einsum('ij,j...->i...', inv_S, [Q0-c[0], Q1-c[1], Q2-c[2]])

        return -A*coeff[0]*y, -A*coeff[1]*y, -A*coeff[2]*y

    def projection(self, Qu, Qv, A, B, mu_u, mu_v, \
                         sigma_u, sigma_v, rho, integrate=False):

        y = self.generalized2d(Qu, Qv, mu_u, mu_v,
                               sigma_u, sigma_v, rho, integrate)

        return A*y+B

    def projection_grad(self, Qu, Qv, A, B, mu_u, mu_v, \
                              sigma_u, sigma_v, rho, integrate=False):

        y = self.generalized2d(Qu, Qv, mu_u, mu_v,
                               sigma_u, sigma_v, rho, integrate)

        u = (Qu-mu_u)/(sigma_u*(1-rho**2))
        v = (Qv-mu_v)/(sigma_v*(1-rho**2))

        return -A*(u-rho*v)/sigma_u*y, -A*(v-rho*u)/sigma_v*y

    def profile(self, Q, A, B, mu, sigma, integrate=False):

        y = self.generalized1d(Q, mu, sigma, integrate)

        return A*y+B

    def profile_grad(self, Q, A, B, mu, sigma, integrate=False):

        y = self.generalized1d(Q, mu, sigma, integrate)

        return -A*(Q-mu)/sigma**2*y

    def profile_params(self, c, S):

        mu = np.dot(c, self.n)

        s = np.sqrt(np.dot(np.dot(S, self.n), self.n))

        return mu, s

    def projection_params(self, c, S):

        mu_u = np.dot(c, self.u)
        mu_v = np.dot(c, self.v)

        s0 = np.sqrt(np.dot(np.dot(S, self.u), self.u))
        s1 = np.sqrt(np.dot(np.dot(S, self.v), self.v))
        s01 = np.dot(np.dot(S, self.u), self.v)

        corr = s01/(s0*s1)

        return mu_u, mu_v, s0, s1, corr

    def generalized3d(self, Q0, Q1, Q2, c, S, integrate):

        mu0, mu1, mu2 = c

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        inv_S = self.inv_3d(S)

        dx = [x0, x1, x2]

        d2 = np.einsum('i...,ij,j...->...', dx, inv_S, dx)

        scale = np.sqrt(np.linalg.det(2*np.pi*S)) if integrate else 1

        return np.exp(-0.5*d2)/scale

    def generalized2d(self, Qu, Qv, mu_u, mu_v,
                      sigma_u, sigma_v, rho, integrate):

        xu, xv = Qu-mu_u, Qv-mu_v

        S = np.array([[sigma_u**2, sigma_u*sigma_v*rho],
                      [sigma_u*sigma_v*rho, sigma_v**2]])

        inv_S = self.inv_2d(S)

        dx = [xu, xv]

        d2 = np.einsum('i...,ij,j...->...', dx, inv_S, dx)

        scale = np.sqrt(np.linalg.det(2*np.pi*S)) if integrate else 1

        return np.exp(-0.5*d2)/scale

    def generalized1d(self, Q, mu, sigma, integrate):

        x = (Q-mu)/sigma

        scale = np.sqrt(2*np.pi*sigma**2) if integrate else 1

        return np.exp(-0.5*x**2)/scale

    def inv_3d(self, A):

        a, d, e = A[0,0], A[0,1], A[0,2]
        b, f    =         A[1,1], A[1,2]
        c       =                 A[2,2]

        det_A = a*(b*c-f*f)-d*(d*c-e*f)+e*(d*f-e*b)

        inv_A = np.array([[b*c-f*f, e*f-d*c, d*f-e*b],
                          [e*f-d*c, a*c-e*e, e*d-a*f],
                          [d*f-e*b, e*d-a*f, a*b-d*d]])/det_A

        return inv_A

    def inv_2d(self, A):

        a, c = A[0,0], A[0,1]
        b    =         A[1,1]

        det_A = a*b-c*c

        inv_A = np.array([[b, -c], [-c, a]])/det_A

        return inv_A

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def bin1d(self, x0, x1, x2, y, e):

        x = x0[:,0,0]

        dx = x[1]-x[0]

        xu = x1[0,:,0]
        xv = x2[0,0,:]

        du = xu[1]-xu[0]
        dv = xv[1]-xv[0]

        y[np.isinf(y)] = np.nan
        e[np.isinf(e)] = np.nan

        d2x = du*dv

        y = np.nansum(y, axis=(1,2))*d2x
        e = np.sqrt(np.nansum(e**2, axis=(1,2)))*d2x

        return x, dx, y, e

    def bin2d(self, x0, x1, x2, y, e):

        x = x0[:,0,0]

        dx = x[1]-x[0]

        xu = x1[0,:,0]
        xv = x2[0,0,:]

        du = xu[1]-xu[0]
        dv = xv[1]-xv[0]

        xu, xv = np.meshgrid(xu, xv, indexing='ij')

        y = np.nansum(y, axis=0)*dx
        e = np.sqrt(np.nansum(e**2, axis=0))*dx

        return (xu, xv), (du, dv), y, e

    def fit(self, x0, x1, x2, y_norm, e_norm):

        mask = (y_norm > 0) & (e_norm > 0)

        self.params.add('A_1d', value=0, min=0, max=1, vary=False)
        self.params.add('A_2d', value=0, min=0, max=1, vary=False)
        self.params.add('A_3d', value=0, min=0, max=1, vary=False)

        self.params.add('B_1d', value=0, min=0, max=np.inf, vary=False)
        self.params.add('B_2d', value=0, min=0, max=np.inf, vary=False)
        self.params.add('B_3d', value=0, min=0, max=np.inf, vary=False)

        if mask.sum() > 34 and (np.array(mask.shape) >= 5).all():

            x = [x0, x1, x2]

            y = y_norm
            e = e_norm

            mask = np.isfinite(e) & np.isfinite(y) & (e > 0)

            Q0, Q1, Q2, y, e = x0[mask], x1[mask], x2[mask], y[mask], e[mask]

            dQ0, dQ1, dQ2 = self.voxels(x0, x1, x2)

            bin_3d = (Q0, Q1, Q2), (dQ0, dQ1, dQ2), y, e

            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_3d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_3d'].set(value=y_min, min=0, max=y_max, vary=True)

            bin_1d = self.bin1d(x0, x1, x2, y_norm, e_norm)
            bin_2d = self.bin2d(x0, x1, x2, y_norm, e_norm)

            Q, dQ, y, e = bin_1d

            mask = np.isfinite(e) & np.isfinite(y) & (e > 0)

            if mask.sum() < 5:
                return None

            Q, y, e = Q[mask], y[mask], e[mask]

            bin_1d = Q, dQ, y, e

            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_1d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_1d'].set(value=y_min, min=0, max=y_max, vary=True)

            (Qu, Qv), (dQu, dQv), y, e = bin_2d

            mask = np.isfinite(e) & np.isfinite(y) & (e > 0)

            if mask.sum() < 5:
                return None

            Qu, Qv, y, e = Qu[mask], Qv[mask], y[mask], e[mask]

            bin_2d = (Qu, Qv), (dQu, dQv), y, e

            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_2d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_2d'].set(value=y_min, min=0, max=y_max, vary=True)

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(bin_1d, bin_2d, bin_3d),
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

            self.params.pop('r0')
            self.params.pop('r1')
            self.params.pop('r2')

            self.params.pop('phi')
            self.params.pop('theta')
            self.params.pop('omega')

            self.params.pop('c0')
            self.params.pop('c1')
            self.params.pop('c2')

            self.params['B_1d'].set(vary=False)
            self.params['B_2d'].set(vary=False)
            self.params['B_3d'].set(vary=False)

            B_1d = self.params['B_1d'].value
            B_2d = self.params['B_2d'].value
            B_3d = self.params['B_3d'].value

            B_1d_err = self.params['B_1d'].stderr
            if B_1d_err is None:
                B_1d_err = 0.0

            B_2d_err = self.params['B_2d'].stderr
            if B_2d_err is None:
                B_2d_err = 0.0

            B_3d_err = self.params['B_3d'].stderr
            if B_3d_err is None:
                B_3d_err = 0.0

            self.bkg_1d = B_1d, B_1d_err
            self.bkg_2d = B_2d, B_2d_err
            self.bkg_3d = B_3d, B_3d_err

            x = [x0, x1, x2]
            dx = [dQ0, dQ1, dQ2]

            y = y_norm
            e = e_norm

            bin_3d = x, dx, y, e

            (Q0, Q1, Q2), dx, y, e = bin_3d

            bin_1d = self.bin1d(x0, x1, x2, y_norm, e_norm)
            bin_2d = self.bin2d(x0, x1, x2, y_norm, e_norm)

            c, cov = self.centroid_covariance(c0, c1, c2,
                                              r0, r1, r2,
                                              phi, theta, omega)

            S = self.S_matrix(r0, r1, r2, phi, theta, omega)

            mu, sigma = self.profile_params(c, cov)
            mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, cov)

            r, ru, rv = np.array([sigma, sigma_u, sigma_v])*4

            R = [ru, rv, rho]

            val_1d = self.integrate_profile(bin_1d, mu, r)
            val_2d = self.integrate_projection(bin_2d, [mu_u, mu_v], R)
            val_3d = self.integrate(bin_3d, c, S)

            if val_1d is not None:
                A_1d, A_1d_sig, B_1d, *p1 = val_1d
            else:
                return None

            if val_2d is not None:
                A_2d, A_2d_sig, B_2d, *p2 = val_2d
            else:
                return None

            if val_3d is not None:
                A_3d, A_3d_sig, B_3d, *p3 = val_3d
            else:
                return None

            self.fit_est = p3

            self.params.pop('A_1d')
            self.params.pop('A_2d')
            self.params.pop('A_3d')

            self.params.pop('B_1d')
            self.params.pop('B_2d')
            self.params.pop('B_3d')

            A = np.array([A_1d, A_2d, A_3d])
            A_sig = np.array([A_1d_sig, A_2d_sig, A_3d_sig])

            self.intens_fit = A, A/A_sig

            args = Q0, Q1, Q2, A_3d, B_3d, *p3, True

            y_3d_fit = self.func(*args)

            (Qu, Qv), (dQu, dQv), y, e = bin_2d

            args = Qu, Qv, A_2d, B_2d, *p2, True

            y_2d_fit = self.projection(*args)

            Q, dQ, y, e = bin_1d

            args = Q, A_1d, B_1d, *p1, True

            y_1d_fit = self.profile(*args)

            fitting = bin_1d, y_1d_fit, bin_2d, y_2d_fit, bin_3d, y_3d_fit

            self.best_fit = c, S, *fitting

            x = A_1d, B_1d, A_2d, B_2d, A_3d, B_3d

            self.interp_fit = mu, mu_u, mu_v, r, ru, rv, rho, *x

            U = self.U_matrix(phi, theta, omega)

            v0, v1, v2 = U.T

            return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_profile(self, bins, c, r):

        x, dx, y, e = bins

        pk = np.abs(x-c)/r < 1

        struct = scipy.ndimage.generate_binary_structure(1, 1)
        pk = scipy.ndimage.binary_dilation(pk, struct, border_value=0)

        pk = pk & (e >= 0)

        bkg, bkg_err = self.bkg_1d

        freq = y[pk]-bkg
        var = e[pk]**2+bkg_err**2

        intens = np.nansum(freq)*dx
        sig = np.sqrt(np.nansum(var))*dx

        if intens <= sig:
            return None

        wgt = freq.copy()
        wgt[wgt < 0] = 0

        if not wgt.sum() > 0:
            return None

        mu = np.average(x[pk], weights=wgt)

        sigma = np.sqrt(np.average((x[pk]-mu)**2, weights=wgt))

        return intens, sig, bkg, mu, sigma

    def integrate_projection(self, bins, c, R):

        (xu, xv), dx, y, e = bins

        S = np.array([[R[0]**2, np.prod(R)], [np.prod(R), R[1]**2]])

        x = np.array([xu-c[0], xv-c[1]])

        pk = np.einsum('ij,jkl,ikl->kl', np.linalg.inv(S), x, x) < 1

        struct = scipy.ndimage.generate_binary_structure(2, 1)
        pk = scipy.ndimage.binary_dilation(pk, struct, border_value=0)

        pk = pk & (e >= 0)

        bkg, bkg_err = self.bkg_2d

        d2x = np.prod(dx)

        freq = y[pk]-bkg
        var = e[pk]**2+bkg_err**2

        intens = np.nansum(freq)*d2x
        sig = np.sqrt(np.nansum(var))*d2x

        if intens <= sig:
            return None

        wgt = freq.copy()
        wgt[wgt < 0] = 0

        if not wgt.sum() > 0:
            return None

        mu_u = np.average(xu[pk], weights=wgt)
        mu_v = np.average(xv[pk], weights=wgt)

        sigma_u = np.sqrt(np.average((xu[pk]-mu_u)**2, weights=wgt))
        sigma_v = np.sqrt(np.average((xv[pk]-mu_v)**2, weights=wgt))

        rho = np.average((xu[pk]-mu_u)*\
                         (xv[pk]-mu_v), weights=wgt)/(sigma_u*sigma_v)

        return intens, sig, bkg, mu_u, mu_v, sigma_u, sigma_v, rho

    def integrate(self, bins, c, S):

        (x0, x1, x2), dx, y, e = bins

        x = np.array([x0-c[0], x1-c[1], x2-c[2]])

        pk = np.einsum('ij,jklm,iklm->klm', np.linalg.inv(S), x, x) < 1

        struct = scipy.ndimage.generate_binary_structure(3, 1)
        pk = scipy.ndimage.binary_dilation(pk, struct, border_value=0)

        pk = pk & (e >= 0)

        bkg, bkg_err = self.bkg_3d

        d3x = np.prod(dx)

        freq = y[pk]-bkg
        var = e[pk]**2+bkg_err**2

        intens = np.nansum(freq)*d3x
        sig = np.sqrt(np.nansum(var))*d3x

        if intens <= sig:
            return None

        wgt = freq.copy()
        wgt[wgt < 0] = 0

        if not wgt.sum() > 0:
            return None

        mu0 = np.average(x0[pk], weights=wgt)
        mu1 = np.average(x1[pk], weights=wgt)
        mu2 = np.average(x2[pk], weights=wgt)

        s0 = np.average((x0[pk]-mu0)**2, weights=wgt)
        s1 = np.average((x1[pk]-mu1)**2, weights=wgt)
        s2 = np.average((x2[pk]-mu2)**2, weights=wgt)

        s01 = np.average((x0[pk]-mu0)*(x1[pk]-mu1), weights=wgt)
        s02 = np.average((x0[pk]-mu0)*(x2[pk]-mu2), weights=wgt)
        s12 = np.average((x1[pk]-mu1)*(x2[pk]-mu2), weights=wgt)

        c = np.array([mu0, mu1, mu2])
        S = np.array([[s0, s01, s02], [s01, s1, s12], [s02, s12, s2]])

        return intens, sig, bkg, c, S

    def integrate_norm(self, bins, c, S):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = bins

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)

        pk = (ellipsoid <= 1) & (e > 0)
        bkg = (ellipsoid > 1) & (ellipsoid <= 1.5**2) & (e > 0)

        dilate = pk | bkg

        d3x = dx0*dx1*dx2

        # y_bkg = y[bkg]
        # e_bkg = e[bkg]

        # w_bkg = 1/e_bkg**2

        # if len(w_bkg) > 2:
        #     b = self.weighted_median(y_bkg, w_bkg)
        #     b_err = self.jackknife_uncertainty(y_bkg, w_bkg)
        # else:
        #     b = b_err = 0.0

        n_pk = y[pk].size
        n_bkg = y[bkg].size

        if n_bkg == 0:
            n_bkg = 1

        vol_ratio = n_pk/n_bkg

        b = np.nansum(y[bkg])
        b_err = np.sqrt(np.nansum(e[bkg]**2))

        self.info = [b/n_bkg, b_err/n_bkg]

        intens = (np.nansum(y[pk])-b*vol_ratio)#*d3x
        sig = np.sqrt(np.nansum(e[pk]**2)-(b_err*vol_ratio)**2)#*d3x

        # d = y**2/e**2
        n = y/e**2

        vals = y*n
        norm = n#/d3x

        mask = vals[pk] > 0
        wgt = vals[pk][mask]

        sum_d = np.nansum(vals[pk])
        err_d = np.sqrt(np.nansum(vals[pk]))

        if not wgt.sum() > 0:
            ave_n = np.average(norm[pk][mask])
            sig_n = np.sqrt(np.average((norm[pk][mask]-ave_n)**2))
        else:
            ave_n = np.average(norm[pk][mask], weights=wgt)
            sig_n = np.sqrt(np.average((norm[pk][mask]-ave_n)**2, weights=wgt))

        info = [sum_d, err_d, ave_n, sig_n, d3x]

        int_intens, sig_noise = self.intens_fit

        info += int_intens.tolist()
        info += sig_noise.tolist()

        self.info += info

        freq = y-b/n_bkg
        freq[~dilate] = np.nan

        if not np.isfinite(sig):
            sig = intens

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq, vals, norm

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        return intens, sig

    def envelope(self, x0, x1, x2, c0, c1, c2, r0, r1, r2, v0, v1, v2):

        W = np.column_stack([v0, v1, v2])
        V = np.diag([1/r0**2, 1/r1**2, 1/r2**2])

        A = (W @ V) @ W.T

        dx = np.array([x0-c0, x1-c1, x2-c2])

        dist = np.einsum('ij,jklm,iklm->klm', A, dx, dx)

        return dist < 1

    def volume_fraction(self, x0, x1, x2, d, n, *params):

        pk = self.envelope(x0, x1, x2, *params)

        y = d[pk]/n[pk]

        return np.sum(np.isfinite(y))/y.size

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

    def smooth_bins(self, data):

        return scipy.ndimage.gaussian_filter(data, sigma=1)