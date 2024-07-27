import os

import numpy as np

import scipy.spatial.transform
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.ndimage
# import scipy.sparse

import astropy.convolution
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
                                  r_cut,
                                  method='sphere')

            # peaks.remove_weak_peaks('peaks')

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
                              method='sphere',
                              centroid=False)

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

            j, max_iter = 0, 2

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

                c0, c1, c2, *_ = params

                ellipsoid = PeakEllipsoid(c0, c1, c2, r_cut, r_cut)

                params = ellipsoid.fit(Q0, Q1, Q2, y, e, dQ)

                if params is not None:

                    c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

                    dx = 2*self.roi(r0, r1, r2, v0, v1, v2)

                    params = self.revert_ellipsoid_parameters(params,
                                                              projections)

                    # if np.isclose(dx, 0).any():
                    #     params = None
                    # elif np.all(np.abs(np.diff(extents, axis=1)/dx-1) < 0.15):
                    #     j = max_iter

            if params is not None:

                params = self.project_ellipsoid_parameters(params, projections)

                c, S, *fitting = ellipsoid.best_fit

                params = self.revert_ellipsoid_parameters(params, projections)

                peak.set_peak_shape(i, *params)

                int_intens, sig_noise = ellipsoid.intens_fit

                if (sig_noise > 3).all():

                    bin_data = ellipsoid.bin_data

                    I, sigma = ellipsoid.integrate_norm(bin_data, c, S)

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

        v = np.array([0, 1, 0])

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

        Q0, Q1, Q2, r0, r1, r2, v0, v1, v2 = params

        dQ = self.roi(r0, r1, r2, v0, v1, v2)

        dQ0, dQ1, dQ2 = dQ

        extents = np.array([[Q0-dQ0, Q0+dQ0],
                            [Q1-dQ1, Q1+dQ1],
                            [Q2-dQ2, Q2+dQ2]])

        # bin_sizes = np.array([bin_size, bin_size, bin_size])
        bin_sizes = np.array([dQ0, dQ1, dQ2])/15

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

        self.params.add('sigma', value=r_cut/6, min=0.01, max=r_cut/4)

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

    def __init__(self, c0, c1, c2, delta, r_cut):

        self.n, self.u, self.v = np.eye(3)

        self.params = Parameters()

        self.params.add('c0', value=c0, min=c0-delta, max=c0+delta)
        self.params.add('c1', value=c1, min=c1-delta, max=c1+delta)
        self.params.add('c2', value=c2, min=c2-delta, max=c2+delta)

        self.params.add('r0', value=r_cut, min=r_cut*0.1, max=r_cut*1.5)
        self.params.add('r1', value=r_cut, min=r_cut*0.1, max=r_cut*1.5)
        self.params.add('r2', value=r_cut, min=r_cut*0.1, max=r_cut*1.5)

        self.params.add('phi', value=0, min=-np.pi, max=np.pi)
        self.params.add('theta', value=np.pi, min=0, max=np.pi)
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

    def centroid_covariance(self, c0, c1, c2, r0, r1, r2, phi, theta, omega):

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        c = np.array([c0, c1, c2])
        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        return c, S

    def integrate_1d(self, x0, x1, x2, y, e, b, c, mu, sigma):

        x = x0[:,0,0].copy()
        xu, xv = x1[0,:,0].copy(), x2[0,0,:].copy()

        dx = x[1]-x[0]
        d2x = (xu[1]-xu[0])*(xv[1]-xv[0])

        #w = 1/e**2
        #w[~np.isfinite(w)] = np.nan

        #vol_fract = np.mean(w > 0, axis=(1,2))

        y1 = np.nansum(y, axis=(1,2))*d2x#-b-c*(x-mu)
        e1 = np.sqrt(np.nansum(e**2, axis=(1,2)))*d2x

        mask = np.isfinite(y1) & np.isfinite(e1)

        y1[~mask] = np.nan
        e1[~mask] = np.nan

        r = 4*sigma

        pk = (np.abs(x-mu)/r < 1)
        bkg = ~pk

        pk = pk & (e1 > 0) & np.isfinite(e1)
        bkg = bkg & (e1 > 0) & np.isfinite(e1)

        if bkg.sum() == 0:
            b = b_err = np.nanmin(y1)
        else:
            b = np.nansum(y1[bkg]/e1[bkg]**2)/np.nansum(1/e1[bkg]**2)
            b_err = 1/np.sqrt(np.nansum(1/e1[bkg]**2))

        b = b_err = 0

        intens = np.nansum(y1[pk]-b)*dx
        sig = np.sqrt(np.nansum(e1[pk]**2+b_err**2))*dx

        return intens, sig, b, b_err, y1, e1

    def integrate_2d(self, x0, x1, x2, y, e, b, cu, cv, 
                          mu_u, mu_v, s_u, s_v, corr):

        x = x0[:,0,0].copy()
        xu, xv = x1[0,:,0].copy(), x2[0,0,:].copy()

        dx = x[1]-x[0]
        d2x = (xu[1]-xu[0])*(xv[1]-xv[0])
        xu, xv = np.meshgrid(xu, xv, indexing='ij')

        S = np.array([[s_u**2, s_u*s_v*corr], [s_u*s_v*corr, s_v**2]])

        x = np.array([xu-mu_u, xv-mu_v])

        #w = 1/e**2
        #w[~np.isfinite(w)] = np.nan

        #vol_fract = np.mean(w > 0, axis=0)

        y2 = np.nansum(y, axis=0)*dx#-b-cu*(xu-mu_u)-cv*(xv-mu_v)#/vol_fract
        e2 = np.sqrt(np.nansum(e**2, axis=0))*dx#/vol_fract

        mask = np.isfinite(y2) & np.isfinite(e2)

        y2[~mask] = np.nan
        e2[~mask] = np.nan

        A = 16*S

        pk = (np.einsum('ij,jkl,ikl->kl', self.inv_2d(A), x, x) < 1)
        bkg = ~pk

        pk = pk & (e2 > 0) & np.isfinite(e2)
        bkg = bkg & (e2 > 0) & np.isfinite(e2)

        if bkg.sum() == 0:
            b = b_err = np.nanmin(y2)
        else:
            b = np.nansum(y2[bkg]/e2[bkg]**2)/np.nansum(1/e2[bkg]**2)
            b_err = 1/np.sqrt(np.nansum(1/e2[bkg]**2))

        b = b_err = 0

        intens = np.nansum(y2[pk]-b)*d2x
        sig = np.sqrt(np.nansum(e2[pk]**2+b_err**2))*d2x

        return intens, sig, b, b_err, y2, e2

    def integrate_3d(self, x0, x1, x2, y, e, b, c, S):

        d3x = self.voxel_volume(x0, x1, x2)

        x = np.array([x0-c[0], x1-c[1], x2-c[2]])

        y3 = y#-b
        e3 = np.sqrt(e.copy()**2)

        mask = np.isfinite(y3) & np.isfinite(e3)

        y3[~mask] = np.nan
        e3[~mask] = np.nan

        A = 16*S

        pk = (np.einsum('ij,jklm,iklm->klm', self.inv_3d(A), x, x) < 1)
        bkg = ~pk

        pk = pk & (e3 > 0) & np.isfinite(e3)
        bkg = bkg & (e3 > 0) & np.isfinite(e3)

        if bkg.sum() == 0:
            b = b_err = np.nanmin(y3)
        else:
            b = np.nansum(y3[bkg]/e3[bkg]**2)/np.nansum(1/e3[bkg]**2)
            b_err = 1/np.sqrt(np.nansum(1/e3[bkg]**2))

        b = b_err = 0

        intens = np.nansum(y3[pk]-b)*d3x
        sig = np.sqrt(np.nansum(e3[pk]**2+b_err**2))*d3x

        return intens, sig, b, b_err, y3, e3

    def objective(self, params, x0, x1, x2, dx0, dx1, dx2,
                        y_1d, e_1d, y_2d, e_2d, y_3d, e_3d):

        a1d = params['a1d'].value
        a2d = params['a2d'].value
        a3d = params['a3d'].value

        b1d = params['b1d'].value
        b2d = params['b2d'].value
        b3d = params['b3d'].value

        c1d = params['c1d'].value
        c2d1 = params['c2d1'].value
        c2d2 = params['c2d2'].value

        c0 = params['c0'].value
        c1 = params['c1'].value
        c2 = params['c2'].value

        r0 = params['r0'].value
        r1 = params['r1'].value
        r2 = params['r2'].value

        phi = params['phi'].value
        theta = params['theta'].value
        omega = params['omega'].value

        c, S = self.centroid_covariance(c0, c1, c2,
                                        r0, r1, r2,
                                        phi, theta, omega)

        mu, sigma = self.profile_params(c, S)

        mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, S)

        x = x0[:,0,0].copy()
        xu, xv = x1[0,:,0].copy(), x2[0,0,:].copy()
        xu, xv = np.meshgrid(xu, xv, indexing='ij')

        y_1d_fit = self.profile(x, a1d, b1d, c1d, mu, sigma)

        y_2d_fit = self.projection(xu, xv, a2d, b2d, c2d1, c2d2, 
                                   mu_u, mu_v, sigma_u, sigma_v, rho)

        y_3d_fit = self.peak(x0, x1, x2, a3d, b3d, c, S)

        mask_1d = np.isfinite(y_1d) & np.isfinite(e_1d) & (e_1d > 0)
        mask_2d = np.isfinite(y_2d) & np.isfinite(e_2d) & (e_2d > 0)
        mask_3d = np.isfinite(y_3d) & np.isfinite(e_3d) & (e_3d > 0)

        res = [((y_1d-y_1d_fit)/e_1d)[mask_1d]/np.sum(mask_1d), #
               ((y_2d-y_2d_fit)/e_2d)[mask_2d]/np.sum(mask_2d), #
               ((y_3d-y_3d_fit)/e_3d)[mask_3d]/np.sum(mask_3d)] #

        return np.concatenate(res)

    def peak(self, Q0, Q1, Q2, A, B, c, S, integrate=True):

        y = self.generalized3d(Q0, Q1, Q2, c, S, integrate)

        return A*y+B

    def projection(self, Qu, Qv, A, B, Cu, Cv, mu_u, mu_v, \
                         sigma_u, sigma_v, rho, integrate=True):

        y = self.generalized2d(Qu, Qv, mu_u, mu_v,
                               sigma_u, sigma_v, rho, integrate)

        return A*y+B+Cu*(Qu-mu_u)+Cv*(Qv-mu_v)

    def profile(self, Q, A, B, C, mu, sigma, integrate=True):

        y = self.generalized1d(Q, mu, sigma, integrate)

        return A*y+B+C*(Q-mu)

    def profile_params(self, c, S):

        mu = np.dot(c, self.n)

        s = np.sqrt(np.dot(np.dot(S, self.n), self.n))

        return mu, s

    def projection_params(self, c, S):

        mu_u = np.dot(c, self.u)
        mu_v = np.dot(c, self.v)

        s_u = np.sqrt(np.dot(np.dot(S, self.u), self.u))
        s_v = np.sqrt(np.dot(np.dot(S, self.v), self.v))
        s_uv = np.dot(np.dot(S, self.u), self.v)

        corr = s_uv/(s_u*s_v)

        return mu_u, mu_v, s_u, s_v, corr

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

    def backfill_invalid(self, data, x0, x1, x2, dx):

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d0 = dx/dx0/2
        d1 = dx/dx1/2
        d2 = dx/dx2/2

        k0 = astropy.convolution.Gaussian1DKernel(d0).array
        k1 = astropy.convolution.Gaussian1DKernel(d1).array
        k2 = astropy.convolution.Gaussian1DKernel(d2).array

        k = k0*k1.reshape((-1,1))*k2.reshape((-1,1,1))

        return astropy.convolution.convolve(data, k, boundary='extend')

    def fit(self, x0, x1, x2, y_norm, e_norm, dQ):

        mask = (y_norm > 0) & (e_norm > 0)

        if mask.sum() > 21 and (np.array(mask.shape) >= 5).all():

            y = y_norm.copy()
            e = e_norm.copy()

            y[~mask] = np.nan
            e[~mask] = np.nan

            #y = self.backfill_invalid(y, x0, x1, x2, dQ)
            #e = np.sqrt(self.backfill_invalid(e**2, x0, x1, x2, dQ))

            mask = np.isfinite(e) & np.isfinite(y) & (e > 0)

            Q0, Q1, Q2 = x0.copy(), x1.copy(), x2.copy()

            y[~mask] = np.nan
            e[~mask] = np.nan

            dQ0, dQ1, dQ2 = self.voxels(x0, x1, x2)

            dx0, dx1, dx2 = dQ0, dQ1, dQ2

            x = x0[:,0,0].copy()
            xu, xv = x1[0,:,0].copy(), x2[0,0,:].copy()
            xu, xv = np.meshgrid(xu, xv, indexing='ij')

            d1x = dx0
            d2x = dx1*dx2
            d3x = dx0*dx1*dx2

            y_1d = np.nansum(y, axis=(1,2))*d2x
            y_2d = np.nansum(y, axis=0)*d1x
            y_3d = y.copy()

            e_1d = np.sqrt(np.nansum(e**2, axis=(1,2)))*d2x
            e_2d = np.sqrt(np.nansum(e**2, axis=0))*d1x
            e_3d = e.copy()

            a1_max = np.nansum(y_1d)
            a2_max = np.nansum(y_2d)
            a3_max = np.nansum(y_3d)

            b1_max = np.nanmax(y_1d)
            b2_max = np.nanmax(y_2d)
            b3_max = np.nanmax(y_3d)

            b1_min = np.nanmin(y_1d)
            b2_min = np.nanmin(y_2d)
            b3_min = np.nanmin(y_3d)

            self.params.add('a1d', value=a1_max, min=0, max=10*a1_max)
            self.params.add('a2d', value=a2_max, min=0, max=10*a2_max)
            self.params.add('a3d', value=a3_max, min=0, max=10*a3_max)

            # self.params.add('a2d', expr='a1d')
            # self.params.add('a3d', expr='a1d')

            self.params.add('b1d', value=b1_min, min=0, max=b1_max)
            self.params.add('b2d', value=b2_min, min=0, max=b2_max)
            self.params.add('b3d', value=b3_min, min=0, max=b3_max)

            self.params.add('c1d', value=0, min=-b1_max, max=b1_max)
            self.params.add('c2d1', value=0, min=-b2_max, max=b2_max)
            self.params.add('c2d2', value=0, min=-b2_max, max=b2_max)

            args = (Q0, Q1, Q2, dQ0, dQ1, dQ2, 
                    y_1d, e_1d, y_2d, e_2d, y_3d, e_3d)

            out = Minimizer(self.objective,
                            self.params,
                            fcn_args=args,
                            reduce_fcn='negentropy',
                            nan_policy='omit')

            result = out.minimize(method='leastsq')

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

            y, e = y_norm.copy(), e_norm.copy()

            mask = (y_norm > 0) & (e_norm > 0)

            y[~mask] = np.nan
            e[~mask] = np.nan

            a1d = self.params['a1d'].value
            a2d = self.params['a2d'].value
            a3d = self.params['a3d'].value

            a1d_err = self.params['a1d'].stderr
            a2d_err = self.params['a2d'].stderr
            a3d_err = self.params['a3d'].stderr

            if a1d_err is None:
                a1d_err = a1d
            if a2d_err is None:
                a2d_err = a2d
            if a3d_err is None:
                a3d_err = a3d

            b1d = self.params['b1d'].value
            b2d = self.params['b2d'].value
            b3d = self.params['b3d'].value

            # print(b1d/a1d)
            # print(b2d/a2d)
            # print(b3d/a3d)

            c1d = self.params['c1d'].value
            c2d1 = self.params['c2d1'].value
            c2d2 = self.params['c2d2'].value

            # y = y_norm.copy()
            # e = e_norm.copy()

            # y_1d = np.nansum(y, axis=(1,2))*d2x
            # y_2d = np.nansum(y, axis=0)*d1x
            # y_3d = y.copy()

            # e_1d = np.sqrt(np.nansum(e**2, axis=(1,2)))*d2x
            # e_2d = np.sqrt(np.nansum(e**2, axis=0))*d1x
            # e_3d = e.copy()

            c, S = self.centroid_covariance(c0, c1, c2,
                                            r0, r1, r2,
                                            phi, theta, omega)

            U = self.U_matrix(phi, theta, omega)

            v0, v1, v2 = U.T

            mu, sigma = self.profile_params(c, S)

            mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, S)

            r, ru, rv = np.array([sigma, sigma_u, sigma_v])*4

            y_1d_fit = self.profile(x, a1d, b1d, c1d, mu, sigma)

            y_2d_fit = self.projection(xu, xv, a2d, b2d, c2d1, c2d2, \
                                       mu_u, mu_v, sigma_u, sigma_v, rho)

            y_3d_fit = self.peak(x0, x1, x2, a3d, b3d, c, S)

            bin_3d = (x0, x1, x2), (dx0, dx1, dx2), y_3d, e_3d
            bin_2d = (xu, xv), (dx1, dx2), y_2d, e_2d
            bin_1d = x, dx0, y_1d, e_1d

            A = np.array([a1d, a2d, a3d])/d3x
            A_sig = np.array([a1d_err, a2d_err, a3d_err])/d3x

            self.intens_fit = A, A/A_sig

            fitting = bin_1d, y_1d_fit, bin_2d, y_2d_fit, bin_3d, y_3d_fit

            self.best_fit = c, S*16, *fitting

            x = a1d, b1d, a2d, b2d, a3d, b3d

            self.interp_fit = mu, mu_u, mu_v, r, ru, rv, rho, *x

            y = y_norm.copy()
            e = e_norm.copy()

            self.bin_data = (x0, x1, x2), (dx0, dx1, dx2), y, e

            return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_norm(self, bins, c, S):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = bins

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)

        pk = (ellipsoid <= 1) & (e >= 0) & (y >= 0)
        bkg = (ellipsoid > 1) & (ellipsoid <= 1.5**2) & (e >= 0) & (y >= 0)

        dilate = pk | bkg

        d3x = dx0*dx1*dx2

        y_bkg = y[bkg]
        e_bkg = e[bkg]

        b = np.nansum(y_bkg)
        b_err = np.sqrt(np.nansum(e_bkg**2))

        vol_ratio = np.sum(pk)/np.sum(bkg)

        b *= vol_ratio
        b_err *= vol_ratio

        self.info = [b, b_err]

        intens = np.nansum(y[pk])-b
        sig = np.sqrt(np.nansum(e[pk]**2)+b_err**2)

        # print(vol_ratio)
        # print(np.round((np.nansum(y[pk])/intens-1)*100, 2))
        # print(np.round(intens/sig, 2))

        n = y/e**2

        vals = y*n
        norm = n

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

        freq = y.copy()
        freq[~dilate] = np.nan

        if not np.isfinite(sig):
            sig = intens

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq

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