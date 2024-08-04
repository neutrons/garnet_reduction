import os

import numpy as np

import scipy.spatial.transform
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.ndimage
import scipy.linalg

from sklearn.cluster import DBSCAN
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

            det_id = peak.get_detector_id(i)

            two_theta, az_phi = angles

            R = peak.get_goniometer_matrix(i)

            dQ = data.get_resolution_in_Q(wavelength, two_theta)

            bins, extents, projections = self.bin_extent(*params,
                                                         R,
                                                         two_theta,
                                                         az_phi,
                                                         bin_size=dQ)

            y, e, Q0, Q1, Q2 = data.normalize_to_Q_sample('md',
                                                          extents,
                                                          bins,
                                                          projections)

            counts = data.extract_counts('md_data')

            ellipsoid = PeakEllipsoid(counts)

            params = ellipsoid.fit(Q0, Q1, Q2, y, e, dQ)

            if params is not None and det_id > 0:

                c, S, *fitting = ellipsoid.best_fit

                params = self.revert_ellipsoid_parameters(params, projections)

                peak.set_peak_shape(i, *params)

                bin_data = ellipsoid.bin_data

                I, sigma, bin_count = ellipsoid.integrate_norm(bin_data, c, S)

                scale = 1

                if data.laue:

                    (Q0, Q1, Q2), weights = ellipsoid.weights

                    Q0, Q1, Q2 = self.trasform_Q(Q0, Q1, Q2, projections)

                    scale = data.calculate_norm(det_id, Q0, Q1, Q2, weights)

                    I /= scale
                    sigma /= scale

                peak.set_scale_factor(i, scale)

                peak.set_peak_intensity(i, I, sigma, bin_count)

                peak.add_diagonstic_info(i, ellipsoid.info)

                plot.add_fitting(*fitting)

                plot.add_ellipsoid(c, S)

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

    def trasform_Q(self, Q0, Q1, Q2, projections):

        W = np.column_stack(projections)

        return np.einsum('ij,jk->ik', W, [Q0, Q1, Q2])

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
        bin_sizes = np.array([bin_size, bin_size, bin_size])/3

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

    def __init__(self, counts):

        self.counts = counts.copy()

    def estimate_weights(self, y, bkg_level=50):

        mask = y > 0

        weights = y/np.nansum(y)*np.nansum(self.counts)

        bkg = np.nanpercentile(weights[mask], bkg_level)

        weights[mask] -= bkg

        return weights

    def voxels(self, x0, x1, x2):

        return x0[1,0,0]-x0[0,0,0], x1[0,1,0]-x1[0,0,0], x2[0,0,1]-x2[0,0,0]

    def voxel_volume(self, x0, x1, x2):

        return np.prod(self.voxels(x0, x1, x2))

    def min_enclosing_ellipsoid(self, vals, tol=1e-6, max_iter=1000, reg=1e-8):

        n, d = vals.shape
        Q = np.vstack([vals.T, np.ones(n)])

        u = np.ones(n)/n
        err = tol+1.0

        i = 0
        while err > tol and i < max_iter:
            X = Q @ np.diag(u) @ Q.T+reg*np.eye(d+1)
            M = np.einsum('ij,ji->i', Q.T, scipy.linalg.solve(X, Q))
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum-d-1)/((d+1)*(maximum-1))
            new_u = (1-step_size)*u
            new_u[j] += step_size
            err = np.linalg.norm(new_u-u)
            u = new_u
            i += 1

        c = vals.T @ u
        S = vals.T @ np.diag(u) @ vals-np.outer(c, c)
        return c, S

    def cluster(self, x0, x1, x2, dx, weights, n_events=30):

        mask = weights > 0

        wgt = weights[mask]

        X = np.column_stack([x0[mask], x1[mask], x2[mask]])

        db = DBSCAN(eps=dx*1.2, min_samples=n_events).fit(X, sample_weight=wgt)
        labels = db.labels_

        return X, labels

    def objective(self, params, mu, c, S_inv, x, y, e):

        vol_x, center_x = params     

        dx = x-(center_x*mu+(1-center_x)*c).reshape(3,1)

        ellip = np.einsum('ij,jk,ik->k', S_inv/np.cbrt(vol_x), dx, dx)

        pk = (ellip <= 1)
        bkg = (ellip > 1) & (ellip <= np.cbrt(2)**2)

        if bkg.sum() == 0 or pk.sum() == 0:
            return 1e9

        y_bkg = y[bkg]
        w_bkg = 1/e[bkg]**2

        b = np.nansum(y_bkg*w_bkg)/np.nansum(w_bkg)
        b_err = np.sqrt(np.nansum((y_bkg-b)**2*w_bkg)/np.nansum(w_bkg))

        I = np.nansum(y[pk]-b)
        sigma = np.sqrt(np.nansum(e[pk]**2+b_err**2))

        signal_to_noise = I/sigma if I > 0 else -1e9

        return -signal_to_noise

    def maximize_signal_to_noise(self, bins, mu, c, S):

        (x0, x1, x2), y, e = bins

        S_inv = np.linalg.inv(S)

        x = np.array([x0, x1, x2])

        res = scipy.optimize.dual_annealing(self.objective,
                                            bounds=([0.5, 2], [0, 1]),
                                            args=(mu, c, S_inv, x, y, e)).x

        return res[1]*mu+(1-res[1])*c, S*np.cbrt(res[0])

    def fit(self, x0, x1, x2, y, e, dx):

        mask = (y > 0) & (e > 0)

        if mask.sum() > 21 and (np.array(mask.shape) >= 5).all():

            dx0, dx1, dx2 = self.voxels(x0, x1, x2)

            y[~mask] = np.nan
            e[~mask] = np.nan

            weights = self.estimate_weights(y)

            mask = weights > 0

            if mask.sum() < 5:
                return None

            n_events = int(np.nanmax(weights))

            if n_events < 1 or np.isinf(n_events):
                return None

            X, labels = self.cluster(x0, x1, x2, dx, weights, n_events)

            signal = labels >= 0

            if signal.sum() < 5:
                return None

            peak = y.copy()*np.nan
            peak[weights > 0] = labels+1.0

            mask = (peak > 0) & (y > 0)

            mu0 = np.average(x0[mask], weights=y[mask])
            mu1 = np.average(x1[mask], weights=y[mask])
            mu2 = np.average(x2[mask], weights=y[mask])

            mu = np.array([mu0, mu1, mu2])

            c, S = self.min_enclosing_ellipsoid(X[signal])

            if np.linalg.det(S) <= 0:
                return None

            mask = (y > 0) & (e > 0)

            bins = (x0[mask], x1[mask], x2[mask]), y[mask], e[mask]

            c, S = self.maximize_signal_to_noise(bins, mu, c, S)

            if np.linalg.det(S) <= 0:
                return None

            V, W = np.linalg.eigh(S)

            c0, c1, c2 = c

            r0, r1, r2 = np.sqrt(V)

            v0, v1, v2 = W.T

            binning = (x0, x1, x2), y, e

            fitting = binning, peak

            self.best_fit = c, S, *fitting

            self.bin_data = (x0, x1, x2), (dx0, dx1, dx2), y, e

            return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate_norm(self, bins, c, S):

        (x0, x1, x2), (dx0, dx1, dx2), y, e = bins

        c0, c1, c2 = c

        x = np.array([x0-c0, x1-c1, x2-c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum('ij,jklm,iklm->klm', S_inv, x, x)

        pk = (ellipsoid <= 1) & (e > 0) & (y > 0)
        bkg = (ellipsoid > 1) & (ellipsoid <= 1.5**2) & (e > 0) & (y > 0)

        dilate = pk | bkg

        d3x = dx0*dx1*dx2

        y_bkg = y[bkg]
        e_bkg = e[bkg]

        y_bkg = y[bkg]
        e_bkg = e[bkg]

        w_bkg = 1/e_bkg**2

        if len(w_bkg) > 2:
            b = self.weighted_median(y_bkg, w_bkg)
            b_err = self.jackknife_uncertainty(y_bkg, w_bkg)
        else:
            b = b_err = 0

        self.info = [b, b_err]

        intens = np.nansum(y[pk]-b)
        sig = np.sqrt(np.nansum(e[pk]**2+b_err**2))

        self.weights = (x0[pk], x1[pk], x2[pk]), self.counts[pk]

        bin_count = np.nansum(self.counts[pk])

        self.info += [d3x*np.sum(pk)]

        freq = y.copy()#-b
        freq[~dilate] = np.nan

        if not np.isfinite(sig):
            sig = intens

        xye = (x0, x1, x2), (dx0, dx1, dx2), freq

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        return intens, sig, bin_count

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