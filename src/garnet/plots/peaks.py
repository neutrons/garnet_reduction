import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import scipy.special

from garnet.plots.base import BasePlot

class RadiusPlot(BasePlot):

    def __init__(self, r, y, y_fit):

        super(RadiusPlot, self).__init__()

        plt.close('all')

        self.fig, self.ax = plt.subplots(1,
                                         1,
                                         figsize=(6.4, 4.8),
                                         layout='constrained')

        self.add_radius_fit(r, y, y_fit)

    def add_radius_fit(self, r, y, y_fit):

        ax = self.ax

        ax.plot(r, y, 'o', color='C0')
        ax.plot(r, y_fit, '.', color='C1')
        ax.minorticks_on()
        ax.set_xlabel(r'$r$ [$\AA^{-1}$]')

    def add_sphere(self, r_cut, A, sigma):

        self.ax.axvline(x=r_cut, color='k', linestyle='--')

        xlim = list(self.ax.get_xlim())
        xlim[0] = 0

        x = np.linspace(*xlim, 256)

        z = x/sigma

        y = A*(scipy.special.erf(z/np.sqrt(2))-\
               np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2))

        self.ax.plot(x, y, '-', color='C1')
        self.ax.set_ylabel(r'$I/\sigma$')

class PeakPlot(BasePlot):

    def __init__(self):

        super(PeakPlot, self).__init__()

        plt.close('all')

        self.fig = plt.figure(figsize=(6.4, 12.8), layout='constrained')

        sp = GridSpec(3, 1, figure=self.fig, height_ratios=[1,1,1])

        self.gs = []

        gs = GridSpecFromSubplotSpec(2,
                                     2,
                                     subplot_spec=sp[0],
                                     width_ratios=[1,0.5])

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(2,
                                     3,
                                     height_ratios=[1,1],
                                     width_ratios=[1,1,1],
                                     subplot_spec=sp[1])

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(1,
                                     3,
                                     height_ratios=[1],
                                     width_ratios=[1,1,1],
                                     subplot_spec=sp[2])

        self.gs.append(gs)

        self.__init_ellipsoid()
        self.__init_projection()
        self.__init_profile()
        self.__init_norm()

    def __init_ellipsoid(self):

        self.ellip = []
        self.ellip_im = []
        self.ellip_el = []
        self.ellip_pt = []

        x = np.arange(5)
        y = np.arange(6)
        z = y+y.size*x[:,np.newaxis]

        gs = self.gs[1]

        ax = self.fig.add_subplot(gs[0,0])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1,0])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$|Q|$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0,1])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1,1])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$|Q|$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0,2])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)
        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1,2])

        self.ellip.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.ellip_el.append(el)

        line = self._draw_intersecting_line(ax, 2.5, 3)
        self.ellip_pt.append(line)

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)
        self.cb_el = self.fig.colorbar(im, ax=[self.ellip[-2], self.ellip[-1]])
        self.cb_el.ax.minorticks_on()
        self.cb_el.formatter.set_powerlimits((0, 0))
        self.cb_el.formatter.set_useMathText(True)

    def __init_projection(self):

        gs = self.gs[0]

        self.proj = []
        self.proj_im = []
        self.proj_el = []

        x = np.arange(5)
        y = np.arange(6)
        z = y+y.size*x[:,np.newaxis]

        ax = self.fig.add_subplot(gs[0,1])

        self.proj.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')
        ax.xaxis.set_ticklabels([])

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.proj_el.append(el)

        ax = self.fig.add_subplot(gs[1,1])

        self.proj.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0)
        self.proj_el.append(el)

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)
        self.cb_proj = self.fig.colorbar(im, ax=self.proj)
        self.cb_proj.ax.minorticks_on()
        self.cb_proj.formatter.set_powerlimits((0, 0))
        self.cb_proj.formatter.set_useMathText(True)

    def __init_profile(self):

        gs = self.gs[0]

        ax = self.fig.add_subplot(gs[:,0])

        x = np.linspace(0, 10)
        y = np.ones_like(x)
        e = np.zeros_like(x)

        y_fit = np.ones_like(x)

        self.eb = ax.errorbar(x, y, e, fmt='o')
        self.prof_dat, = ax.step(x, y, where='mid', color='C0')
        self.prof_fit, = ax.step(x, y_fit, where='mid', color='C1')
        ax.minorticks_on()
        ax.set_xlabel(r'$|Q|$ [$\AA^{-1}$]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True

        self.lower = ax.axvline(x=3, color='k', linestyle='--')
        self.upper = ax.axvline(x=7, color='k', linestyle='--')

        self.prof = ax

    def __init_norm(self):

        self.norm = []
        self.norm_im = []
        self.norm_el = []

        x = np.arange(5)
        y = np.arange(6)
        z = y+y.size*x[:,np.newaxis]

        gs = self.gs[2]

        ax = self.fig.add_subplot(gs[0,0])

        self.norm.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, 'k')
        self.norm_el.append(el)

        ax.set_xlabel(r'$|Q|$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[0,1])

        self.norm.append(ax)

        im = ax.imshow(z.T, 
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, 'k')
        self.norm_el.append(el)

        ax.set_xlabel(r'$|Q|$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[0,2])

        self.norm.append(ax)

        im = ax.imshow(z.T,
                       extent=(0, 5, 0, 6),
                       origin='lower',
                       interpolation='nearest')

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, 'k')
        self.norm_el.append(el)

        ax.set_xlabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)

        self.cb_norm = self.fig.colorbar(im, 
                                         ax=self.norm,
                                         orientation='vertical')
        self.cb_norm.ax.minorticks_on()
        self.cb_norm.formatter.set_powerlimits((0, 0))
        self.cb_norm.formatter.set_useMathText(True)

    def add_data_norm_fit(self, xye, params):

        axes, bins, y = xye

        x0, x1, x2 = axes

        y[np.isinf(y)] = np.nan

        y0 = np.nansum(y, axis=0)
        y1 = np.nansum(y, axis=1)
        y2 = np.nansum(y, axis=2)

        mask = np.isfinite(y)

        if mask.sum() == 0:
            mask = np.ones_like(mask, dtype=bool)
            
        s0 = 0.5*(x0[1,0,0]-x0[0,0,0])
        s1 = 0.5*(x1[0,1,0]-x1[0,0,0])
        s2 = 0.5*(x2[0,0,1]-x2[0,0,0])

        x0min, x0max = x0[mask].min()-s0, x0[mask].max()+s0
        x1min, x1max = x1[mask].min()-s1, x1[mask].max()+s1
        x2min, x2max = x2[mask].min()-s2, x2[mask].max()+s2

        x0_min, x0_max = x0[0,0,0]-s0, x0[-1,0,0]+s0
        x1_min, x1_max = x1[0,0,0]-s1, x1[0,-1,0]+s1
        x2_min, x2_max = x2[0,0,0]-s2, x2[0,0,-1]+s2

        mask_0 = np.isfinite(y0) & (y0 > 0)
        mask_1 = np.isfinite(y1) & (y1 > 0)
        mask_2 = np.isfinite(y2) & (y2 > 0)

        y0[~mask_0] = np.nan
        y1[~mask_1] = np.nan
        y2[~mask_2] = np.nan

        vmin, vmax = np.nanmin(y2), np.nanmax(y2)

        self.norm_im[0].set_data(y2.T)
        self.norm_im[0].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.norm_im[0].set_clim(vmin, vmax)
        self.norm[0].set_xlim([x0min, x0max])
        self.norm[0].set_ylim([x1min, x1max])

        # ---

        vmin, vmax = np.nanmin(y1), np.nanmax(y1)

        self.norm_im[1].set_data(y1.T)
        self.norm_im[1].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.norm_im[1].set_clim(vmin, vmax)
        self.norm[1].set_xlim([x0min, x0max])
        self.norm[1].set_ylim([x2min, x2max])

        # ---

        vmin, vmax = np.nanmin(y0), np.nanmax(y0)

        self.norm_im[2].set_data(y0.T)
        self.norm_im[2].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.norm_im[2].set_clim(vmin, vmax)
        self.norm[2].set_xlim([x1min, x1max])
        self.norm[2].set_ylim([x2min, x2max])

        self.cb_norm.update_normal(self.norm_im[2])
        self.cb_norm.ax.minorticks_on()
        self.cb_norm.formatter.set_powerlimits((0, 0))
        self.cb_norm.formatter.set_useMathText(True)

        I = r'$I={}$'
        I_sig = '$I/\sigma={:.1f}$'
        B = r'$B={}$'

        self.norm[0].set_title(I.format(self._sci_notation(params[0])))
        self.norm[1].set_title(I_sig.format(params[0]/params[1]))
        self.norm[2].set_title(B.format(self._sci_notation(params[2])))

    def add_fitting(self, fitting):

        xye_1d, y_1d_fit, xye_2d, y_2d_fit, xye_3d, y_3d_fit = fitting

        self.add_profile_fit(xye_1d, y_1d_fit)
        self.add_projection_fit(xye_2d, y_2d_fit)
        self.add_ellipsoid_fit(xye_3d, y_3d_fit)

    def _color_limits(self, y1, y2):
        """
        Calculate color limits common for two arrays.

        Parameters
        ----------
        y1 : array-like
            Data array.
        y2 : array-like
            Data array.

        Returns
        -------
        vmin, vmax : float
            Color limits

        """

        vmin = np.nanmax([np.nanmin(y1), np.nanmin(y2)])
        vmax = np.nanmin([np.nanmax(y1), np.nanmax(y2)])

        if vmin >= vmax:
            vmin, vmax = 0, 1

        return vmin, vmax

    def add_ellipsoid_fit(self, xye, y_fit):
        """
        Three-dimensional ellipsoids.

        Parameters
        ----------
        x, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        axes, bins, y, e = xye

        x0, x1, x2 = axes

        y[np.isinf(y)] = np.nan
        y_fit[np.isinf(y_fit)] = np.nan

        y0 = np.nansum(y, axis=0)
        y1 = np.nansum(y, axis=1)
        y2 = np.nansum(y, axis=2)

        y0_fit = np.nansum(y_fit, axis=0)
        y1_fit = np.nansum(y_fit, axis=1)
        y2_fit = np.nansum(y_fit, axis=2)

        mask_0 = np.isfinite(y0) & (y0 > 0)
        mask_1 = np.isfinite(y1) & (y1 > 0)
        mask_2 = np.isfinite(y2) & (y2 > 0)

        y0[~mask_0] = np.nan
        y1[~mask_1] = np.nan
        y2[~mask_2] = np.nan

        y0_fit[~mask_0] = np.nan
        y1_fit[~mask_1] = np.nan
        y2_fit[~mask_2] = np.nan

        d0 = 0.5*(x0[1,0,0]-x0[0,0,0])
        d1 = 0.5*(x1[0,1,0]-x1[0,0,0])
        d2 = 0.5*(x2[0,0,1]-x2[0,0,0])

        x0_min, x0_max = x0[0,0,0]-d0, x0[-1,0,0]+d0
        x1_min, x1_max = x1[0,0,0]-d1, x1[0,-1,0]+d1
        x2_min, x2_max = x2[0,0,0]-d2, x2[0,0,-1]+d2

        vmin, vmax = self._color_limits(y2, y2_fit)

        self.ellip_im[0].set_data(y2.T)
        self.ellip_im[0].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.ellip_im[0].set_clim(vmin, vmax)

        self.ellip_im[1].set_data(y2_fit.T)
        self.ellip_im[1].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.ellip_im[1].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y1, y1_fit)

        self.ellip_im[2].set_data(y1.T)
        self.ellip_im[2].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.ellip_im[2].set_clim(vmin, vmax)

        self.ellip_im[3].set_data(y1_fit.T)
        self.ellip_im[3].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.ellip_im[3].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y0, y0_fit)

        self.ellip_im[4].set_data(y0.T)
        self.ellip_im[4].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.ellip_im[4].set_clim(vmin, vmax)

        self.ellip_im[5].set_data(y0_fit.T)
        self.ellip_im[5].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.ellip_im[5].set_clim(vmin, vmax)

        self.cb_el.update_normal(self.ellip_im[5])
        self.cb_el.ax.minorticks_on()
        self.cb_el.formatter.set_powerlimits((0, 0))
        self.cb_el.formatter.set_useMathText(True)

    def add_projection_fit(self, xye, y_fit):
        """
        Two-dimensional ellipses.

        Parameters
        ----------
        x, dx, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        (xu, xv), (dxu, dxv), y, e = xye

        mask = np.isfinite(y) & (y > 0) & np.isfinite(e) & (e > 0)

        y[~mask] = np.nan
        y_fit[~mask] = np.nan

        vmin, vmax = self._color_limits(y, y_fit)

        du = 0.5*(xu[1,0]-xu[0,0])
        dv = 0.5*(xv[0,1]-xv[0,0])

        xu_min, xu_max = xu[0,0]-du, xu[-1,0]+du
        xv_min, xv_max = xv[0,0]-dv, xv[0,-1]+dv

        self.proj_im[0].set_data(y.T)
        self.proj_im[0].set_extent((xu_min, xu_max, xv_min, xv_max))
        self.proj_im[0].set_clim(vmin, vmax)

        self.proj_im[1].set_data(y_fit.T)
        self.proj_im[1].set_extent((xu_min, xu_max, xv_min, xv_max))
        self.proj_im[1].set_clim(vmin, vmax)

        self.cb_proj.update_normal(self.proj_im[1])
        self.cb_proj.ax.minorticks_on()
        self.cb_proj.formatter.set_powerlimits((0, 0))
        self.cb_proj.formatter.set_useMathText(True)

    def add_profile_fit(self, xye, y_fit):
        """
        One-dimensional Gaussian.

        Parameters
        ----------
        x, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        x, dx, y, e = xye

        line, caps, bars = self.eb

        barsy, = bars

        line.set_data(x, y)

        barsy.set_segments([np.array([[x, yt],
                                      [x, yb]]) for x, yt, yb in zip(x,
                                                                     y-e,
                                                                     y+e)])

        self.prof_dat.set_data(x, y)
        self.prof_fit.set_data(x, y_fit)

    def add_ellipsoid(self, c, S, vals):
        """
        Draw ellipsoid envelopes.

        Parameters
        ----------
        c : 1d-array
            3 component center.
        S : 2d-array
            3x3 covariance matrix.
        vals : list
            Fitted parameters.

        """

        r = np.sqrt(np.diag(S))

        rho = [S[1,2]/r[1]/r[2],
               S[0,2]/r[0]/r[2],
               S[0,1]/r[0]/r[1]]

        mu, mu_u, mu_v, rad, ru, rv, corr, *_ = vals

        self.lower.set_xdata([mu-rad, mu-rad])
        self.upper.set_xdata([mu+rad, mu+rad])

        self.prof.relim()
        self.prof.autoscale_view()

        for el, ax in zip(self.proj_el,self.proj):
             self._update_ellipse(el, ax, mu_u, mu_v, ru, rv, corr)

        for el, pt, ax in zip(self.ellip_el[0:2],
                              self.ellip_pt[0:2],
                              self.ellip[0:2]):
            self._update_ellipse(el, ax, c[0], c[1], r[0], r[1], rho[2])
            self._update_intersecting_line(pt, ax, c[0], c[1])

        for el, pt, ax in zip(self.ellip_el[2:4],
                              self.ellip_pt[2:4],
                              self.ellip[2:4]):
            self._update_ellipse(el, ax, c[0], c[2], r[0], r[2], rho[1])
            self._update_intersecting_line(pt, ax, c[0], c[2])

        for el, pt, ax in zip(self.ellip_el[4:6],
                              self.ellip_pt[4:6],
                              self.ellip[4:6]):
            self._update_ellipse(el, ax, c[1], c[2], r[1], r[2], rho[0])
            self._update_intersecting_line(pt, ax, c[1], c[2])

        for el, ax in zip(self.norm_el[0:1], self.norm[0:1]):
            self._update_ellipse(el, ax, c[0], c[1], r[0], r[1], rho[2])

        for el, ax in zip(self.norm_el[1:2], self.norm[1:2]):
            self._update_ellipse(el, ax, c[0], c[2], r[0], r[2], rho[1])

        for el, ax in zip(self.norm_el[2:3], self.norm[2:3]):
            self._update_ellipse(el, ax, c[1], c[2], r[1], r[2], rho[0])

    def _update_ellipse(self, ellipse, ax, cx, cy, rx, ry, rho):

        ellipse.set_center((0, 0))

        ellipse.width = 2*np.sqrt(1+rho)
        ellipse.height = 2*np.sqrt(1-rho)

        trans = Affine2D()
        trans.rotate_deg(45).scale(rx, ry).translate(cx, cy)

        ellipse.set_transform(trans+ax.transData)

    def _draw_ellipse(self, ax, cx, cy, rx, ry, rho, color='w'):
        """
        Draw ellipse with center, size, and orientation.

        Parameters
        ----------
        ax : axis
            Plot axis.
        cx, cy : float
            Center.
        rx, xy : float
            Radii.
        rho : float
            Correlation.

        """

        peak = Ellipse((0, 0),
                       width=2*np.sqrt(1+rho),
                       height=2*np.sqrt(1-rho),
                       linestyle='-',
                       edgecolor=color,
                       facecolor='none',
                       rasterized=False,
                       zorder=100)

        self._update_ellipse(peak, ax, cx, cy, rx, ry, rho)

        ax.add_patch(peak)

        return peak

    def _update_intersecting_line(self, line, ax, x0, y0):

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if x0 != 0:
            slope = y0/x0
        else:
            slope = np.inf

        y_at_x_min = slope*(x_min-x0)+y0 if slope != np.inf else y_min
        y_at_x_max = slope*(x_max-x0)+y0 if slope != np.inf else y_max
        x_at_y_min = (y_min-y0)/slope+x0 if slope != 0 else x_min
        x_at_y_max = (y_max-y0)/slope+x0 if slope != 0 else x_max

        points = []
        if y_min <= y_at_x_min <= y_max:
            points.append((x_min, y_at_x_min))
        if y_min <= y_at_x_max <= y_max:
            points.append((x_max, y_at_x_max))
        if x_min <= x_at_y_min <= x_max:
            points.append((x_at_y_min, y_min))
        if x_min <= x_at_y_max <= x_max:
            points.append((x_at_y_max, y_max))

        if len(points) > 2:
            points = sorted(points, key=lambda p: (p[0], p[1]))[:2]
        elif len(points) == 0:
            points = (x_min, y_min), (x_max, y_max)

        (x1, y1), (x2, y2) = points

        line.set_data([x1, x2], [y1, y2])

    def _draw_intersecting_line(self, ax, x0, y0):
        """
        Draw line toward origin.

        Parameters
        ----------
        ax : axis
            Plot axis.
        x0, y0 : float
            Center.

        """

        line, = ax.plot([], [], color='k', linestyle='--')

        self._update_intersecting_line(line, ax, x0, y0)

        return line

    def _sci_notation(self, x):
        """
        Represent float in scientific notation using LaTeX.

        Parameters
        ----------
        x : float
            Value to convert.

        Returns
        -------
        s : str
            String representation in LaTeX.

        """

        if np.isfinite(x):
            val = np.floor(np.log10(abs(x)))
            if np.isfinite(val):
                exp = int(val)
                return '{:.2f}\\times 10^{{{}}}'.format(x / 10**exp, exp)
            else:
                return '\\infty'
        else:
            return '\\infty'

    def add_peak_intensity(self, intens, sig_noise):
        """
        Add integrated intensities.

        Parameters
        ----------
        intens : list
            Intensity.
        sig_noise : list
            Signal-to-noise ratio.

        """

        I = r'$I={}$'
        I_sig = '$I/\sigma={:.1f}$'

        title = I.format(self._sci_notation(intens[0]))+' '+\
                I_sig.format(sig_noise[0])

        self.prof.set_title(title)
        self.proj[0].set_title(I.format(self._sci_notation(intens[1])))
        self.proj[1].set_title(I_sig.format(sig_noise[1]))

        self.ellip[0].set_title(I.format(self._sci_notation(intens[2])))
        self.ellip[1].set_title(I_sig.format(sig_noise[2]))

    def add_peak_info(self, wavelength, angles, gon):
        """
        Add peak information.

        Parameters
        ----------
        wavelength : float
            Wavelength.
        angles : list
            Scattering and azimuthal angles.
        gon : list
            Goniometer Euler angles.

        """

        ellip = self.ellip

        ellip[2].set_title(r'$\lambda={:.4f}$ [$\AA$]'.format(wavelength))
        ellip[3].set_title(r'$({:.1f},{:.1f},{:.1f})^\circ$'.format(*gon))
        ellip[4].set_title(r'$2\theta={:.2f}^\circ$'.format(angles[0]))
        ellip[5].set_title(r'$\phi={:.2f}^\circ$'.format(angles[1]))