from mantid.simpleapi import (FindPeaksMD,
                              PredictPeaks,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              PeakIntensityVsRadius,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              DeleteWorkspace,
                              DeleteTableRows,
                              ExtractSingleSpectrum,
                              CombinePeaksWorkspaces,
                              CreatePeaksWorkspace,
                              ConvertPeaksWorkspace,
                              CopySample,
                              CloneWorkspace,
                              SaveNexus,
                              LoadNexus,
                              AddPeakHKL,
                              HasUB,
                              SetUB,
                              mtd)

from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid, NoShape

from mantid import config

config['Q.convention'] = 'Crystallography'

import numpy as np

centering_reflection = {'P': 'Primitive',
                        'I': 'Body centred',
                        'F': 'All-face centred',
                        'R': 'Primitive', # rhomb axes
                        'R(obv)': 'Rhombohderally centred, obverse', # hex axes
                        'R(rev)': 'Rhombohderally centred, reverse', # hex axes
                        'A': 'A-face centred',
                        'B': 'B-face centred',
                        'C': 'C-face centred'}

class PeaksModel:

    def __init__(self):

        self.edge_pixels = 0

    def find_peaks(self, md,
                         peaks,
                         max_d,
                         density=1000,
                         max_peaks=50):
        """
        Harvest strong peak locations from Q-sample into a peaks table.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        max_d : float
            Maxium d-spacing enforcing lower limit of peak spacing.
        density : int, optional
            Threshold density. The default is 1000.
        max_peaks : int, optional
            Maximum number of peaks to find. The default is 50.

        """

        FindPeaksMD(InputWorkspace=md,
                    PeakDistanceThreshold=2*np.pi/max_d,
                    MaxPeaks=max_peaks,
                    PeakFindingStrategy='VolumeNormalization',
                    DensityThresholdFactor=density,
                    EdgePixels=self.edge_pixels,
                    OutputWorkspace=peaks)

    def centroid_peaks(self, md, peaks, peak_radius):
        """
        Re-center peak locations using centroid within given radius

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.

        """

        CentroidPeaksMD(InputWorkspace=md,
                        PeakRadius=peak_radius,
                        PeaksWorkspace=peaks,
                        OutputWorkspace=peaks)

    def integrate_peaks(self, md,
                              peaks,
                              peak_radius,
                              background_inner_fact=1,
                              background_outer_fact=1.5,
                              method='sphere',
                              centroid=True):
        """
        Integrate peaks using spherical or ellipsoidal regions.
        Ellipsoid integration adapts itself to the peak distribution.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        method : str, optional
            Integration method. The default is 'sphere'.
        centroid : str, optional
            Shift peak position to centroid. The default is True.
            
        """

        background_inner_radius = peak_radius*background_inner_fact
        background_outer_radius = peak_radius*background_outer_fact

        if method == 'sphere' and centroid:
            self.centroid_peaks(md, peaks, peak_radius)

        IntegratePeaksMD(InputWorkspace=md,
                         PeaksWorkspace=peaks,
                         PeakRadius=peak_radius,
                         BackgroundInnerRadius=background_inner_radius,
                         BackgroundOuterRadius=background_outer_radius,
                         Ellipsoid=True if method == 'ellipsoid' else False,
                         FixQAxis=False,
                         FixMajorAxisLength=False,
                         UseCentroid=True,
                         MaxIterations=3,
                         ReplaceIntensity=True,
                         IntegrateIfOnEdge=False,
                         AdaptiveQBackground=False,
                         MaskEdgeTubes=False,
                         OutputWorkspace=peaks)

    def intensity_vs_radius(self, md,
                                  peaks,
                                  peak_radius,
                                  background_inner_fact=1,
                                  background_outer_fact=1.5,
                                  steps=51,
                                  fix=False):
        """
        Integrate peak intensity with radius varying from zero to cut off.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integrat region radius cut off.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        steps : int, optional
            Number of integration steps. The default is 101.
        fix : bool, optional
            Fix the background shell size

        Returns
        -------

        radius : list
            Peak radius.
        sig_noise : list
            Peak signal/noise ratio at lowest threshold.
        intens : list
            Peak intensity.

        """

        background_inner_rad = background_inner_fact*peak_radius if fix else 0
        background_outer_rad = background_outer_fact*peak_radius if fix else 0

        background_inner_fact = 0 if fix else background_inner_fact
        background_outer_fact = 0 if fix else background_outer_fact

        PeakIntensityVsRadius(InputWorkspace=md,
                              PeaksWorkspace=peaks,
                              RadiusStart=0.0,
                              RadiusEnd=peak_radius,
                              NumSteps=steps,
                              BackgroundInnerFactor=background_inner_fact,
                              BackgroundOuterFactor=background_outer_fact,
                              BackgroundInnerRadius=background_inner_rad,
                              BackgroundOuterRadius=background_outer_rad,
                              OutputWorkspace=peaks+'_intens_vs_rad',
                              OutputWorkspace2=peaks+'_sig/noise_vs_rad')

        ExtractSingleSpectrum(InputWorkspace=peaks+'_sig/noise_vs_rad',
                              OutputWorkspace=peaks+'_sig/noise_vs_rad/lowest',
                              WorkspaceIndex=0)

        peak_radius = mtd[peaks+'_sig/noise_vs_rad/lowest'].extractX().ravel()
        sig_noise = mtd[peaks+'_sig/noise_vs_rad/lowest'].extractY().ravel()
        intens = mtd[peaks+'_intens_vs_rad'].extractY()

        return peak_radius, sig_noise, intens

    def get_max_d_spacing(self, ws):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=ws):

            if hasattr(mtd[ws], 'sample'):
                ol = mtd[ws].sample().getOrientedLattice()
            else:
                for i in range(mtd[ws].getNumExperimentInfo()):
                    sample = mtd[ws].getExperimentInfo(i).sample()
                    if sample.hasOrientedLattice():
                        ol = sample.getOrientedLattice()
                        SetUB(Workspace=ws, UB=ol.getUB())
                ol = mtd[ws].getExperimentInfo(i).sample().getOrientedLattice()

            return max([ol.a(), ol.b(), ol.c()])

    def predict_peaks(self, ws, peaks, centering, d_min, lamda_min, lamda_max):
        """
        Predict peak Q-sample locations with UB and lattice centering.

        +--------+-----------------------+
        | Symbol | Reflection condition  |
        +========+=======================+
        | P      | None                  |
        +--------+-----------------------+
        | I      | :math:`h+k+l=2n`      |
        +--------+-----------------------+
        | F      | :math:`h,k,l` unmixed |
        +--------+-----------------------+
        | R      | None                  |
        +--------+-----------------------+
        | R(obv) | :math:`-h+k+l=3n`     |
        +--------+-----------------------+
        | R(rev) | :math:`h-k+l=3n`      |
        +--------+-----------------------+
        | A      | :math:`k+l=2n`        |
        +--------+-----------------------+
        | B      | :math:`l+h=2n`        |
        +--------+-----------------------+
        | C      | :math:`h+k=2n`        |
        +--------+-----------------------+

        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.

        """

        d_max = self.get_max_d_spacing(ws)

        PredictPeaks(InputWorkspace=ws,
                     WavelengthMin=lamda_min,
                     WavelengthMax=lamda_max,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max*1.2,
                     ReflectionCondition=centering_reflection[centering],
                     RoundHKL=True,
                     EdgePixels=self.edge_pixels,
                     OutputWorkspace=peaks)

    def predict_modulated_peaks(self, peaks,
                                      d_min,
                                      lamda_min,
                                      lamda_max,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):
        """
        Predict the modulated peak positions based on main peaks.

        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of main peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        d_max = self.get_max_d_spacing(peaks)

        sat_peaks = peaks+'_sat'

        PredictSatellitePeaks(Peaks=peaks,
                              SatellitePeaks=sat_peaks,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              IncludeIntegerHKL=False,
                              IncludeAllPeaksInRange=True,
                              WavelengthMin=lamda_min,
                              WavelengthMax=lamda_max,
                              MinDSpacing=d_min,
                              MaxDSpacing=d_max*10)

        CombinePeaksWorkspaces(LHSWorkspace=peaks,
                               RHSWorkspace=sat_peaks,
                               OutputWorkspace=peaks)

        DeleteWorkspace(Workspace=sat_peaks)

    def predict_satellite_peaks(self, peaks_ws,
                                      data_ws, 
                                      lamda_min,
                                      lamda_max,
                                      d_min,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):
        """
        Locate satellite peaks from goniometer angles.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data with goniometer(s).
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min : float
            Minimum wavelength.
        lamda_max : float
            Maximum wavelength.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        Rs = self.get_all_goniometer_matrices(data_ws)

        for R in Rs:

            self.set_goniometer(peaks_ws, R)

            self.predict_modulated_peaks(peaks_ws,
                                         d_min,
                                         lamda_min,
                                         lamda_max,
                                         mod_vec_1,
                                         mod_vec_2,
                                         mod_vec_3,
                                         max_order,
                                         cross_terms)

            self.remove_duplicate_peaks(peaks_ws)

    def sort_peaks_by_hkl(self, peaks):
        """
        Sort peaks table by descending hkl values.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        columns = ['l', 'k', 'h']

        for col in columns:

            SortPeaksWorkspace(InputWorkspace=peaks,
                               ColumnNameToSortBy=col,
                               SortAscending=False,
                               OutputWorkspace=peaks)

    def sort_peaks_by_d(self, peaks):
        """
        Sort peaks table by descending d-spacing.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(InputWorkspace=peaks,
                           ColumnNameToSortBy='DSpacing',
                           SortAscending=False,
                           OutputWorkspace=peaks)

    def remove_duplicate_peaks(self, peaks):
        """
        Omit duplicate peaks from different based on indexing.
        Table will be sorted.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        self.sort_peaks_by_hkl(peaks)

        for no in range(mtd[peaks].getNumberPeaks()-1,0,-1):

            if (mtd[peaks].getPeak(no).getHKL()-\
                mtd[peaks].getPeak(no-1).getHKL()).norm2() == 0:

                DeleteTableRows(TableWorkspace=peaks, Rows=no)

    def get_all_goniometer_matrices(self, ws):
        """
        Extract all goniometer matrices.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.

        Returns
        -------
        Rs: list
            Goniometer matrices.

        """

        Rs = []

        for ei in range(mtd[ws].getNumExperimentInfo()):

            run = mtd[ws].getExperimentInfo(ei).run()

            n_gon = run.getNumGoniometers()

            Rs += [run.getGoniometer(i).getR() for i in range(n_gon)]

        return np.array(Rs)

    def renumber_runs_by_index(self, ws, peaks):
        """
        Re-label the runs by index based on goniometer setting.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.
        peaks : str
            Name of peaks table.

        """

        Rs = self.get_all_goniometer_matrices(ws)

        for no in range(mtd[peaks].getNumberPeaks()):

            peak = mtd[peaks].getPeak(no)

            R = peak.getGoniometerMatrix()

            ind = np.isclose(Rs, R).all(axis=(1,2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)

            peak.setRunNumber(i+1)

    def load_peaks(self, filename, peaks):
        """
        Load peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        LoadNexus(Filename=filename,
                  OutputWorkspace=peaks)

    def save_peaks(self, filename, peaks):
        """
        Save peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        SaveNexus(Filename=filename,
                  InputWorkspace=peaks)

    def convert_peaks(self, peaks):
        """
        Remove instrument from peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        ConvertPeaksWorkspace(PeakWorkspace=peaks,
                              OutputWorkspace=peaks)

        for i in range(mtd[peaks].getNumberPeaks()):
            mtd[peaks].getPeak(i).setPeakShape(NoShape())

    def combine_peaks(self, peaks, merge):
        """
        Merge two peaks workspaces into one.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.
        merge : str
            Name of peaks table to be accumulated.

        """

        if not mtd.doesExist(merge):

            CloneWorkspace(InputWorkspace=peaks,
                           OutputWorkspace=merge)

        else:

            CombinePeaksWorkspaces(LHSWorkspace=merge,
                                   RHSWorkspace=peaks,
                                   OutputWorkspace=merge)

            merge_run = mtd[merge].run()
            peaks_run = mtd[peaks].run()

            keys = ['run', 'h', 'k', 'l', 'm', 'n', 'p', 'bkg', 'bkg_err', 
                    'data', 'data_err', 'norm', 'norm_err', 'vol',
                    'A1d', 'A2d', 'A3d', 'sn1d', 'sn2d', 'sn3d']

            for key in keys:
                log = 'peaks_{}'.format(key)
                peaks_list = np.array(peaks_run.getLogData(log).value).tolist()
                merge_list = np.array(merge_run.getLogData(log).value).tolist()
                merge_run[log] = merge_list+peaks_list

    def delete_peaks(self, peaks):
        """
        Remove peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.

        """

        if mtd.doesExist(peaks):

            DeleteWorkspace(Workspace=peaks)

    def remove_weak_peaks(self, peaks, sig_noise=3):
        """
        Filter out weak peaks based on signal-to-noise ratio.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        sig_noise : float, optional
            Minimum signal-to-noise ratio. The default is 3.

        """

        FilterPeaks(InputWorkspace=peaks,
                    OutputWorkspace=peaks,
                    FilterVariable='Signal/Noise',
                    FilterValue=sig_noise,
                    Operator='>',
                    Criterion='!=',
                    BankName='None')

    def create_peaks(self, ws, peaks):
        """
        Create a new peaks table.

        ws : str
            Name of workspace.
        peaks : str
            Name of peaks table.

        """

        CreatePeaksWorkspace(InstrumentWorkspace=ws,
                             NumberOfPeaks=0,
                             OutputWorkspace=peaks)

        CopySample(InputWorkspace=ws,
                   OutputWorkspace=peaks,
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)

    def add_peak(self, peaks, hkl):
        """
        Add a peak to an existing table.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        hkl : list
            Miller index.

        """

        AddPeakHKL(Workspace=peaks, HKL=hkl)

    def set_goniometer(self, peaks, R):
        """
        Update the goniometer on the run.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        R : 2d-array
            Goniometer matrix.

        """

        mtd[peaks].run().getGoniometer().setR(R)

    def get_peaks_name(self, peaks):
        """
        Name of peaks.

        Returns
        -------
        name : str
            Readable name of peaks.

        """

        peak = mtd[peaks].getPeak(0)

        run = peak.getRunNumber()

        name = 'peaks_run#{}'
        return name.format(run)

class PeakModel:

    def __init__(self, peaks):

        self.peaks = peaks

    def get_number_peaks(self):
        """
        Total number of peaks in the table.

        Returns
        -------
        n : int
            Number of peaks

        """

        return mtd[self.peaks].getNumberPeaks()

    def set_peak_intensity(self, no, intens, sig):
        """
        Update the peak intensity.

        Parameters
        ----------
        no : int
            Peak index number.
        intens : float
            Intensity.
        sig : float
            Uncertainty.

        """

        mtd[self.peaks].getPeak(no).setIntensity(intens)
        mtd[self.peaks].getPeak(no).setSigmaIntensity(sig)

    def get_wavelength(self, no):
        """
        Wavelength of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        lamda : float
            Wavelength in angstroms.

        """

        peak = mtd[self.peaks].getPeak(no)

        return peak.getWavelength()

    def get_sample_Q(self, no):
        """
        Scattering vector in Q sample coordinates.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        Q : list
            Scattering Q-vector.

        """

        peak = mtd[self.peaks].getPeak(no)

        return peak.getQSampleFrame()

    def get_angles(self, no):
        """
        Scattering and azimuthal angle of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        two_theta : float
            Scattering (polar) angle in degrees.
        az_phi : float
            Azimuthal angle in degrees.

        """

        peak = mtd[self.peaks].getPeak(no)

        two_theta = np.rad2deg(peak.getScattering())
        az_phi = np.rad2deg(peak.getAzimuthal())

        return two_theta, az_phi
    
    def get_goniometer_matrix(self, no):
        """
        Goniometer matrix of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        R : 2d-array
            Rotation matrix.

        """
        
        peak = mtd[self.peaks].getPeak(no)

        return peak.getGoniometerMatrix()

    def get_goniometer_angles(self, no):
        """
        Goniometer Euler angles of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        angles : list
            Euler angles (YZY convention) in degrees.

        """

        peak = mtd[self.peaks].getPeak(no)
        gon = mtd[self.peaks].run().getGoniometer()

        R = peak.getGoniometerMatrix()
        gon.setR(R)

        return list(gon.getEulerAngles())

    def add_diagonstic_info(self, no, values):
        """
        Log diagnostic info

        Parameters
        ----------
        no : int
            Peak index number.
        values : list
           Diagnostics.

        """

        peak = mtd[self.peaks].getPeak(no)

        run = int(peak.getRunNumber())
        h, k, l = [int(val) for val in peak.getIntHKL()]
        m, n, p = [int(val) for val in peak.getIntMNP()]

        run_info = mtd[self.peaks].run()
        run_info_keys = run_info.keys()

        keys = ['run', 'h', 'k', 'l', 'm', 'n', 'p', 'bkg', 'bkg_err', 
                'data', 'data_err', 'norm', 'norm_err', 'vol',
                'A1d', 'A2d', 'A3d', 'sn1d', 'sn2d', 'sn3d']

        vals = [run, h, k, l, m, n, p]+values

        for key, val in zip(keys, vals):
            log = 'peaks_{}'.format(key)
            if log not in run_info_keys:
                run_info[log] = [val]
            else:
                items = np.array(run_info.getLogData(log).value).tolist()
                items.append(val)
                run_info[log] = np.array(items).tolist()

    def get_peak_name(self, no):
        """
        Name of peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        name : str
            Readable name of peak.

        """

        peak = mtd[self.peaks].getPeak(no)

        run = peak.getRunNumber()
        hkl = [int(val) for val in peak.getIntHKL()]
        mnp = [int(val) for val in peak.getIntMNP()]
        lamda = peak.getWavelength()

        if HasUB(Workspace=self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()
            mod_1 = ol.getModVec(0)
            mod_2 = ol.getModVec(1)
            mod_3 = ol.getModVec(2)
            h, k, l, m, n, p = *hkl, *mnp
            dh, dk, dl = m*np.array(mod_1)+n*np.array(mod_2)+p*np.array(mod_3)
            d = ol.d(V3D(h+dh,k+dk,l+dl))
        else:
            d = peak.getDSpacing()

        name = 'peak_d={:.4f}_({:d},{:d},{:d})'+\
                            '_({:d},{:d},{:d})_lambda={:.4f}_run#{:d}'
        return name.format(d,*hkl,*mnp,lamda,run)

    def get_peak_shape(self, no, r_cut=np.inf):
        """
        Obtain the peak shape parameters.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        Q0, Q1, Q2 = mtd[self.peaks].getPeak(no).getQSampleFrame()

        shape = mtd[self.peaks].getPeak(no).getPeakShape()

        c0, c1, c2 = Q0, Q1, Q2

        if shape.shapeName() == 'ellipsoid':

            shape_dict = eval(shape.toJSON())

            c0 = Q0+shape_dict['translation0']
            c1 = Q1+shape_dict['translation1']
            c2 = Q2+shape_dict['translation2']

            v0 = [float(val) for val in shape_dict['direction0'].split(' ')]
            v1 = [float(val) for val in shape_dict['direction1'].split(' ')]
            v2 = [float(val) for val in shape_dict['direction2'].split(' ')]

            r0 = shape_dict['radius0']
            r1 = shape_dict['radius1']
            r2 = shape_dict['radius2']

            r0 = r0 if r0 < r_cut else r_cut
            r1 = r1 if r1 < r_cut else r_cut
            r2 = r2 if r2 < r_cut else r_cut

        else:

            r0 = r1 = r2 = r_cut
            v0, v1, v2 = np.eye(3).tolist()

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def set_peak_shape(self, no, c0, c1, c2, r0, r1, r2, v0, v1, v2):
        """
        Update the shape of the peak.

        Parameters
        ----------
        no : int
            Peak index number.
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        R = mtd[self.peaks].getPeak(no).getGoniometerMatrix()

        Q = np.array([c0, c1, c2])
        Qx, Qy, Qz = R @ Q

        radii = [0, 0, 0]

        if -4*np.pi*Qz/np.linalg.norm(Q)**2 > 0:

            mtd[self.peaks].getPeak(no).setQSampleFrame(V3D(c0, c1, c2))

            radii = [r0, r1, r2]

        shape = PeakShapeEllipsoid([V3D(*v0), V3D(*v1), V3D(*v2)],
                                   radii,
                                   radii,
                                   radii)

        mtd[self.peaks].getPeak(no).setPeakShape(shape)


class PeaksStatisticsModel(PeaksModel):

    def __init__(self, peaks):

        super(PeaksModel, self).__init__(peaks)

        self.peaks = peaks+'_stats'

        CloneWorkspace(InputWorkspace=peaks,
                       OutputWorkspace=self.peaks)

    def set_scale(self, scale='auto'):

        if scale == 'auto':
            scale = 1
            if mtd[self.peaks].getNumberPeaks() > 1:
                I_max = max(mtd[self.peaks].column('Intens'))
                if I_max > 0:
                    scale = 1e4/I_max

        for peak in mtd[self.peaks]:
            peak.setIntensity(scale*peak.getIntensity())
            peak.setSigmaIntensity(scale*peak.getSigmaIntensity())

