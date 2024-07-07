import os

import numpy as np

from mantid.simpleapi import mtd
from mantid import config
config['Q.convention'] = 'Crystallography'

from garnet.config.instruments import beamlines
from garnet.reduction.ub import UBModel
from garnet.reduction.peaks import PeaksModel
from garnet.reduction.data import DataModel
from garnet.reduction.plan import SubPlan

class Determination(SubPlan):

    def __init__(self, plan):
        """
        Tool for UB matrix determination and refimenent.

        Parameters
        ----------
        plan : dict
            Reduction plan.

        """

        super(Determination, self).__init__(plan)

        self.output = 'determination'

        self.data = DataModel(beamlines[plan['Instrument']])
        self.data.update_raw_path(self.plan)

        self.table = None
        self.Q = None

        self.peaks = PeaksModel()

    def load_data(self, skip_runs=None,
                        apply_lorentz=True,
                        time_cut=None):

        if skip_runs is None:
            skip_runs = len(self.plan['Runs']) if self.data.laue else 1

        if self.data.laue:

            grouping_file = self.get_diagnostic_file('grouping', '.xml')

            self.data.preprocess_detectors()
            self.data.create_grouping(grouping_file, self.plan.get('Grouping'))
            mtd.remove('detectors')
            self.plan['GroupingFile'] = grouping_file

            runs = self.plan['Runs'][::skip_runs]

            self.data.load_data('data', self.plan['IPTS'], runs, time_cut)

            self.data.apply_calibration('data',
                                        self.plan.get('DetectorCalibration'),
                                        self.plan.get('TubeCalibration'))

            self.data.apply_mask('data', self.plan.get('MaskFile'))

            self.data.crop_for_normalization('data')

            self.data.group_pixels(grouping_file, 'data')

            if self.plan['UBFile'] is not None:

                self.data.load_clear_UB(self.plan['UBFile'], 'data')

            self.data.convert_to_Q_sample('data',
                                          'md',
                                          lorentz_corr=apply_lorentz)

        else:

            UB_file = self.get_output_file('.mat')

            if self.plan['Instrument'] == 'WAND²':

                self.data.load_data('data',
                                    self.plan['IPTS'],
                                    runs,
                                    self.plan.get('Grouping'))


                self.data.convert_to_Q_sample('data',
                                              'md',
                                              lorentz_corr=apply_lorentz)

                if self.plan.get('UBFile') is None:
                    self.data.save_UB(UB_file, 'md')
                    self.plan['UBFile'] = UB_file

            else:

                for run in runs:

                    self.data.load_data('data',
                                        self.plan['IPTS'],
                                        run,
                                        self.plan.get('Grouping'))

                    self.data.convert_to_Q_sample('data',
                                                  'tmp',
                                                  lorentz_corr=apply_lorentz)

                    self.data.combine_histograms('tmp', 'md')

                    if self.plan.get('UBFile') is None:
                        self.data.save_UB(UB_file, 'md_data')
                        self.plan['UBFile'] = UB_file

            self.data.load_clear_UB(self.plan['UBFile'], 'md_data')

        self.Q = 'md'

    def has_Q(self):

        if self.Q is None:
            return False
        elif mtd.doesExist(self.Q):
            return True
        else:
            return False

    def has_peaks(self):

        if self.table is None:
            return False
        elif mtd.doesExist(self.table):
            return True
        else:
            return False

    def has_UB(self):

        if self.has_peaks():
            ol = UBModel(self.table)
            if ol.has_UB():
                return True
            else:
                return False
        else:
            return False

    def get_UB(self):

        return UBModel(self.table)

    def convert_to_hkl(self, projections, extents, bins):

        if self.has_Q() and self.has_UB():

            self.data.convert_to_hkl(self.Q,
                                     self.table,
                                     projections,
                                     extents,
                                     bins)

    def find_peaks(self, max_d, density, max_peaks):

        if self.has_Q():

            self.peaks.find_peaks(self.Q,
                                  'peaks',
                                  max_d,
                                  density,
                                  max_peaks)

            self.table = 'peaks'

    def integrate_peaks(self, radius,
                              inner_fact,
                              outer_fact,
                              adaptive,
                              centroid):

        if self.has_Q() and self.has_peaks():

            method = 'ellipsoid' if adaptive else 'sphere'

            self.peaks.integrate_peaks(self.Q,
                                       self.table,
                                       radius,
                                       inner_fact,
                                       outer_fact,
                                       method=method,
                                       centroid=centroid)

    def find_primitive_UB(self, min_d, max_d, tol):

        if self.has_peaks():

            ub = self.get_UB()

            ub.determine_UB_with_niggli_cell(min_d, max_d, tol)

    def list_possible_conventional_cells(self, max_error, permutations):

        if self.has_peaks() and self.has_UB():

            ub = self.get_UB()

            cells = ub.possible_conventional_cells(max_error, permutations)
            print(cells)

    def find_conventional_UB(self, a, b, c, alpha, beta, gamma, tol):

        if self.has_peaks():

            ub = self.get_UB()

            ub.determine_UB_with_lattice_parameters(a,
                                                    b,
                                                    c,
                                                    alpha,
                                                    beta,
                                                    gamma,
                                                    tol)

    def predict_peaks(self, centering,
                            d_min,
                            lamda_min,
                            lamda_max,
                            d_min_sat,
                            mod_vec_1,
                            mod_vec_2,
                            mod_vec_3,
                            max_order,
                            cross_terms):

        if self.has_Q() and self.has_UB():

            self.peaks.predict_peaks(self.Q,
                                    'peaks',
                                    centering,
                                    d_min,
                                    lamda_min,
                                    lamda_max)

            valid_mod = self._modulation(mod_vec_1, mod_vec_2, mod_vec_3)

            if max_order > 0 and valid_mod:

                if d_min_sat is None:
                    d_min_sat = d_min

                self.peaks.predict_satellite_peaks('peaks',
                                                   self.Q,
                                                   lamda_min,
                                                   lamda_max,
                                                   d_min_sat,
                                                   mod_vec_1,
                                                   mod_vec_2,
                                                   mod_vec_3,
                                                   max_order,
                                                   cross_terms)

            self.table = 'peaks'

    def _modulation(self, mod_vec_1, mod_vec_2, mod_vec_3):

        if not any(mod_vec_1):
            return False

        if np.any(mod_vec_3) and not np.any(mod_vec_2):
            return False

        if self._collinear(mod_vec_1, mod_vec_2) or \
           self._collinear(mod_vec_2, mod_vec_3) or \
           self._collinear(mod_vec_1, mod_vec_3):
            return False

        return True

    def _collinear(self, vec_1, vec_2):
        if not np.any(vec_1) and not np.any(vec_2):
            return False
        return np.linalg.norm(np.cross(vec_1, vec_2)) == 0