import os

from mantid.simpleapi import mtd
from mantid import config
config['Q.convention'] = 'Crystallography'

import numpy as np

from garnet.reduction.data import DataModel
from garnet.reduction.crystallography import space_point, point_laue
from garnet.config.instruments import beamlines

class Normalization:

    def __init__(self, plan):

        self.plan = plan
        self.params = plan['Normalization']

        self.validate_params()

    def validate_params(self):

        symbols = list(space_point.keys())+list(point_laue.keys())

        if self.params.get('Symmetry') is not None:

            symmetry = self.params['Symmetry'].replace(' ','')
            assert symmetry in symbols
            if space_point.get(symmetry) is not None:
                symmetry = space_point[symmetry]
            symmetry = point_laue[symmetry]
            self.params['Symmetry'] = symmetry

        assert len(self.params['Projections']) == 3
        assert np.abs(np.linalg.det(self.params['Projections'])) > 0

        assert len(self.params['Bins']) == 3
        assert all([type(val) is int for val in self.params['Bins']])
        assert (np.array(self.params['Bins']) > 0).all()
        assert np.prod(self.params['Bins']) < 1001**3 # memory usage limit

        assert len(self.params['Extents']) == 3
        assert (np.diff(self.params['Extents'], axis=1) >= 0).all()

    @staticmethod
    def normalize_parallel(plan, runs, proc):

        plan['Runs'] = runs
        plan['OutputName'] += '_p{}'.format(proc)

        instance = Normalization(plan)

        return instance.normalize()

    def normalize(self):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'normalization',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        data.load_generate_normalization(self.plan['VanadiumFile'],
                                         self.plan.get('FluxFile'))

        runs = self.plan['Runs']

        if data.laue:

            for run in runs:

                data.load_data('data', self.plan['IPTS'], run)

                data.apply_calibration('data',
                                       self.plan.get('DetectorCalibration'),
                                       self.plan.get('TubeCalibration'))

                data.apply_mask('data', self.plan.get('MaskFile'))

                data.crop_for_normalization('data')

                data.load_clear_UB(self.plan['UBFile'], 'data')

                data.convert_to_Q_sample('data', 'md', lorentz_corr=False)

                data.load_background(self.plan.get('BackgroundFile'), 'data')

                data.normalize_to_hkl('md',
                                      self.params['Projections'],
                                      self.params['Extents'],
                                      self.params['Bins'],
                                      symmetry=self.params.get('Symmetry'))
        else:

            if self.plan['Instrument'] == 'WAND²':

                data.load_data('md', self.plan['IPTS'], runs)

                if self.plan['UBFile'] is not None:

                    data.load_clear_UB(self.plan['UBFile'], 'md')

                data.load_background(self.plan.get('BackgroundFile'), 'md')

                data.normalize_to_hkl('md',
                                      self.params['Projections'],
                                      self.params['Extents'],
                                      self.params['Bins'],
                                      symmetry=self.params.get('Symmetry'))

            else:

                for run in runs:

                    data.load_data('md', self.plan['IPTS'], run)

                    if self.plan['UBFile'] is not None:

                        data.load_clear_UB(self.plan['UBFile'], 'md')

                    data.load_background(self.plan.get('BackgroundFile'), 'md')

                    data.normalize_to_hkl('md',
                                          self.params['Projections'],
                                          self.params['Extents'],
                                          self.params['Bins'],
                                          symmetry=self.params.get('Symmetry'))

        UB_file = output_file.replace('.nxs','.mat')
        data.save_UB(UB_file, 'md')

        data_file = self.get_file(output_file, 'data')
        norm_file = self.get_file(output_file, 'norm')

        data.save_histograms(data_file, 'md_data')
        data.save_histograms(norm_file, 'md_norm')

        if mtd.doesExist('md_bkg_data') and mtd.doesExist('md_bkg_norm'):

            data_file = self.get_file(output_file, 'bkg_data')
            norm_file = self.get_file(output_file, 'bkg_norm')

            data.save_histograms(data_file, 'md_bkg_data')
            data.save_histograms(norm_file, 'md_bkg_norm')

        mtd.clear()

        return output_file

    def get_file(self, file, ws=''):

        if len(ws) > 0:
            ws = '_'+ws

        return self.append_name(file).replace('.nxs', ws+'.nxs')

    def append_name(self, file):

        append = self.projection_name() \
               + self.extents_name() \
               + self.binning_name() \
               + self.symmetry_name()

        name, ext = os.path.splitext(file)

        return name+append+ext

    def extents_name(self):

        extents = self.params.get('Extents')

        return ''.join(['_[{},{}]'.format(*extent) for extent in extents])

    def binning_name(self):

        bins = self.params.get('Bins')

        return '_'+'x'.join(np.array(bins).astype(str).tolist())

    def symmetry_name(self):

        symmetry = self.params.get('Symmetry')

        name = '' if symmetry is None else '_'+symmetry.replace(' ', '')

        return name.replace('/', '_')

    def projection_name(self):

        W = np.column_stack(self.params['Projections'])

        char_dict = {0: '0', 1: '{1}', -1: '-{1}'}

        chars = ['h', 'k', 'l']

        axes = []
        for j in [0, 1]:
            axis = []
            for w in W[:,j]:
                char = chars[np.argmax(W[:, j])]
                axis.append(char_dict.get(w, '{0}{1}').format(j, char))
            axes.append(axis)

        result = []
        for item0, item1 in zip(axes[0], axes[1]):
            if item0 == '0':
                result.append(item1)
            elif item1 == '0':
                result.append(item0)
            elif '-' in item1:
                result.append(item0+item1)
            else:
                result.append(item0+'+'+item1)

        proj = '_('+','.join(result)+')'      

        return proj

    @staticmethod
    def combine_parallel(plan, files):

        instance = Normalization(plan)

        return instance.combine(files)

    def combine(self, files):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'normalization',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        for ind, file in enumerate(files):

            data_file = self.get_file(file, 'data')
            norm_file = self.get_file(file, 'norm')

            data.load_histograms(data_file, 'tmp_data')
            data.load_histograms(norm_file, 'tmp_norm')

            data.combine_histograms('tmp_data', 'data')
            data.combine_histograms('tmp_norm', 'norm')

            bkg_data_file = self.get_file(file, 'bkg_data')
            bkg_norm_file = self.get_file(file, 'bkg_norm')

            if os.path.exists(bkg_data_file) and os.path.exists(bkg_norm_file):

                data.load_histograms(bkg_data_file, 'tmp_bkg_data')
                data.load_histograms(bkg_norm_file, 'tmp_bkg_norm')

                data.combine_histograms('tmp_bkg_data', 'bkg_data')
                data.combine_histograms('tmp_bkg_norm', 'bkg_norm')

                os.remove(bkg_data_file)
                os.remove(bkg_norm_file)

            os.remove(data_file)
            os.remove(norm_file)

        data_file = self.get_file(output_file, 'data')
        norm_file = self.get_file(output_file, 'norm')
        result_file = self.get_file(output_file, '')

        data.divide_histograms('result', 'data', 'norm')

        UB_file = file.replace('.nxs','.mat')

        for ws in ['data', 'norm', 'result']:

            data.add_UBW(ws, UB_file, self.params['Projections'])

        for ind, file in enumerate(files):

            UB_file = file.replace('.nxs','.mat')

            os.remove(UB_file)

        data.save_histograms(data_file, 'data', sample_logs=True)
        data.save_histograms(norm_file, 'norm', sample_logs=True)
        data.save_histograms(result_file, 'result', sample_logs=True)

        if mtd.doesExist('bkg_data') and mtd.doesExist('bkg_norm'):

            data_file = self.get_file(output_file, 'bkg_data')
            norm_file = self.get_file(output_file, 'bkg_norm')

            data.save_histograms(data_file, 'bkg_data', sample_logs=True)
            data.save_histograms(norm_file, 'bkg_norm', sample_logs=True)

            bkg_output_file = self.get_file(output_file, 'bkg')

            data.divide_histograms('bkg_result', 'bkg_data', 'bkg_norm')
            data.save_histograms(bkg_output_file, 'bkg_result')

            data.subtract_histograms('sub', 'result', 'bkg_result')

            sub_output_file = self.get_file(output_file, 'sub_bkg')
            data.save_histograms(sub_output_file, 'sub', sample_logs=True)
