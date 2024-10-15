import os
import sys
import yaml
import shutil
import concurrent.futures

from garnet.config.instruments import beamlines

class Dumper(yaml.Dumper):
    def represent_list(self, data):
        return self.represent_sequence('tag:yaml.org,2002:seq',
                                       data,
                                       flow_style=True)

Dumper.add_representer(list, Dumper.represent_list)

def save_YAML(output, filename):
    """
    Save reduction output file.

    Parameters
    ----------
    output : str
        Name of file.
    filename : str
        Output file name.

    """
    with open(filename, 'w') as f:

        yaml.dump(output, f, Dumper=Dumper, sort_keys=False)

def delete_directory(path):
    """
    Fast removal of nonempty directories.

    Parameters
    ----------
    path : str
        Directory to remove.

    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for root, dirs, files in os.walk(path, topdown=False):
            executor.map(os.remove,
                         [os.path.join(root, name) for name in files])
        shutil.rmtree(path, ignore_errors=True)

class SubPlan:

    def __init__(self, plan):

        self.plan = plan
        self.output = 'test'
        self.proc = 0

    def create_directories(self):

        output = self.get_output_path()
        if not os.path.exists(output):
            os.mkdir(output)

        for output in [self.get_plot_path(), self.get_diagnostic_path()]:
            if os.path.exists(output):
                delete_directory(output)
            os.mkdir(output)

    def get_output_file(self, ext='.nxs'):
        """
        Output file.

        Returns
        -------
        output_file : str
            Output file.

        """

        proc = ''
        if self.plan.get('ProcName') is not None:
            proc += self.plan['ProcName']

        output_file = os.path.join(self.get_output_path(),
                                   self.plan['OutputName']+proc+ext)

        return output_file

    def get_plot_file(self, name, ext='.png'):
        """
        Plot file.

        Returns
        -------
        name : str
            Name to use for file.
        plot_file : str
            Path name to save plots file.

        """

        return os.path.join(self.get_plot_path(), name+ext)

    def get_diagnostic_file(self, name, ext='.nxs'):
        """
        Diagnostic file.

        Returns
        -------
        name : str
            Name to use for file.
        diag_file : str
            Path name to save diagnostics file.

        """

        return os.path.join(self.get_diagnostic_path(), name+ext)

    def get_output_path(self):
        """
        Output path.

        Returns
        -------
        output_path : str
            Output path.

        """

        return os.path.join(self.plan['OutputPath'], self.output)

    def get_plot_path(self):
        """
        Plot path.

        Returns
        -------
        plot_path : str
            Path to save plots file.

        """

        plots = self.append_name(self.plan['OutputName'])+'_plots'

        return os.path.join(self.plan['OutputPath'], self.output, plots)

    def get_diagnostic_path(self):
        """
        Diagnostic path.

        Returns
        -------
        diag_path : str
            Path to save diagnostics file.

        """

        diagnostics = self.append_name(self.plan['OutputName'])+'_diagnostics'

        return os.path.join(self.plan['OutputPath'], self.output, diagnostics)

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

        return file

class ReductionPlan:

    def __init__(self):

        self.plan = None

    def validate_file(self, fname, ext):

        try:
            assert os.path.exists(fname)
        except AssertionError:
            print('{} does not exist!'.format(fname))
            sys.exit(1)
        try:
            if type(ext) is list:
                assert os.path.splitext(fname)[1].lower() in ext
            else:
                assert os.path.splitext(fname)[1].lower() == ext
        except AssertionError:
            print('{} not valid!'.format(fname))
            sys.exit(1)

    def validate_plan(self):

        assert self.plan['Instrument'] in beamlines.keys()

        if self.plan.get('UBFile') is not None:
            UB = self.plan['UBFile']
            self.validate_file(UB, '.mat')

        nxs_items = ['VanadiumFile',
                     'FluxFile',
                     'TubeCalibration',
                     'BackgroundFile']

        for item in nxs_items:
            if self.plan.get(item) is not None:
                fname = self.plan[item]
                self.validate_file(fname, ['.nxs', '.h5'])

        if self.plan.get('MaskFile') is not None:
            mask = self.plan['MaskFile']
            self.validate_file(mask, '.xml')
            assert os.path.splitext(mask)[1] == '.xml'

        if self.plan.get('DetectorCalibration') is not None:
            detcal = self.plan['DetectorCalibration']
            self.validate_file(detcal, ['.xml', '.detcal'])

    def set_output(self, filename):
        """
        Change the output directory and name.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """

        path = os.path.dirname(os.path.abspath(filename))
        self.plan['OutputPath'] = path

        if self.plan.get('OutputName') is None:
            name = os.path.splitext(os.path.basename(filename))[0]
            self.plan['OutputName'] = name 

    def load_plan(self, filename):
        """
        Load a data reduction plan.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """

        with open(filename, 'r') as f:

            self.plan = yaml.safe_load(f)

        self.validate_plan()

        self.set_output(filename)
        runs = self.plan['Runs']
        self.plan['Runs'] = self.runs_string_to_list(runs)

    def save_plan(self, filename):
        """
        Save a data reduction plan.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """

        if self.plan is not None:

            self.set_output(filename)
            runs = self.plan['Runs']
            if type(runs) is list:
                self.plan['Runs'] = self.runs_list_to_string(runs)

            save_YAML(self.plan, filename)

    def runs_string_to_list(self, runs_str):
        """
        Convert runs string to list.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list
            Integer run numbers.

        """

        ranges = runs_str.split(',')
        runs = []
        for part in ranges:
            if ':' in part:
                start, end = map(int, part.split(':'))
                runs.extend(range(start, end + 1))
            else:
                runs.append(int(part))
        return runs

    def runs_list_to_string(self, runs):
        """
        Convert runs list to string.

        Parameters
        ----------
        runs : list
            Integer run numbers.

        Returns
        -------
        runs_str : str
            Condensed notation for run numbers.

        """

        if not runs:
            return ''

        runs.sort()
        result = []
        range_start = runs[0]

        for i in range(1, len(runs)):
            if runs[i] != runs[i-1] + 1:
                if range_start == runs[i-1]:
                    result.append(str(range_start))
                else:
                    result.append('{}:{}'.format(range_start, runs[i-1]))
                range_start = runs[i]

        if range_start == runs[-1]:
            result.append(str(range_start))
        else:
            result.append('{}:{}'.format(range_start, runs[-1]))

        run_str = ','.join(result)

        return run_str

    def generate_plan(self, instrument):
        """
        Create a template plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        """

        plan = {}

        assert instrument in beamlines.keys()
        params = beamlines[instrument]

        plan['Instrument'] = instrument
        plan['IPTS'] = 0
        plan['Runs'] = '1:2'

        if instrument == 'DEMAND':
            plan['Experiment'] = 1

        if params['Facility'] == 'HFIR':
            plan['UBFile'] = None
        else:
            plan['UBFile'] = ''

        plan['VanadiumFile'] = ''
        plan['BackgroundFile'] = None

        if params['Facility'] == 'SNS':
            plan['FluxFile'] = ''
            plan['MaskFile'] = None
            plan['DetectorCalibration'] = None

        if instrument == 'CORELLI':
            plan['TubeCalibration'] = '/SNS/CORELLI/shared/calibration/tube'\
                                    + '/calibration_corelli_20200109.nxs.h5'
            plan['Elastic'] = False
            plan['TimeOffset'] = None

        self.plan = plan

        self.plan['Integration'] = self.template_integration(instrument)
        self.plan['Normalization'] = self.template_normalization()

        self.plan.pop('OutputPath')
        self.plan.pop('OutputName')

    def template_integration(self, instrument):
        """
        Generate template integration plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        Returns
        -------
        params : dict
            Integration plan.

        """

        inst_config = beamlines[instrument]

        wl = inst_config['Wavelength']
        min_d = max(wl)/2 if type(wl) is list else wl/2

        params = {}
        params['Cell'] = 'Triclinic'
        params['Centering'] = 'P'
        params['ModVec1'] = [0,0,0]
        params['ModVec2'] = [0,0,0]
        params['ModVec3'] = [0,0,0]
        params['MaxOrder'] = 0
        params['CrossTerms'] = False
        params['MinD'] = min_d
        params['Radius'] = 0.2

        return params

    def template_normalization(self):
        """
        Generate template integration plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        Returns
        -------
        params : dict
            Integration plan.

        """

        params = {}
        params['Symmetry'] = None
        params['Projections'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        params['Extents'] = [[-10, 10], [-10, 10], [-10, 10]]
        params['Bins'] = [201, 201, 201]

        return params