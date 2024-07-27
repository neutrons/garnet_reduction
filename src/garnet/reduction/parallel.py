import os

import multiprocess as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from mantid import config
config['Q.convention'] = 'Crystallography'
config.setLogLevel(0, quiet=True)

class ParallelTasks:

    def __init__(self, function, combine=None):

        self.function = function
        self.combine = combine
        self.results = None

    def run_tasks(self, plan, n_proc):
        """
        Run parallel tasks with processing pool.

        Parameters
        ----------
        plan : dict
            Data reduction plan split over each process.
        n_proc : int
            Number of processes.

        """

        runs = plan['Runs']

        pool = multiprocessing.Pool(processes=n_proc)

        split = [split.tolist() for split in np.array_split(runs, n_proc)]

        join_args = [(plan, s, proc) for proc, s in enumerate(split)]

        config['MultiThreaded.MaxCores'] == '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TBB_THREAD_ENABLED'] = '0'

        self.results = pool.starmap(self.safe_function_wrapper, join_args)

        pool.close()
        pool.join()

        config['MultiThreaded.MaxCores'] == '4'
        os.environ.pop('OPENBLAS_NUM_THREADS')
        os.environ.pop('MKL_NUM_THREADS')
        os.environ.pop('NUMEXPR_NUM_THREADS')
        os.environ.pop('OMP_NUM_THREADS')
        os.environ.pop('TBB_THREAD_ENABLED')

        if self.combine is not None:
            self.combine(plan, self.results)

    def safe_function_wrapper(self, *args, **kwargs):

        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print('Exception in worker function: {}'.format(e))
            import traceback
            traceback.print_exc()
            raise
