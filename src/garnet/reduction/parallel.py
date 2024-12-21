import os
import sys
import traceback

import concurrent.futures
import multiprocessing

from concurrent.futures import ProcessPoolExecutor

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
        Run parallel tasks with concurrent futures.

        Parameters
        ----------
        plan : dict
            Data reduction plan split over each process.
        n_proc : int
            Number of processes.

        """

        runs = plan['Runs']

        split = [split.tolist() for split in np.array_split(runs, n_proc)]
        join_args = [(plan, s, proc) for proc, s in enumerate(split)]

        config['MultiThreaded.MaxCores'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TBB_THREAD_ENABLED'] = '0'

        mp_context = multiprocessing.get_context('spawn')
                                       
        try:
            with ProcessPoolExecutor(max_workers=n_proc,
                                     mp_context=mp_context) as executor:
                future_to_task = {
                    executor.submit(self.safe_function_wrapper, *args):
                        args for args in join_args
                    }
                self.results = []

                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        self.results.append(future.result())
                    except Exception as e:
                        print(f"Exception in worker function: {e}")
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)
                        sys.exit(1)

        except Exception as e:
            print(f"Exception in pool: {e}")
            traceback.print_exc()
            sys.exit(1)

        finally:
            config['MultiThreaded.MaxCores'] = '4'
            os.environ.pop('OPENBLAS_NUM_THREADS', None)
            os.environ.pop('MKL_NUM_THREADS', None)
            os.environ.pop('NUMEXPR_NUM_THREADS', None)
            os.environ.pop('OMP_NUM_THREADS', None)
            os.environ.pop('TBB_THREAD_ENABLED', None)

        if self.combine is not None:
            self.combine(plan, self.results)

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print(f"Exception in worker function: {e}")
            traceback.print_exc()
            raise

class ParallelProcessor:

    def __init__(self, n_proc=1):

        self.n_proc = n_proc

    def process_dict(self, data, func):

        if self.n_proc > 1:
            with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
                results = executor.map(func, data.items())
        else:
            results = [func(kv) for kv in data.items()]

        return dict(results)
