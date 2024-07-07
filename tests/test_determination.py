import os
import pytest
import tempfile

import numpy as np

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.determination import Determination

benchmark = 'shared/benchmark/ref'

@pytest.mark.skipif(not os.path.exists('/SNS/TOPAZ/'), reason='file mount')
def test_topaz():

    config_file = 'topaz_reduction_plan.yaml'
    reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))

    with tempfile.TemporaryDirectory() as tmpdir:

        os.chdir(tmpdir)

        rp = ReductionPlan()
        rp.load_plan(reduction_plan)
        rp.save_plan(os.path.join(tmpdir, config_file))

        rp = ReductionPlan()
        rp.load_plan(os.path.join(tmpdir, config_file))

        det = Determination(rp.plan)
        det.create_directories()

        det.load_data(time_cut=60)

        assert det.has_Q()

        a = 6.51
        b = 18.95
        c = 9.76
        beta = 108.86
        alpha = gamma = 90

        max_d = b

        det.find_peaks(max_d, 100, 100)

        assert det.has_peaks()

        det.find_conventional_UB(a, b, c, alpha, beta, gamma, 0.1)

        assert det.has_UB()

        det.find_primitive_UB(5, max_d, 0.1)

        assert det.has_UB()

        # vals = det.list_possible_conventional_cells(0.2, True)

        det.predict_peaks('C', 
                          0.5,
                          0.5,
                          3.5,
                          None,
                          [0,0,0],
                          [0,0,0],
                          [0,0,0],
                          1,
                          False)