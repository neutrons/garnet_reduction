import os

import pytest

from garnet.reduction.integration import Integration
from garnet.reduction.normalization import Normalization
from garnet.reduction.plan import ReductionPlan

filepath = os.path.dirname(os.path.abspath(__file__))


def test_runs():
    garnet_plan = ReductionPlan()

    assert garnet_plan.plan is None

    run_str = "345:347,349,350:352"

    runs = garnet_plan.runs_string_to_list(run_str)

    assert runs == [345, 346, 347, 349, 350, 351, 352]

    assert garnet_plan.runs_list_to_string(runs) == "345:347,349:352"


@pytest.mark.mount_sns
def test_load_plan(has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    garnet_plan = ReductionPlan()

    reduction_plan = os.path.join(filepath, "data/corelli_reduction_plan.yaml")

    garnet_plan.load_plan(reduction_plan)

    plan = garnet_plan.plan

    assert plan is not None

    assert os.path.splitext(plan["DetectorCalibration"])[1] == ".xml"


@pytest.mark.mount_sns
def test_save_plan(has_sns_mount, tmpdir):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    garnet_plan = ReductionPlan()

    reduction_plan = os.path.join(filepath, "data/corelli_reduction_plan.yaml")

    garnet_plan.load_plan(reduction_plan)

    tmp_name = "tmp_plan.yaml"
    tmp_plan = os.path.join(tmpdir, tmp_name)

    assert garnet_plan.plan["OutputName"] == "corelli_reduction_plan"
    assert garnet_plan.plan["OutputPath"] == os.path.join(filepath, "data")

    garnet_plan.save_plan(tmp_plan)

    assert garnet_plan.plan["OutputName"] == "tmp_plan"
    assert garnet_plan.plan["OutputPath"] == tmpdir

    tmp_garnet_plan = ReductionPlan()

    tmp_garnet_plan.load_plan(tmp_plan)

    garnet_plan.plan == tmp_garnet_plan.plan

    assert tmp_garnet_plan.plan["OutputName"] == "tmp_plan"
    assert tmp_garnet_plan.plan["OutputPath"] == tmpdir


@pytest.mark.mount_sns
def test_integration_plan(has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    garnet_plan = ReductionPlan()

    reduction_plan = os.path.join(filepath, "data/corelli_reduction_plan.yaml")

    garnet_plan.load_plan(reduction_plan)

    params = garnet_plan.plan.get("Integration")

    assert params is not None

    assert params["Cell"] == "Cubic"
    assert params["Centering"] == "I"
    assert params["ModVec1"] == [0, 0, 0]
    assert params["ModVec2"] == [0, 0, 0]
    assert params["ModVec3"] == [0, 0, 0]
    assert params["MaxOrder"] == 0
    assert params["CrossTerms"] is False
    assert params["MinD"] == 0.7
    assert params["Radius"] == 0.25

    integrate = Integration(garnet_plan.plan)

    assert integrate is not None


@pytest.mark.mount_sns
def test_normalization_plan(has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    garnet_plan = ReductionPlan()

    reduction_plan = os.path.join(filepath, "data/corelli_reduction_plan.yaml")

    garnet_plan.load_plan(reduction_plan)

    params = garnet_plan.plan.get("Normalization")

    assert params is not None

    assert params["Symmetry"] == "m-3m"

    normalize = Normalization(garnet_plan.plan)

    assert normalize is not None
