import os
import subprocess

import pytest

from garnet.reduction.normalization import Normalization
from garnet.reduction.plan import ReductionPlan

benchmark = "shared/benchmark/norm"


def test_get_file():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")

    data = Normalization(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_(h,k,0)_[0,0,l]_[-10,10]_[-10,10]_[-10,10]_201x201x201"
    ext = ".nxs"
    symm = "_2_m"

    assert data == base + app + ext

    data = Normalization(rp.plan).get_file(file, ws="data")

    assert data == base + app + "_data" + ext

    rp.plan["Normalization"]["Symmetry"] = "2/m"

    data = Normalization(rp.plan).get_file(file, ws="")

    assert data == base + app + symm + ext

    data = Normalization(rp.plan).get_file(file, ws="data")

    assert data == base + app + symm + "_data" + ext


@pytest.mark.resources_intensive
@pytest.mark.mount_sns
def test_corelli(tmpdir, has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    config_file = "corelli_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")
    # reduction plan filepaths point to analysis
    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    command = ["python", script, saved_plan, "norm", "16"]
    subprocess.run(command, check=False)


@pytest.mark.resources_intensive
@pytest.mark.mount_sns
def test_topaz(tmpdir, has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    config_file = "topaz_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")

    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    command = ["python", script, saved_plan, "norm", "6"]
    subprocess.run(command, check=False)


@pytest.mark.resources_intensive
@pytest.mark.mount_hfir
def test_demand(tmpdir, has_hfir_mount):
    if not has_hfir_mount:
        pytest.skip("Test is skipped. HFIR mount is not available.")

    config_file = "demand_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")

    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    command = ["python", script, saved_plan, "norm", "4"]
    subprocess.run(command, check=False)


@pytest.mark.resources_intensive
@pytest.mark.mount_hfir
def test_wand2(tmpdir, has_hfir_mount):
    if not has_hfir_mount:
        pytest.skip("Test is skipped. HFIR mount is not available.")

    config_file = "wand2_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")

    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    command = ["python", script, saved_plan, "norm", "48"]
    subprocess.run(command, check=False)
