import os
import subprocess

import numpy as np
import pytest

from garnet.reduction.integration import PeakEllipsoid, PeakSphere
from garnet.reduction.plan import ReductionPlan

benchmark = "shared/benchmark/int"


@pytest.mark.resources_intensive
@pytest.mark.mount_sns
def test_corelli(tmpdir, has_sns_mount):
    if not has_sns_mount:
        pytest.skip("Test is skipped. SNS mount is not available.")

    config_file = "corelli_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")
    command = ["python", script, config_file, "int", "16"]

    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    command = ["python", script, saved_plan, "int", "16"]
    subprocess.run(command, check=False)


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

    command = ["python", script, saved_plan, "int", "4"]
    subprocess.run(command, check=False)


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

    command = ["python", script, saved_plan, "int", "4"]
    subprocess.run(command, check=False)


def test_sphere():
    r_cut = 0.25

    A = 1.2
    s = 0.1

    r = np.linspace(0, r_cut, 51)

    integration = A * np.tanh((r / s) ** 3)

    sphere = PeakSphere(r_cut)

    radius = sphere.fit(r, integration)

    assert np.tanh((radius / s) ** 3) > 0.95
    assert radius < r_cut


def test_ellipsoid():
    np.random.seed(13)

    nx, ny, nz = 41, 41, 41

    Qx_min, Qx_max = 0, 2
    Qy_min, Qy_max = -1.9, 2.1
    Qz_min, Qz_max = -3.2, 0.8

    Q0_x, Q0_y, Q0_z = 1.1, 0.1, -1.2

    sigma_x, sigma_y, sigma_z = 0.1, 0.15, 0.12
    rho_yz, rho_xz, rho_xy = 0.1, -0.1, -0.15

    a = 0.2
    b = 0.5
    c = 1.3

    sigma_yz = sigma_y * sigma_z
    sigma_xz = sigma_x * sigma_z
    sigma_xy = sigma_x * sigma_y

    cov = np.array(
        [
            [sigma_x**2, rho_xy * sigma_xy, rho_xz * sigma_xz],
            [rho_xy * sigma_xy, sigma_y**2, rho_yz * sigma_yz],
            [rho_xz * sigma_xz, rho_yz * sigma_yz, sigma_z**2],
        ]
    )

    Q0 = np.array([Q0_x, Q0_y, Q0_z])

    signal = np.random.multivariate_normal(Q0, cov, size=1000000)

    data_norm, bins = np.histogramdd(
        signal,
        density=False,
        bins=[nx, ny, nz],
        range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)],
    )

    data_norm /= np.max(data_norm)
    data_norm /= np.sqrt(np.linalg.det(2 * np.pi * cov))

    x_bin_edges, y_bin_edges, z_bin_edges = bins

    Qx = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
    Qy = 0.5 * (y_bin_edges[1:] + y_bin_edges[:-1])
    Qz = 0.5 * (z_bin_edges[1:] + z_bin_edges[:-1])

    data = data_norm * c + b + a * (2 * np.random.random(data_norm.shape) - 1)
    norm = np.full_like(data, c)

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing="ij")

    params = 1.05, 0.05, -1.15, 0.5, 0.5, 0.5, [1, 0, 0], [0, 1, 0], [0, 0, 1]

    ellipsoid = PeakEllipsoid(*params, 1, 1)

    params = ellipsoid.fit(Qx, Qy, Qz, data, norm)

    mu = params[0:3]
    radii = params[3:6]
    vectors = params[6:9]

    S = ellipsoid.S_matrix(*ellipsoid.scale(*radii), *ellipsoid.angles(*vectors))

    s = np.sqrt(np.linalg.det(S))
    sigma = np.sqrt(np.linalg.det(cov))

    assert np.isclose(mu, Q0, atol=0.01).all()
    assert np.isclose(s, sigma, atol=0.001).all()


def test_ellipsoid_methods():
    params = 1.05, 0.05, -1.15, 0.5, 0.5, 0.5, [1, 0, 0], [0, 1, 0], [0, 0, 1]

    ellipsoid = PeakEllipsoid(*params, 1, 1)

    vals = 1.0, 2.0, 3.0, 0.2, 1.1, -0.4

    S = ellipsoid.S_matrix(*vals)
    inv_S = ellipsoid.inv_S_matrix(*vals)

    assert np.allclose(np.linalg.inv(S), inv_S)

    P = np.eye(3) - np.outer(ellipsoid.n, ellipsoid.n)

    assert np.allclose(ellipsoid.u, P @ ellipsoid.u)
    assert np.allclose(ellipsoid.v, P @ ellipsoid.v)

    W = np.column_stack([ellipsoid.n, ellipsoid.u, ellipsoid.v])

    assert np.isclose(np.abs(np.linalg.det(W)), 1)

    x = np.linspace(1, 3, 1000)

    dx = x[1] - x[0]

    A, B, mu, sigma = 1.2, 0.2, 2, 0.2

    y = ellipsoid.profile(x, A, B, mu, sigma)
    yp = ellipsoid.profile_grad(x, A, B, mu, sigma)

    grad_y = np.gradient(y, dx)

    assert np.allclose(yp, grad_y, rtol=1e-2, atol=1e-4)

    xu = np.linspace(1, 3, 500)
    xv = np.linspace(2, 4, 501)

    dxu, dxv = xu[1] - xu[0], xv[1] - xv[0]

    xu, xv = np.meshgrid(xu, xv, indexing="ij")

    mu_u, mu_v, sigma_u, sigma_v, rho = 2, 3, 0.4, 0.5, 0.1

    y = ellipsoid.projection(xu, xv, A, B, mu_u, mu_v, sigma_u, sigma_v, rho)
    ypu, ypv = ellipsoid.projection_grad(xu, xv, A, B, mu_u, mu_v, sigma_u, sigma_v, rho)

    grad_yu, grad_yv = np.gradient(y, dxu, dxv)

    assert np.allclose(ypu, grad_yu, rtol=1e-2, atol=1e-3)
    assert np.allclose(ypv, grad_yv, rtol=1e-2, atol=1e-3)

    x0 = np.linspace(1, 3, 250)
    x1 = np.linspace(2, 4, 251)
    x2 = np.linspace(3, 5, 252)

    dx0, dx1, dx2 = x0[1] - x0[0], x1[1] - x1[0], x2[1] - x2[0]

    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    c = [2, 3, 4]
    S = np.array([[0.04, 0.005, 0.003], [0.005, 0.041, 0.02], [0.003, 0.02, 0.042]])

    y = ellipsoid.func(x0, x1, x2, A, B, c, S)
    yp0, yp1, yp2 = ellipsoid.func_grad(x0, x1, x2, A, B, c, S)

    grad_y0, grad_y1, grad_y2 = np.gradient(y, dx0, dx1, dx2)

    assert np.allclose(yp0, grad_y0, rtol=1e-2, atol=1e-3)
    assert np.allclose(yp1, grad_y1, rtol=1e-2, atol=1e-3)
    assert np.allclose(yp2, grad_y2, rtol=1e-2, atol=1e-3)
