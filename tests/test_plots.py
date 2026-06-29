import os

import numpy as np
import scipy.linalg
import scipy.spatial

from garnet.plots.peaks import PeakPlot, RadiusPlot
from garnet.plots.volume import SlicePlot
from garnet.reduction.integration import PeakEllipsoid, PeakSphere


def test_slice_plot(tmpdir):
    np.random.seed(13)

    U = scipy.spatial.transform.Rotation.random().as_matrix()

    a, b, c = 5, 5, 7
    alpha, beta, gamma = np.deg2rad([90, 90, 120])

    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
        ]
    )

    B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    UB = U @ B

    W = np.eye(3)

    x0 = np.linspace(-5, 10, 31)
    x1 = np.linspace(-3, 9, 25)
    x2 = np.linspace(-7, 8, 61)

    axes = x0, x1, x2

    X0, X1, X2 = np.meshgrid(*axes, indexing="ij")

    signal = np.ones_like(X0)

    signal[(X0 % 1 == 0) & (X1 % 1 == 0) & (X2 % 1 == 0)] = 1000

    plot = SlicePlot(UB, W)
    params = plot.calculate_transforms(signal, axes, ["h", "k", "l"], [0, 0, 1], 0)
    plot.make_slice(*params)
    slice2 = os.path.join(tmpdir, "slice2.png")
    plot.save_plot(slice2)
    assert os.path.exists(slice2)

    plot = SlicePlot(UB, W)
    params = plot.calculate_transforms(signal, axes, ["h", "k", "l"], [0, 1, 0], 0)
    plot.make_slice(*params)
    slice1 = os.path.join(tmpdir, "slice1.png")
    plot.save_plot(slice1)
    assert os.path.exists(slice1)

    plot = SlicePlot(UB, W)
    params = plot.calculate_transforms(signal, axes, ["h", "k", "l"], [1, 0, 0], 0)
    plot.make_slice(*params)
    slice0 = os.path.join(tmpdir, "slice0.png")
    plot.save_plot(slice0)
    assert os.path.exists(slice0)


def test_radius_plot(tmpdir):
    r_cut = 0.25

    A = 1.2
    s = 0.1

    r = np.linspace(0, r_cut, 51)

    integration = A * np.tanh((r / s) ** 3)

    sphere = PeakSphere(r_cut)

    radius = sphere.fit(r, integration)

    I_fit, *vals = sphere.best_fit(r)

    plot = RadiusPlot(r, integration, I_fit)

    plot.add_sphere(radius, *vals)

    sphere = os.path.join(tmpdir, "sphere.png")
    plot.save_plot(sphere)
    assert os.path.exists(sphere)


def test_peak_plot(tmpdir):
    np.random.seed(13)

    nx, ny, nz = 21, 24, 29

    Qx_min, Qx_max = 0, 2
    Qy_min, Qy_max = -1.9, 2.1
    Qz_min, Qz_max = -3.2, 0.8

    Q0_x, Q0_y, Q0_z = 1.1, 0.1, -1.2

    sigma_x, sigma_y, sigma_z = 0.15, 0.25, 0.1
    rho_yz, rho_xz, rho_xy = 0.5, -0.1, -0.12

    a = 0.3
    b = 1.4
    c = 6.4
    d = 5.0

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

    signal = np.random.multivariate_normal(Q0, cov, size=10000)

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

    data_norm *= c
    data_norm += b + a * (2 * np.random.random(data_norm.shape) - 1)

    norm = np.full_like(data_norm, d)
    data = data_norm * norm

    m = 1

    i, j, k = np.array(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")).reshape(3, -1)

    i_min = np.clip(i - m, 0, nx)
    i_max = np.clip(i + m + 1, 0, nx)
    j_min = np.clip(j - m, 0, ny)
    j_max = np.clip(j + m + 1, 0, ny)
    k_min = np.clip(k - m, 0, nz)
    k_max = np.clip(k + m + 1, 0, nz)

    n = nx * ny * nz

    ic = np.random.randint(i_min, i_max, size=n).ravel()
    jc = np.random.randint(j_min, j_max, size=n).ravel()
    kc = np.random.randint(k_min, k_max, size=n).ravel()

    temp = np.copy(data)
    data[i, j, k] = data[ic, jc, kc]
    data[ic, jc, kc] = temp[i, j, k]

    mask = np.random.random(norm.shape) < 0.05
    norm[mask] = 0
    data[mask] = 0

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing="ij")

    params = 1.05, 0.05, -1.15, 0.5, 0.5, 0.5, [1, 0, 0], [0, 1, 0], [0, 0, 1]

    ellipsoid = PeakEllipsoid(*params, 2, 2)

    params = ellipsoid.fit(Qx, Qy, Qz, data, norm)

    c, S, W, *fitting = ellipsoid.best_fit

    vals = ellipsoid.interp_fit

    intens, sig_noise = ellipsoid.intens_fit

    assert ellipsoid.volume_fraction(Qx, Qy, Qz, data, norm, *params) > 0.75

    plot = PeakPlot(fitting)

    plot.add_ellipsoid(c, S, W, vals)
    plot.add_peak_intensity(intens, sig_noise)

    file = os.path.join(tmpdir, "ellipsoid.png")
    plot.save_plot(file)

    assert os.path.exists(file)
