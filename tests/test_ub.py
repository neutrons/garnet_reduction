import numpy as np
from mantid.simpleapi import AddPeakHKL, CreatePeaksWorkspace, SetUB

from garnet.reduction.ub import Optimization


def create_peaks(name, a, b, c, alpha, beta, gamma):
    CreatePeaksWorkspace(NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace=name)

    SetUB(Workspace=name, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    np.random.seed(13)

    hkls = np.random.randint(-10, 10, size=(25, 3))

    for hkl in hkls:
        AddPeakHKL(Workspace=name, HKL=hkl)


def test_optimization():
    create_peaks("cubic", 6.1, 5.9, 6.05, 91, 88, 89)

    opt = Optimization("cubic")
    opt.optimize_lattice("Cubic")

    a, b, c, alpha, beta, gamma = opt.get_lattice_parameters()

    assert np.allclose([a, b], c)
    assert np.allclose([alpha, beta, gamma], 90)

    create_peaks("hex", 6.1, 5.9, 8.05, 91, 88, 118)

    opt = Optimization("hex")
    opt.optimize_lattice("Hexagonal")

    a, b, c, alpha, beta, gamma = opt.get_lattice_parameters()

    assert np.allclose(a, b)
    assert np.allclose([alpha, beta], 90)
    assert np.allclose(gamma, 120)

    create_peaks("rhom", 6.1, 5.9, 6.05, 61, 61, 58)

    opt = Optimization("rhom")
    opt.optimize_lattice("Rhombohedral")

    a, b, c, alpha, beta, gamma = opt.get_lattice_parameters()

    assert np.allclose([a, b], c)
    assert np.allclose([alpha, beta], gamma)

    create_peaks("ortho", 6.1, 4.9, 12.05, 89, 91, 88)

    opt = Optimization("ortho")
    opt.optimize_lattice("Orthorhombic")

    a, b, c, alpha, beta, gamma = opt.get_lattice_parameters()

    assert np.allclose([alpha, beta, gamma], 90)

    create_peaks("mono", 6.1, 4.9, 5.05, 91, 109, 90)

    opt = Optimization("mono")
    opt.optimize_lattice("Monoclinic")

    a, b, c, alpha, beta, gamma = opt.get_lattice_parameters()

    assert np.allclose([alpha, gamma], 90)
