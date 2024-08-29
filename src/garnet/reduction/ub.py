from mantid.simpleapi import (SelectCellWithForm,
                              ShowPossibleCells,
                              TransformHKL,
                              CalculatePeaksHKL,
                              IndexPeaks,
                              FindUBUsingFFT,
                              FindUBUsingLatticeParameters,
                              FindUBUsingIndexedPeaks,
                              OptimizeLatticeForCellType,
                              CalculateUMatrix,
                              HasUB,
                              SetUB,
                              LoadIsawUB,
                              SaveIsawUB,
                              CopySample,
                              mtd)

from mantid.geometry import PointGroupFactory

import json

import numpy as np

import scipy.spatial
import scipy.optimize

lattice_group = {'Triclinic': '-1',
                 'Monoclinic': '2/m',
                 'Orthorhombic': 'mmm',
                 'Tetragonal': '4/mmm',
                 'Rhombohedral': '-3m',
                 'Hexagonal': '6/mmm',
                 'Cubic': 'm-3m'}

class UBModel:

    def __init__(self, peaks):
        """
        Tools for working with peaks and UB.

        Parameters
        ----------
        peaks : str
            Table of peaks.

        """

        self.peaks = mtd[peaks]

    def has_UB(self):
        """
        Check if peaks table has a UB determined.

        """

        return HasUB(Workspace=self.peaks)

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def load_UB(self, filename):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        LoadIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def determine_UB_with_niggli_cell(self, min_d, max_d, tol=0.1):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingFFT(PeaksWorkspace=self.peaks,
                       MinD=min_d,
                       MaxD=max_d,
                       Tolerance=tol)

    def determine_UB_with_lattice_parameters(self, a,
                                                   b,
                                                   c,
                                                   alpha,
                                                   beta,
                                                   gamma,
                                                   tol=0.1):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(PeaksWorkspace=self.peaks,
                                     a=a,
                                     b=b,
                                     c=c,
                                     alpha=alpha,
                                     beta=beta,
                                     gamma=gamma,
                                     Tolerance=tol)

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(PeaksWorkspace=self.peaks,
                                Tolerance=tol,
                                ToleranceForSatellite=tol_for_sat)

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.

        +----------------+---------------+----------------------+
        | Lattice system | Lengths       | Angles               |
        +================+===============+======================+
        | Cubic          | :math:`a=b=c` | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Hexagonal      | :math:`a=b`   | :math:`α=β=90,γ=120` |
        +----------------+---------------+----------------------+
        | Rhombohedral   | :math:`a=b=c` | :math:`α=β=γ`        |
        +----------------+---------------+----------------------+
        | Tetragonal     | :math:`a=b`   | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Orthorhombic   | None          | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Monoclinic     | None          | :math:`α=γ=90`       |
        +----------------+---------------+----------------------+
        | Triclinic      | None          | None                 |
        +----------------+---------------+----------------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        OptimizeLatticeForCellType(PeaksWorkspace=self.peaks,
                                   CellType=cell,
                                   Apply=True,
                                   Tolerance=tol)

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        CalculateUMatrix(PeaksWorkspace=self.peaks,
                         a=a,
                         b=b,
                         c=c,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma)

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        SelectCellWithForm(PeaksWorkspace=self.peaks,
                           FormNumber=number,
                           Apply=True,
                           Tolerance=tol)

    def possible_conventional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form results.

        """

        result = ShowPossibleCells(PeaksWorkspace=self.peaks,
                                   MaxScalarError=max_error,
                                   AllowPermutations=permutations,
                                   BestOnly=False)

        vals = [json.loads(cell) for cell in result.Cells]

        return vals

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ','.join(['{},{},{}'.format(*row) for row in transform])

        TransformHKL(PeaksWorkspace=self.peaks,
                     Tolerance=tol,
                     HKLTransform=hkl_trans)

    def generate_lattice_transforms(self, cell):
        """
        Obtain possible transforms compatabile with a unit cell lattice.

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = '{}: '.format(symop.getOrder())+symop.getIdentifier()
                transform[name] = T.tolist()

        return {key:transform[key] for key in sorted(transform.keys())}

    def index_peaks(self, tol=0.1,
                          sat_tol=None,
                          mod_vec_1=[0,0,0],
                          mod_vec_2=[0,0,0],
                          mod_vec_3=[0,0,0],
                          max_order=0,
                          cross_terms=False):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        indexing = IndexPeaks(PeaksWorkspace=self.peaks,
                              Tolerance=tol,
                              ToleranceForSatellite=tol_for_sat,
                              RoundHKLs=True,
                              CommonUBForAll=True,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True)

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """

        CalculatePeaksHKL(PeaksWorkspace=self.peaks,
                          OverWrite=True)

    def copy_UB(self, workspace):
        """
        Copy UB to another workspace.

        Parameters
        ----------
        workspace : float
            Target workspace to copy the UB to.

        """

        CopySample(InputWorkspace=self.peak,
                   OutputWorkspace=workspace,
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)


class Optimization:

    def __init__(self, peaks):
        """
        Optimize lattice and orientation using nonlinear least squares.

        Parameters
        ----------
        peaks : str
            Name of peaks workspace to perform constrained UB optimization.

        """

        Q, hkl = [], []

        for pk in mtd[peaks]:

            hkl.append(pk.getHKL())
            Q.append(pk.getQSampleFrame())

        self.Q, self.hkl = np.array(Q), np.array(hkl)

        self.peaks = peaks

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        if mtd.doesExist(self.peaks):

            ol = mtd[self.peaks].sample().getOrientedLattice()

            a, b, c = ol.a(), ol.b(), ol.c()
            alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

            return a, b, c, alpha, beta, gamma

    def get_orientation_angles(self):
        """
        Current orientation angles.

        Returns
        -------
        phi : float
            Rotation axis azimuthal angle in radians.
        theta : float
            Rotation axis polar angle in radians.
        omega : float
            Rotation angle in radians.

        """

        if mtd.doesExist(self.peaks):

            U = mtd[self.peaks].sample().getOrientedLattice().getU()

            omega = np.arccos((np.trace(U)-1)/2)

            val, vec = np.linalg.eig(U)

            ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

            theta = np.arccos(uz)
            phi = np.arctan2(uy, ux)

            return phi, theta, omega

    def U_matrix(self, phi, theta, omega):

        u0 = np.cos(phi)*np.sin(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(theta)

        w = omega*np.array([u0,u1,u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):

        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                      [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                      [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def cubic(self, x):

        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):

        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):

        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):

        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):

        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):

        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):

        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, hkl, Q, fun):
        """
        Optimization residual function.

        Parameters
        ----------
        x : list
            Parameters.
        hkl : list
            Miller indices.
        Q : list
            Q-sample vectors.
        fun : function
            Lattice constraint function.

        Returns
        -------
        residual : list
            Least squares residuals.

        """

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U,B)

        return (np.einsum('ij,lj->li', UB, hkl)*2*np.pi-Q).flatten()

    def optimize_lattice(self, cell):
        """
        Refine the orientation and lattice parameters under constraints.

        Parameters
        ----------
        cell : str
            Lattice centering to constrain paramters.

        """

        if mtd.doesExist(self.peaks):

            a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

            phi, theta, omega = self.get_orientation_angles()

            fun_dict = {'Cubic': self.cubic,
                        'Rhombohedral': self.rhombohedral,
                        'Tetragonal': self.tetragonal,
                        'Hexagonal': self.hexagonal,
                        'Orthorhombic': self.orthorhombic,
                        'Monoclinic': self.monoclinic,
                        'Triclinic': self.triclinic}

            x0_dict = {'Cubic': (a, ),
                       'Rhombohedral': (a, alpha),
                       'Tetragonal': (a, c),
                       'Hexagonal': (a, c),
                       'Orthorhombic': (a, b, c),
                       'Monoclinic': (a, b, c, beta),
                       'Triclinic': (a, b, c, alpha, beta, gamma)}

            fun = fun_dict[cell]
            x0 = x0_dict[cell]

            x0 += (phi, theta, omega)
            args = (self.hkl, self.Q, fun)

            sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

            a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

            B = self.B_matrix(a, b, c, alpha, beta, gamma)
            U = self.U_matrix(phi, theta, omega)

            UB = np.dot(U, B)

            J = sol.jac
            cov = np.linalg.inv(J.T.dot(J))

            chi2dof = np.sum(sol.fun**2)/(sol.fun.size-sol.x.size)
            cov *= chi2dof

            sig = np.sqrt(np.diagonal(cov))

            sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

            if np.isclose(a, sig_a):
                sig_a = 0
            if np.isclose(b, sig_b):
                sig_b = 0
            if np.isclose(c, sig_c):
                sig_c = 0

            if np.isclose(alpha, sig_alpha):
                sig_alpha = 0
            if np.isclose(beta, sig_beta):
                sig_beta = 0
            if np.isclose(gamma, sig_gamma):
                sig_gamma = 0

            SetUB(Workspace=self.peaks, UB=UB)

            mtd[self.peaks].sample().getOrientedLattice().setError(sig_a,
                                                                   sig_b,
                                                                   sig_c,
                                                                   sig_alpha,
                                                                   sig_beta,
                                                                   sig_gamma)
