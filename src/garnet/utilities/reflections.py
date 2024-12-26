from mantid.simpleapi import (LoadNexus, 
                              IndexPeaks,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              SetGoniometer,
                              SetSample,
                              mtd)

import matplotlib.pyplot as plt
import numpy as np

import scipy.optimize
from lmfit import Minimizer, Parameters

from mantid.geometry import PointGroupFactory
from mantid import config
config['Q.convention'] = 'Crystallography'

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

filename = '/SNS/CORELLI/IPTS-31429/shared/Fe2Mo3O8/integration_base/Fe2Mo3O8_integration/Fe2Mo3O8_Hexagonal_P_d(min)=0.70_r(max)=0.20.nxs'

LoadNexus(Filename=filename,
          OutputWorkspace='peaks')

for i, peak in enumerate(mtd['peaks']):
    row = peak.getRow()
    col = peak.getCol()

    if row <= row_lim[0] or row >= row_lim[1] or col <= col_lim[0] or col >= col_lim[1]:
        peak.setIntensity(peak.getIntensity())
        peak.setSigmaIntensity(peak.getIntensity())

    shape = eval(peak.getPeakShape().toJSON())

    if shape['shape'] == 'none':
        peak.setIntensity(peak.getIntensity())
        peak.setSigmaIntensity(peak.getIntensity())

    elif shape['radius0'] == 0 or shape['radius1'] == 0 or shape['radius2'] == 0:
        peak.setIntensity(peak.getIntensity())
        peak.setSigmaIntensity(peak.getIntensity())

FilterPeaks(InputWorkspace='peaks',
            OutputWorkspace='peaks',
            FilterVariable='Signal/Noise',
            FilterValue=5,
            Operator='>')

IndexPeaks(PeaksWorkspace='peaks',
           RoundHKLs=False,
           Tolerance=0.1)

FilterPeaks(InputWorkspace='peaks',
            OutputWorkspace='peaks',
            FilterVariable='h^2+k^2+l^2',
            FilterValue=0,
            Operator='>')

SortPeaksWorkspace(InputWorkspace='peaks',
                   OutputWorkspace='peaks',
                   ColumnNameToSortBy='RunNumber',
                   SortAscending=False)

SortPeaksWorkspace(InputWorkspace='peaks',
                   OutputWorkspace='peaks',
                   ColumnNameToSortBy='Intens',
                   SortAscending=False)

def AbsorptionCorrection(ws,
                          chemical_formula,
                          z_parameter,
                          u_vector,
                          v_vector,
                          params=[0.1,0.2,0.4],
                          shape='plate'):

    if shape == 'sphere':
        assert type(params) is float
    elif shape == 'cylinder':
        assert len(params) == 2
    else:
        assert len(params) == 3

    UB = mtd[ws].sample().getOrientedLattice().getUB().copy()

    u = np.dot(UB, u_vector)
    v = np.dot(UB, v_vector)

    u /= np.linalg.norm(u)

    w = np.cross(u, v)
    w /= np.linalg.norm(w)

    v = np.cross(w, u)

    T = np.column_stack([v, w, u])

    mtd[ws].run().getGoniometer().setR(T)
    gamma, beta, alpha = mtd[ws].run().getGoniometer().getEulerAngles('ZYX')

    if shape == 'sphere':
        shape = ' \
        <sphere id="sphere"> \
        <radius val="{}" /> \
        <centre x="0.0" y="0.0" z="0.0" /> \
        <rotate x="{}" y="{}" z="{}" /> \
        </sphere> \
        '.format(params/200, alpha, beta, gamma)
    elif shape == 'cylinder':
        shape = ' \
        <cylinder id="cylinder"> \
        <centre-of-bottom-base r="0.0" t="0.0" p="0.0" />   \
        <axis x="0.0" y="1.0" z="0" /> \
        <radius val="{}" />  \
        <height val="{}" />  \
        <rotate x="{}" y="{}" z="{}" /> \
        </cylinder> \
      '.format(params[0]/200, params[1]/100, alpha, beta, gamma)
    else:
        shape = ' \
        <cuboid id="cuboid"> \
        <width val="{}" /> \
        <height val="{}"  /> \
        <depth val="{}" /> \
        <centre x="0.0" y="0.0" z="0.0"  /> \
        <rotate x="{}" y="{}" z="{}" /> \
        </cuboid> \
        <algebra val="cuboid" /> \
      '.format(params[0]/100, params[1]/100, params[2]/100, alpha, beta, gamma)

    shape_dict = {'Shape': 'CSG', 'Value': shape}

    Rs = [peak.getGoniometerMatrix() for peak in mtd[ws]]
    matrix_dict = {}

    runs = []
    for peak in mtd[ws]:
        R = peak.getGoniometerMatrix()

        matrix_tuple = tuple(R.flatten())

        if matrix_tuple in matrix_dict:
            run = matrix_dict[matrix_tuple]
        else:
            ind = np.isclose(Rs, R).all(axis=(1,2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)
            run = i + 1
            matrix_dict[matrix_tuple] = run

        runs.append(run)
        peak.setRunNumber(run)

    runs = np.unique(runs).astype(int).tolist()

    mat_dict = {'ChemicalFormula': chemical_formula,
                'ZParameter': float(z_parameter),
                'UnitCellVolume': mtd[ws].sample().getOrientedLattice().volume()}

    for i, run in enumerate(runs):

        FilterPeaks(InputWorkspace=ws,
                  FilterVariable='RunNumber',
                  FilterValue=run,
                  Operator='=',
                  OutputWorkspace='tmp')

        R = mtd['tmp'].getPeak(0).getGoniometerMatrix()

        mtd['tmp'].run().getGoniometer().setR(R)
        omega, chi, phi = mtd['tmp'].run().getGoniometer().getEulerAngles('YZY')

        SetGoniometer(Workspace='tmp',
                    Axis0='{},0,1,0,1'.format(omega),
                    Axis1='{},0,0,1,1'.format(chi),
                    Axis2='{},0,1,0,1'.format(phi))

        SetSample(InputWorkspace='tmp',
                  Geometry=shape_dict,
                  Material=mat_dict)

        AddAbsorptionWeightedPathLengths(InputWorkspace='tmp',
                                         ApplyCorrection=False)

        if i == 0:
          CloneWorkspace(InputWorkspace='tmp',
                         OutputWorkspace=ws+'_corr')
        else:
          CombinePeaksWorkspaces(LHSWorkspace=ws+'_corr',
                                 RHSWorkspace='tmp',
                                 OutputWorkspace=ws+'_corr')

    mat = mtd['tmp'].sample().getMaterial()

    sigma_a = mat.absorbXSection()
    sigma_s = mat.totalScatterXSection()

    M = mat.relativeMolecularMass()
    n = mat.numberDensityEffective # A^-3
    N = mat.totalAtoms

    V = np.abs(mtd['tmp'].sample().getShape().volume()*100**3) # cm^3

    rho = (n/N)/0.6022*M
    m = rho*V*1000 # mg
    r = np.cbrt(0.75*np.pi*V)

    mu_s = n*sigma_s
    mu_a = n*sigma_a

    mu = mat.numberDensityEffective*(mat.totalScatterXSection()+mat.absorbXSection(1.8))

    print('{}\n'.format(chemical_formula))
    print('absoption cross section: {:.4f} barn\n'.format(sigma_a))
    print('scattering cross section: {:.4f} barn\n'.format(sigma_s))

    print('linear absorption coefficient: {:.4f} 1/cm\n'.format(mu_a))
    print('linear scattering coefficient: {:.4f} 1/cm\n'.format(mu_s))
    print('absorption parameter: {:.4f} \n'.format(mu*r))

    print('total atoms: {:.4f}\n'.format(N))
    print('molar mass: {:.4f} g/mol\n'.format(M))
    print('number density: {:.4f} 1/A^3\n'.format(n))

    print('mass density: {:.4f} g/cm^3\n'.format(rho))
    print('volume: {:.4f} cm^3\n'.format(V))
    print('mass: {:.4f} mg\n'.format(m))

    for peak in mtd[ws+'_corr']:

        wl = peak.getWavelength()
        tbar = peak.getAbsorptionWeightedPathLength()

        mu = mat.numberDensityEffective*(mat.totalScatterXSection()+mat.absorbXSection(wl))

        corr = np.exp(mu*tbar)

        peak.setIntensity(peak.getIntensity()*corr)
        peak.setSigmaIntensity(peak.getSigmaIntensity()*corr)

#         hkl = np.eye(3)
#         Q = np.matmul(UB, hkl)
# 
#         reciprocal_lattice = np.matmul(R, Q)
# 
#         shape = mtd['tmp'].sample().getShape()
#         mesh = shape.getMesh()*100
# 
#         mesh_polygon = Poly3DCollection(mesh, edgecolors='k', facecolors='w', alpha=0.5, linewidths=1)
# 
#         fig, ax = plt.subplots(subplot_kw={'projection': 'mantid3d', 'proj_type': 'persp'})
#         ax.add_collection3d(mesh_polygon)
# 
#         ax.set_title('Run #{}'.format(run))
#         ax.set_xlabel('x [cm]')
#         ax.set_ylabel('y [cm]')
#         ax.set_zlabel('z [cm]')
# 
#         ax.set_mesh_axes_equal(mesh)
#         ax.set_box_aspect((1,1,1))
# 
#         colors = ['r', 'g', 'b']
#         origin = (ax.get_xlim3d()[1], ax.get_ylim3d()[1], ax.get_zlim3d()[1])
#         origin = (0, 0, 0)
#         lims = ax.get_xlim3d()
#         factor = (lims[1]-lims[0])/3
# 
#         for j in range(3):
#             vector = reciprocal_lattice[:,j]
#             vector_norm = vector/np.linalg.norm(vector)
#             ax.quiver(origin[0], origin[1], origin[2],
#                       vector_norm[0], vector_norm[1], vector_norm[2],
#                       length=factor,
#                       color=colors[j],
#                       linestyle='-')
# 
#             ax.view_init(vertical_axis='y', elev=27, azim=50)
#             fig.show()

AbsorptionCorrection('peaks', 'Yb3 Al5 O12', 8, [1,0,0], [0,1,0], 0.3, 'sphere')

# CloneWorkspace(InputWorkspace='peaks',
#                OutputWorkspace='peaks_corr')

StatisticsOfPeaksWorkspace(InputWorkspace='peaks_corr',
                           PointGroup=point_group,
                           OutputWorkspace='stats',
                           EquivalentIntensities='Median',
                           SigmaCritical=3,
                           WeightedZScore=False)

scale = 1
if mtd['stats'].getNumberPeaks() > 1:
    I_max = max(mtd['stats'].column('Intens'))
    if I_max > 0:
        scale = 1e4/I_max

_, indices = np.unique(mtd['stats'].column(0), return_inverse=True)

for i, peak in zip(indices.tolist(), mtd['stats']):
    peak.setIntensity(scale*peak.getIntensity())
    peak.setSigmaIntensity(scale*peak.getSigmaIntensity())
    peak.setRunNumber(1)

FilterPeaks(InputWorkspace='stats',
            OutputWorkspace='stats',
            FilterVariable='Signal/Noise',
            FilterValue=3,
            Operator='>')