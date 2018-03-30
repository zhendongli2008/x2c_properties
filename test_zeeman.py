#
# ZL: To completely agree with my previous implementation in BDF package:
# 1. Modify exp_drop=0.0 in scf/x2c.py 
#    def _uncontract_mol(mol, xuncontract=False, exp_drop=0.0):
# 2. for larger basis like cc-pvdz, in x2c.py 
#    410 # part2: binding the pGTOs of small exps to the pGTOs of large coefficients
#    there are some special treatments in PYSCF, which do not fully uncontract basis.
# See:
#   xmol,contr_coeff = myhf.with_x2c.get_xmol(mol)
#   print xmol._bas
#   print xmol._env
#   for i in range(10):
#      print i,xmol.bas_exp(i),xmol.bas_ctr_coeff(i)
# 0 [ 6665.] [[ 1.]]
# 1 [ 1000.] [[ 1.]]
# 2 [ 228.] [[ 1.]]
# 3 [ 64.71] [[ 1.]]
# 4 [ 21.06] [[ 1.]]
# 5 [ 2.797] [[ 1.]]
# 6 [ 7.495   0.5215] [[ 0.9879071  -0.29277498]
# 		 [ 0.03348494  1.0629644 ]]
# 7 [ 0.1596] [[ 1.]]
# 8 [ 9.439] [[ 1.]]
# 9 [ 2.002] [[ 1.]]
#
import scipy.linalg
import numpy
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci

mol = gto.M(
    atom = 'C 0 0 0', # \n N 0 0 1.1718',
    basis = 'cc-pvdz', #cc-pvtz',
    verbose = 4,
    charge = 0,
    symmetry = 1,
    spin = 0)

from pyscf import lib
lib.param.LIGHT_SPEED = 137.0359895000

myhf = scf.sfx2c(scf.RHF(mol))
myhf.kernel()

c = lib.param.LIGHT_SPEED 
# set gauge origin
z = mol.atom_charges()
r = mol.atom_coords()
org = numpy.einsum('z,zx->x', z, r) / z.sum()
mol.set_common_origin(org)

#
# Convention:
#  h10[p,q] = i*Bm*Lm[p,q]
#  h11[p,q] = Bm*Gmn[p,q]*Sn
# 
import sfX2C_zeeman
h10,h11 = sfX2C_zeeman.get_zeeman(myhf,mol,c,org)
print h10.shape,numpy.linalg.norm(h10) 
print h11.shape,numpy.linalg.norm(h11)

for i in range(3):
   print i,numpy.linalg.norm(h10[i]+h10[i].T)

for i in range(3):
   for j in range(3):
      print i,numpy.linalg.norm(h11[i,j]-h11[i,j].T)
