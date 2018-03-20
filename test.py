import scipy.linalg
import numpy
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci

mol = gto.M(
    atom = 'I 0 0 0',
    basis = 'sto-3g',
    verbose=4,
    symmetry=1,
    spin = 1)

from pyscf import lib
lib.param.LIGHT_SPEED = 137.0359895000

#myhf = scf.RHF(mol)
myhf = scf.sfx2c(scf.RHF(mol))
myhf.irrep_nelec = {'A1g' : (7,7), 'A1u' :(4,3), 'E1ux' : (4,4), 'E1uy' : (4,4), 'E2gx' : (2,2), 'E2gy' :(2,2), 'E1gx':(2,2), 'E1gy':(2,2)}
myhf.kernel()

c = lib.param.LIGHT_SPEED 
# Modify exp_drop=0.0 in scf/x2c.py 
# def _uncontract_mol(mol, xuncontract=False, exp_drop=0.0):

import sfX2C_soDKH1 
VsoDKH1_a = sfX2C_soDKH1.get_soDKH1_somf(myhf,mol,c,iop='bp',debug=False) #True)
VsoDKH1_b = sfX2C_soDKH1.get_soDKH1_somf(myhf,mol,c,iop='x2c',debug=False) #True)
print numpy.linalg.norm(VsoDKH1_a-VsoDKH1_b)
