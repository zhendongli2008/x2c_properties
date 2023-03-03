#
# Description: The sfX2C_soDKH1 Hamiltonian
#
# OriginalTheory: JCP 137, 154114 (2012).
#		  JCP 141, 054111 (2014)
#
# Implementation: Mol. Phys. 111 (24), 3741-3755
#		  see Eqs(34)-(53) for details of formulae.
#
# Implemented by Zhendong Li (zhendongli2008@gmail.com)
#
# Functions:
#
# def inv12(s):
# def sfx2c1e(t, v, w, s, c):
# def get_p(dm,x,rp):
# def get_wso(mol):
# def get_kint(mol):
# def get_hso1e(wso,x,rp):
# def get_fso2e(kint,x,rp,pLL,pLS,pSS):
# def get_soDKH1_somf(myhf,mol,c,iop='x2c',debug=False):
# 
import numpy
import scipy.linalg

def inv12(s):
    e,v = scipy.linalg.eigh(s)
    return reduce(numpy.dot,(v,numpy.diag(1/numpy.sqrt(e)),v.T))

def sfx2c1e(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao,nao:]
    cs = a[nao:,nao:]
    x = cs.dot(cl.T.dot(numpy.linalg.inv(cl.dot(cl.T)))) 
    stilde = s + x.T.dot(m[nao:,nao:].dot(x))
    sih = inv12(s)
    sh = numpy.linalg.inv(inv12(s))
    rp = sih.dot(inv12(sih.dot(stilde.dot(sih))).dot(sh))
    l1e = h[:nao,:nao] \
 	+ h[:nao,nao:].dot(x) \
	+ x.T.dot(h[nao:,:nao]) \
	+ x.T.dot(h[nao:,nao:].dot(x))
    h1e = rp.T.dot(l1e.dot(rp)) 
    return x,rp,h1e

def get_p(dm,x,rp):
   pLL = rp.dot(dm.dot(rp.T))
   pLS = pLL.dot(x.T)
   pSS = x.dot(pLL.dot(x.T))
   return pLL,pLS,pSS

def get_wso(mol):
   nb = mol.nao_nr()
   wso = numpy.zeros((3,nb,nb))
   for iatom in range(mol.natm):
      zA  = mol.atom_charge(iatom)
      xyz = mol.atom_coord(iatom)
      mol.set_rinv_orig(xyz)
      wso += -zA*mol.intor('cint1e_prinvxp_sph', 3) # sign due to integration by part
   return wso

def get_kint(mol):
   nb = mol.nao_nr() 
   np = nb*nb
   nq = np*np
   ddint = mol.intor('int2e_ip1ip2_sph',9).reshape(3,3,nq)
   kint = numpy.zeros((3,nq))
   kint[0] = ddint[1,2]-ddint[2,1]# x = yz - zy
   kint[1] = ddint[2,0]-ddint[0,2]# y = zx - xz
   kint[2] = ddint[0,1]-ddint[1,0]# z = xy - yx
   return kint.reshape(3,nb,nb,nb,nb)

def get_hso1e(wso,x,rp):
   nb = x.shape[0]
   hso1e = numpy.zeros((3,nb,nb))
   for ic in range(3):
      hso1e[ic] = reduce(numpy.dot,(rp.T,x.T,wso[ic],x,rp))
   return hso1e

def get_fso2e(kint,x,rp,pLL,pLS,pSS):
   nb = x.shape[0]
   fso2e = numpy.zeros((3,nb,nb))
   for ic in range(3):
      gsoLL = -2.0*numpy.einsum('lmkn,lk->mn',kint[ic],pSS)
      gsoLS = -numpy.einsum('mlkn,lk->mn',kint[ic],pLS) \
      	      -numpy.einsum('lmkn,lk->mn',kint[ic],pLS) 
      gsoSS = -2.0*numpy.einsum('mnkl,lk',kint[ic],pLL) \
   	      -2.0*numpy.einsum('mnlk,lk',kint[ic],pLL) \
   	      +2.0*numpy.einsum('mlnk,lk',kint[ic],pLL)
      fso2e[ic] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
   	     + x.T.dot(gsoSS.dot(x))
      fso2e[ic] = reduce(numpy.dot,(rp.T,fso2e[ic],rp))
   return fso2e

def get_soDKH1_somf(myhf,mol,c,iop='x2c',debug=False):
   xmol,contr_coeff = myhf.with_x2c.get_xmol(mol)
   print 'get_soDKH1_somf with iop=',iop,' (np,nc)=',contr_coeff.shape
   nb = contr_coeff.shape[0]
   nc = contr_coeff.shape[1]
   if iop == 'x2c': 
      t = xmol.intor_symmetric('int1e_kin')
      v = xmol.intor_symmetric('int1e_nuc')
      s = xmol.intor_symmetric('int1e_ovlp')
      w = xmol.intor_symmetric('int1e_pnucp')
      x,rp,h1e = sfx2c1e(t, v, w, s, c)
   elif iop == 'bp':
      x = numpy.identity(nb)
      rp = numpy.identity(nb)
   dm = myhf.make_rdm1()
   # Spin-Averaged for ROHF or UHF
   if len(dm.shape)==3: dm = (dm[0]+dm[1])/2.0
   dm = reduce(numpy.dot,(contr_coeff,dm,contr_coeff.T))
   pLL,pLS,pSS = get_p(dm,x,rp)
   wso = get_wso(xmol)
   kint = get_kint(xmol)
   if debug:
      for ic in range(3):
         print 'ic=',ic,numpy.linalg.norm(kint[ic]+kint[ic].transpose(2,3,0,1))
   hso1e = get_hso1e(wso,x,rp)
   fso2e = get_fso2e(kint,x,rp,pLL,pLS,pSS)
   a4 = 0.25/c**2
   VsoDKH1 = a4*(hso1e + fso2e)
   if debug:
      for ic in range(3):
	 tmp = hso1e[ic]
         print ic,'hso1e',numpy.linalg.norm(tmp),\
	          	  numpy.linalg.norm(tmp+tmp.T)
	 tmp = fso2e[ic]
         print ic,'fso2e',numpy.linalg.norm(tmp),\
	          	  numpy.linalg.norm(tmp+tmp.T)
	 tmp = hso1e[ic]+fso2e[ic]
         print ic,'vso2e',numpy.linalg.norm(tmp),\
	          	  numpy.linalg.norm(tmp+tmp.T)
   # Contractioin at the last step
   VsoDKH1contr = numpy.zeros((3,nc,nc))
   for ic in range(3):
      VsoDKH1contr[ic] = reduce(numpy.dot,(contr_coeff.T,VsoDKH1[ic],contr_coeff))
   if debug:
      for ic in range(3):
	 tmp = VsoDKH1contr[ic]
         print ic,numpy.linalg.norm(tmp),\
	          numpy.linalg.norm(tmp+tmp.T)
   return VsoDKH1contr
