#
# Description: Magnetic interactions in the spin-separated X2C formalism
#
# OriginalTheory: JCP 141, 054111 (2014)
#
# Implementation: Eqs (181) and (182) for uniform magnetic field.
#
# Implemented by Zhendong Li (zhendongli2008@gmail.com)
#
import numpy
import scipy.linalg
import sfX2C_soDKH1 

debug = False

# Tsf = 1/2 B.l => 1/2 lpq 
# Tsd - 1/2 B.sigma => 1/2Spq
def get_Tsfsd(mol):
   nb = mol.nao_nr()
   tsf = numpy.empty((3,nb,nb))
   tsf = -0.5*mol.intor('int1e_cg_irxp', comp=3) # -1/2 rm \times dm
   tsd = numpy.zeros((3,3,nb,nb))
   s = mol.intor_symmetric('int1e_ovlp')
   for ic in range(3):
      tsd[ic,ic] = 0.5*s
   if debug:
      print 'debug ova'
      print s[:,0]
      print
      print 'debug tsf,tsd'
      for ic in range(3):
         print 'ic=',ic,numpy.linalg.norm(tsf[ic]),numpy.sum(tsf[ic])
	 print tsf[ic][:,0]
      print 
      for ic in range(3):
       for jc in range(3):
         print 'ic=',ic,jc,numpy.linalg.norm(tsd[ic,jc]),numpy.sum(tsd[ic,jc])
	 print tsd[ic,jc][:3,0]
	 print tsd[ic,jc][0,:3]
      print 
   return tsf,tsd

def get_Wsfsd(mol):
   nb = mol.nao_nr()
   # Correct? definition: <mu|AVp|nu>
   ints = mol.intor('int1e_cg_sa10nucsp_sph', comp=12).reshape(3,4,nb,nb) 
   wsf = (ints - ints.transpose(0,1,3,2))[:,3]   
   wsd = -(ints.transpose(1,0,3,2) + ints.transpose(1,0,2,3))[:,0:3] 
   if debug:
      print 'debug wsf,wsd'
      for ic in range(3):
         print 'ic=',ic,numpy.linalg.norm(wsf[ic]),numpy.sum(wsf[ic])
	 print wsf[ic][:,0]
      print 
      for ic in range(3):
       for jc in range(3):
         print 'ic=',ic,jc,numpy.linalg.norm(wsd[ic,jc]),numpy.sum(wsd[ic,jc])
	 print wsd[ic,jc][:3,0]
	 print wsd[ic,jc][0,:3]
      print 
   return wsf,wsd

def get_mag(a4,sinv,x,rp,h1e,tsf,wsf,sgn):
   tmp1 = reduce(numpy.dot,(rp.T,
	         tsf.dot(x)+x.T.dot(tsf)+\
	         (x.T.dot(a4*wsf-tsf)).dot(x),rp))
   tmp2 = reduce(numpy.dot,(h1e,sinv,rp.T,x.T,tsf,x,rp))
   hmat = tmp1 - a4*(tmp2+sgn*tmp2.T)
   return hmat

#
# Convention:
#  h10[p,q] = i*Bm*Lm[p,q]
#  h11[p,q] = Bm*Gmn[p,q]*Sn
# 
def get_zeeman(myhf,mol,c,org):
   xmol,contr_coeff = myhf.with_x2c.get_xmol(mol)
   xmol.set_common_origin(org)
   print 'get_zeeman (np,nc)=',contr_coeff.shape
   nb = contr_coeff.shape[0]
   nc = contr_coeff.shape[1]
   t = xmol.intor_symmetric('int1e_kin')
   v = xmol.intor_symmetric('int1e_nuc')
   s = xmol.intor_symmetric('int1e_ovlp')
   w = xmol.intor_symmetric('int1e_pnucp')
   x,rp,h1e = sfX2C_soDKH1.sfx2c1e(t, v, w, s, c)
   sinv = scipy.linalg.pinv(s)
   # compute integrals
   tsf,tsd = get_Tsfsd(xmol)
   wsf,wsd = get_Wsfsd(xmol)
   a4 = 0.25/c**2
   h10 = numpy.zeros((3,nb,nb))
   h11 = numpy.zeros((3,3,nb,nb))
   if debug: print 'debug h01,h11'
   for ic in range(3):
      h10[ic] = get_mag(a4,sinv,x,rp,h1e,tsf[ic],wsf[ic],-1.0) # antisymmetric
      if debug: 
	 print 'ic=',ic,numpy.linalg.norm(h10[ic]),numpy.sum(h10[ic])
	 print h10[ic][:,0]
         if ic == 2: print 
   for ic in range(3):
    for jc in range(3):
      h11[ic,jc] = get_mag(a4,sinv,x,rp,h1e,tsd[ic,jc],wsd[ic,jc],1.0)
      if debug: 
	 print 'ic=',ic,jc,numpy.linalg.norm(h11[ic,jc]),numpy.sum(h10[ic])
	 print h11[ic,jc][:3,0]
	 print h11[ic,jc][0,:3]
   # Contractioin at the last step
   h10_contr = numpy.zeros((3,nc,nc))
   h11_contr = numpy.zeros((3,3,nc,nc))
   for ic in range(3):
      h10_contr[ic] = reduce(numpy.dot,(contr_coeff.T,h10[ic],contr_coeff))
   for ic in range(3):
    for jc in range(3):
      h11_contr[ic,jc] = reduce(numpy.dot,(contr_coeff.T,h11[ic,jc],contr_coeff))
   return h10_contr,h11_contr
