import numpy as np
import scipy 
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd
import copy as cp

from sys import getsizeof
import h5py
import pyscf
from pyscf import lib

from functools import reduce

import openfermion 




class SQ_Hamiltonian:
    """
    General hamiltonian operator: 
    H = H(pq){p'q} + V(pqrs){p'q'sr} + A(p){p'} + B(p){p} + C(pq){p'q'} + D(pq){pq} + ...
    """
    def __init__(self):
        # operator integrals tensors
        self.int_H = np.array(())
        self.int_V = np.array(())
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())

        # MO basis : really, just any transformation from AO->Current basis such that C'SC = I
        self.C = np.array(())
                            
        # AO overlap        
        self.S = np.array(())

        #number of spatial orbitals
        self.n_orb = 0

    def init(self, h, v, C, S):
        # molecule is a pyscf molecule object from gto.Mole()
        
        self.S = cp.deepcopy(S) 
        self.C = cp.deepcopy(C) 
        self.n_orb = self.C.shape[1] // 2
        
        self.int_H = cp.deepcopy(h) 
        self.int_V = cp.deepcopy(v) 

    def transform_orbitals(self, U):
        """
        A(p,q)U(q,s) with s is new basis
        """
        assert(U.shape[0] == self.C.shape[1])

        new_dim = U.shape[1]

        print(" Rotate orbitals")
        self.int_H = np.einsum("pq,pr->rq",self.int_H,U)
        self.int_H = np.einsum("rq,qs->rs",self.int_H,U)
        self.int_V = np.einsum("pqrs,pl->lqrs",self.int_V,U)
        self.int_V = np.einsum("lqrs,qm->lmrs",self.int_V,U)
        self.int_V = np.einsum("lmrs,rn->lmns",self.int_V,U)
        self.int_V = np.einsum("lmns,so->lmno",self.int_V,U)

        if self.int_A.size > 0: 
            self.int_A = np.einsum(' r,rs->s', self.int_A,U) 
        if self.int_B.size > 0: 
            self.int_B = np.einsum(' r,rs->s', self.int_B,U) 
        if self.int_C.size > 0: 
            self.int_C = np.einsum(' pq,qr,rs->ps', U.T,self.int_C,U) 
        if self.int_D.size > 0: 
            self.int_D = np.einsum(' pq,qr,rs->ps', U.T,self.int_D,U) 

        
        self.C = self.C.dot(U)
        self.n_orb = self.C.shape[1] // 2


    def energy_of_determinant(self, config_a, config_b):
        """ This only returns electronic energy"""
        e1 = 0
        e2 = 0
        for i in config_a:
            e1 += self.int_H[2*i,2*i]
        for i in config_b:
            e1 += self.int_H[2*i+1,2*i+1]
        for i in config_a:
            for j in config_a:
                if i>=j:
                    continue
                e2 += self.int_V[2*i,2*i,2*j,2*j]
                e2 -= self.int_V[2*i,2*j,2*j,2*i]
        for i in config_b:             
            for j in config_b:         
                if i>=j:                
                    continue           
                e2 += self.int_V[2*i+1,2*i+1,2*j+1,2*j+1]
                e2 -= self.int_V[2*i+1,2*j+1,2*j+1,2*i+1]
        for i in config_a:             
            for j in config_b:         
                e2 += self.int_V[2*i,2*i,2*j+1,2*j+1]
        print("One-body = %16.10f"%e1)
        print("Two-body = %16.10f"%e2)
        e = e1+e2
        return e


    def extract_local_hamiltonian(self,orb_subset):
        """ Extract local Hamiltonian acting only on subset of orbitals """
        assert(orb_subset <= self.n_orb)
        
        H = SQ_Hamiltonian()
        H.C = self.C[:,0:2*orb_subset]
        H.n_orb = H.C.shape[1] // 2
        H.int_H = self.int_H[:,0:2*orb_subset][0:2*orb_subset]
        H.int_V = self.int_V[:,:,:,0:2*orb_subset][:,:,0:2*orb_subset][:,0:2*orb_subset][0:2*orb_subset]

        # Note: this doesn't pass other terms - can't see why it'd need to

        return H

    def export_FermionOperator(self, shift=0):
        #Integral objects are already in spin-orbital form (chemist's notation)
        fermi_op = openfermion.FermionOperator()

        #H
        for p in range(0,self.n_orb):
            p_a = 2*p
            p_b = 2*p+1
            for q in range(0,self.n_orb):
                q_a = 2*q
                q_b = 2*q+1
                #aa
                fermi_op += openfermion.FermionOperator(((p_a+shift,1),(q_a+shift,0)), self.int_H[p_a,q_a])
                #bb
                fermi_op += openfermion.FermionOperator(((p_b+shift,1),(q_b+shift,0)), self.int_H[p_b,q_b])

        #V
        for p in range(0,self.n_orb):
            p_a = 2*p
            p_b = 2*p+1
            for q in range(0,self.n_orb):
                q_a = 2*q
                q_b = 2*q+1
                for r in range(0,self.n_orb):
                    r_a = 2*r
                    r_b = 2*r+1
                    for s in range(0,self.n_orb):
                        s_a = 2*s
                        s_b = 2*s+1
                        #aaaa
                        fermi_op += 0.5 * openfermion.FermionOperator(((p_a+shift,1),
                                    (q_a+shift,1),(s_a+shift,0),(r_a+shift,0)),self.int_V[p_a,r_a,q_a,s_a])
                        #aabb
                        fermi_op += 0.5 * openfermion.FermionOperator(((p_a+shift,1),
                                    (q_b+shift,1),(s_b+shift,0),(r_a+shift,0)),self.int_V[p_a,r_a,q_b,s_b])
                        #bbaa
                        fermi_op += 0.5 * openfermion.FermionOperator(((p_b+shift,1),
                                    (q_a+shift,1),(s_a+shift,0),(r_b+shift,0)),self.int_V[p_b,r_b,q_a,s_a])
                        #bbbb
                        fermi_op += 0.5 * openfermion.FermionOperator(((p_b+shift,1),
                                    (q_b+shift,1),(s_b+shift,0),(r_b+shift,0)),self.int_V[p_b,r_b,q_b,s_b])

        #A
        if self.int_A.size > 0 :
            for p in range(0,self.n_orb):
                p_a = 2*p
                p_b = 2*p+1
                fermi_op += openfermion.FermionOperator(((p_a+shift,1)), self.int_A[p_a])
                fermi_op += openfermion.FermionOperator(((p_b+shift,1)), self.int_A[p_b])

        #B
        if self.int_B.size > 0 :
            for p in range(0,self.n_orb):
                p_a = 2*p
                p_b = 2*p+1
                fermi_op += openfermion.FermionOperator(((p_a+shift,0)), self.int_B[p_a])
                fermi_op += openfermion.FermionOperator(((p_b+shift,0)), self.int_B[p_b])

        #C
        if self.int_C.size > 0 :
            for p in range(0,self.n_orb):
                p_a = 2*p
                p_b = 2*p+1
                for q in range(0,self.n_orb):
                    q_a = 2*q
                    q_b = 2*q+1
                    #aa
                    fermi_op += openfermion.FermionOperator(((p_a+shift,1),(q_a+shift,1)), self.int_C[p_a,q_a])
                    #ab
                    fermi_op += openfermion.FermionOperator(((p_a+shift,1),(q_b+shift,1)), self.int_C[p_a,q_b])
                    #ba
                    fermi_op += openfermion.FermionOperator(((p_b+shift,1),(q_a+shift,1)), self.int_C[p_b,q_a])
                    #bb
                    fermi_op += openfermion.FermionOperator(((p_b+shift,1),(q_b+shift,1)), self.int_C[p_b,q_b])

        #D
        if self.int_D.size > 0 :
            for p in range(0,self.n_orb):
                p_a = 2*p
                p_b = 2*p+1
                for q in range(0,self.n_orb):
                    q_a = 2*q
                    q_b = 2*q+1
                    #aa
                    fermi_op += openfermion.FermionOperator(((p_a+shift,0),(q_a+shift,0)), self.int_D[p_a,q_a])
                    #ab
                    fermi_op += openfermion.FermionOperator(((p_a+shift,0),(q_b+shift,0)), self.int_D[p_a,q_b])
                    #ba
                    fermi_op += openfermion.FermionOperator(((p_b+shift,0),(q_a+shift,0)), self.int_D[p_b,q_a])
                    #bb
                    fermi_op += openfermion.FermionOperator(((p_b+shift,0),(q_b+shift,0)), self.int_D[p_b,q_b])

        return fermi_op


        comment = """
        #We have spatial orbital integrals, so we need to convert back to spin orbitals
        
        fermi_op = openfermion.FermionOperator()

        #H
        for p in range(self.int_H.shape[0]):
            pa = 2*p + shift
            pb = 2*p+1 +  shift
            for q in range(self.int_H.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q]) 
        
        #V
        for p in range(self.int_V.shape[0]):
            pa = 2*p +shift
            pb = 2*p+1 +shift
            for q in range(self.int_V.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                for r in range(self.int_V.shape[2]):
                    ra = 2*r +shift
                    rb = 2*r+1 +shift
                    for s in range(self.int_V.shape[3]):
                        sa = 2*s +shift
                        sb = 2*s+1 +shift
                        #aa
                        fermi_op += .5* openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ab
                        fermi_op += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ba
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s]) 
                        #bb
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s]) 

        #A
        for p in range(self.int_A.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,1)), self.int_A[p]) 
            fermi_op += openfermion.FermionOperator(((pb,1)), self.int_A[p]) 
    
        #B
        for p in range(self.int_B.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,0)), self.int_B[p]) 
            fermi_op += openfermion.FermionOperator(((pb,0)), self.int_B[p]) 
        
        #C
        for p in range(self.int_C.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_C.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,1),(qb,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,1)), self.int_C[p,q]) 
        
        #D
        for p in range(self.int_D.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_D.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,0),(qb,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qb,0)), self.int_D[p,q]) 
        


        return fermi_op
        """

    def export_FN(self, config_a, config_b, shift=0):
        # Builds the F_{N} operator from a set of alpha and beta 
        # occupations and exports it as a FermionOperator
        # $f_{pq} = h_{pq} + \sum_{i \in occ} \langle pi || qi \rangle $
        fermi_op = openfermion.FermionOperator()
        for p in range(0, self.n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0, self.n_orb): 
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa+shift,1),(qa+shift,0)), self.int_H[pa,qa])
                fermi_op += openfermion.FermionOperator(((pb+shift,1),(qb+shift,0)), self.int_H[pb,qb])
                for i in config_a:
                    fermi_op += openfermion.FermionOperator(((pa+shift,1),(qa+shift,0)), self.int_V[pa,qa,2*i,2*i] - self.int_V[pa,2*i,2*i,qa])
                    fermi_op += openfermion.FermionOperator(((pb+shift,1),(qb+shift,0)), self.int_V[pb,qb,2*i,2*i])
                for j in config_b:
                    fermi_op += openfermion.FermionOperator(((pa+shift,1),(qa+shift,0)), self.int_V[pa,qa,2*j+1,2*j+1])
                    fermi_op += openfermion.FermionOperator(((pb+shift,1),(qb+shift,0)), self.int_V[pb,qb,2*j+1,2*j+1] - self.int_V[pb,2*j+1,2*j+1,qb])
        return fermi_op

    
    def export_VN(self, config_a, config_b, shift=0):
        # Builds the V_{N} operator from a set of alpha and beta
        # occupations and exports it as a FermionOperator
        fermi_op = openfermion.FermionOperator()
        for p in range(0, self.n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0, self.n_orb):
                qa = 2*q
                qb = 2*q+1
                for i in config_a:
                    fermi_op -= openfermion.FermionOperator(((pa+shift,1),(qa+shift,0)), self.int_V[pa,qa,2*i,2*i] - self.int_V[pa,2*i,2*i,qa])
                    fermi_op -= openfermion.FermionOperator(((pb+shift,1),(qb+shift,0)), self.int_V[pb,qb,2*i,2*i])
                for j in config_b:
                    fermi_op -= openfermion.FermionOperator(((pa+shift,1),(qa+shift,0)), self.int_V[pa,qa,2*j+1,2*j+1])
                    fermi_op -= openfermion.FermionOperator(((pb+shift,1),(qb+shift,0)), self.int_V[pb,qb,2*j+1,2*j+1] - self.int_V[pb,2*j+1,2*j+1,qb])
                for r in range(0, self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    for s in range(0, self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                        fermi_op += 0.5 * openfermion.FermionOperator(((pa+shift,1),(qa+shift,1),(sa+shift,0),(ra+shift,0)),
                            self.int_V[pa,ra,qa,sa])
                        fermi_op += 0.5 * openfermion.FermionOperator(((pa+shift,1),(qb+shift,1),(sb+shift,0),(ra+shift,0)),
                            self.int_V[pa,ra,qb,sb])
                        fermi_op += 0.5 * openfermion.FermionOperator(((pb+shift,1),(qa+shift,1),(sa+shift,0),(rb+shift,0)),
                            self.int_V[pb,rb,qa,sa])
                        fermi_op += 0.5 * openfermion.FermionOperator(((pb+shift,1),(qb+shift,1),(sb+shift,0),(rb+shift,0)),
                            self.int_V[pb,rb,qb,sb])
        return fermi_op

    def make_f(self, config_a, config_b):
        n_orb = self.n_orb
        f = cp.deepcopy(self.int_H)
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q+1
                for i in config_a:
                    f[pa,qa] += self.int_V[pa,qa,2*i,2*i] - self.int_V[pa,2*i,2*i,qa]
                    f[pb,qb] += self.int_V[pb,qb,2*i,2*i]
                for j in config_b:
                    f[pa,qa] += self.int_V[pa,qa,2*j+1,2*j+1]
                    f[pb,qb] += self.int_V[pb,qb,2*j+1,2*j+1] - self.int_V[pb,2*j+1,2*j+1,qb]
        return f

    def make_u(self, config_a, config_b):
        """
        u_{pq} = \sum_{i} <pi||qi>
        """
        n_orb = self.n_orb
        u = np.zeros((2*n_orb,2*n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q+1
                for i in config_a:
                    u[pa,qa] += self.int_V[pa,qa,2*i,2*i] - self.int_V[pa,2*i,2*i,qa]
                    u[pb,qb] += self.int_V[pb,qb,2*i,2*i]
                for j in config_b:
                    u[pa,qa] += self.int_V[pa,qa,2*j+1,2*j+1]
                    u[pb,qb] += self.int_V[pb,qb,2*j+1,2*j+1] - self.int_V[pb,2*j+1,2*j+1,qb]
        return u

    def make_v(self):
        """
        v^{pq}_{rs} = <pq||rs>
        """
        n_orb = self.n_orb
        v = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q+1
                for r in range(0,n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    for s in range(0,n_orb):
                        sa = 2*s
                        sb = 2*s+1
                        v[pa,qa,ra,sa] = self.int_V[pa,ra,qa,sa] - self.int_V[pa,sa,qa,ra]
                        v[pa,qb,ra,sb] = self.int_V[pa,ra,qb,sb]
                        v[pa,qb,rb,sa] = -self.int_V[pa,sa,qb,rb]
                        v[pb,qa,rb,sa] = self.int_V[pb,rb,qa,sa]
                        v[pb,qa,ra,sb] = -self.int_V[pb,sb,qa,ra]
                        v[pb,qb,rb,sb] = self.int_V[pb,rb,qb,sb] - self.int_V[pb,sb,qb,rb]
        return v

    def export_FN_ph(self, config_a, config_b):
        """
        Builds the F_{N} operator in the particle-hole basis
        \hat{F}_{N} = \sum_{pq} f_{q}^{p} \{\hat{p}^{\dagger}\hat{q}\}
        """
        n_orb = self.n_orb 
        n_a = len(config_a)
        n_b = len(config_b)
        fmat = self.make_f(config_a,config_b)
        fermi_op = openfermion.FermionOperator()
        # OO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((qa,1),(pa,0)), -fmat[pa,qa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((qb,1),(pb,0)), -fmat[pb,qb])
        # OV / VO
        for p in range(0,n_a):
            pa = 2*p
            for q in range(n_a,n_orb):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((pa,0),(qa,0)), fmat[pa,qa])
                fermi_op += openfermion.FermionOperator(((qa,1),(pa,1)), fmat[qa,pa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(n_b,n_orb):
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pb,0),(qb,0)), fmat[pb,qb])
                fermi_op += openfermion.FermionOperator(((qb,1),(pb,1)), fmat[qb,pb])
        # VV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), fmat[pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), fmat[pb,qb])
        return fermi_op

    def export_FN_ph2(self, config_a, config_b):
        """
        Builds the F_{N} operator in the particle-hole basis
        \hat{F}_{N} = \sum_{pq} f_{q}^{p} \{\hat{p}^{\dagger}\hat{q}\}
        """
        n_orb = self.n_orb 
        n_a = len(config_a)
        n_b = len(config_b)
        fmat = self.make_f(config_a,config_b)
        fermi_op = openfermion.FermionOperator()
        # OO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((qa,0),(pa,1)), -fmat[pa,qa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((qb,0),(pb,1)), -fmat[pb,qb])
        # OV / VO
        for p in range(0,n_a):
            pa = 2*p
            for q in range(n_a,n_orb):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), fmat[pa,qa])
                fermi_op += openfermion.FermionOperator(((qa,1),(pa,0)), fmat[qa,pa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(n_b,n_orb):
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), fmat[pb,qb])
                fermi_op += openfermion.FermionOperator(((qb,1),(pb,0)), fmat[qb,pb])
        # VV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), fmat[pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), fmat[pb,qb])
        return fermi_op

    def export_WN_ph(self, config_a, config_b):
        """
        Builds the W_{N} operator in the particle-hole basis
        \hat{W}_{N} = \frac{1}{4} \sum_{pqrs} \langle pq || rs \rangle \{\hat{p}^{\dagger}\hat{q}^{\dagger}\hat{s}\hat{r}\}
        """
        n_orb = self.n_orb
        n_a = len(config_a)
        n_b = len(config_b)
        vmat = self.make_v()
        fermi_op = openfermion.FermionOperator()
        #OOOO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,1),(ra,1),(pa,0),(qa,0)), 0.25*vmat[pa,qa,ra,sa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,1),(rb,1),(pb,0),(qb,0)), 0.25*vmat[pb,qb,rb,sb])
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,1),(ra,1),(pa,0),(qb,0)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(ra,1),(qb,0),(pa,0)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(pa,0),(qb,0)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(qb,0),(pa,0)), 0.25*vmat[qb,pa,sb,ra])
        #OO(OV) / (VO)OO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,1),(pa,0),(qa,0),(ra,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,1),(qa,1),(pa,1),(sa,0)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(0,n_b):
            pb = 2*p+1 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(n_b,n_orb):
                    rb = 2*r+1 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,1),(pb,0),(qb,0),(rb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(qb,1),(pb,1),(sb,0)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,1),(pa,0),(qb,0),(ra,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qb,0),(pa,0),(ra,0)), 0.5*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(qb,1),(pa,1),(sb,0)), 0.5*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(pa,1),(qb,1),(sb,0)), 0.5*vmat[ra,sb,qb,pa])
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,1),(pa,0),(qb,0),(rb,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qb,0),(pa,0),(rb,0)), 0.5*vmat[qb,pa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((rb,1),(qb,1),(pa,1),(sa,0)), 0.5*vmat[rb,sa,pa,qb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(pa,1),(qb,1),(sa,0)), 0.5*vmat[rb,sa,qb,pa])
        #VVOO / OOVV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,1),(ra,1)), 0.25*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,0),(sa,0),(qa,0),(pa,0)), 0.25*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1 
                    for s in range(0,n_b):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,1),(rb,1)), 0.25*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,0),(sb,0),(qb,0),(pb,0)), 0.25*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,1),(ra,1)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,1),(ra,1)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(ra,1),(sb,1)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(ra,1),(sb,1)), 0.25*vmat[qb,pa,sb,ra])
                        fermi_op += openfermion.FermionOperator(((ra,0),(sb,0),(qb,0),(pa,0)), 0.25*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,0),(sb,0),(pa,0),(qb,0)), 0.25*vmat[ra,sb,qb,pa])
                        fermi_op += openfermion.FermionOperator(((sb,0),(ra,0),(qb,0),(pa,0)), 0.25*vmat[sb,ra,pa,qb])
                        fermi_op += openfermion.FermionOperator(((sb,0),(ra,0),(pa,0),(qb,0)), 0.25*vmat[sb,ra,qb,pa])
        #VO(OV) / OV(VO)
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(ra,1),(qa,0),(sa,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qa,1),(ra,0),(pa,0)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(rb,1),(qb,0),(sb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qb,1),(rb,0),(pb,0)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pa,1),(ra,1),(qb,0),(sb,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qb,1),(ra,0),(pa,0)), 0.5*vmat[ra,sb,pa,qb])
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(rb,1),(qb,0),(sa,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qb,1),(rb,0),(pa,0)), 0.5*vmat[rb,sa,pa,qb])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(ra,1),(qa,0),(sb,0)), 0.5*vmat[pb,qa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qa,1),(ra,0),(pb,0)), 0.5*vmat[ra,sb,pb,qa])
                for r in range(0,n_b):
                    rb = 2*r+1 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pb,1),(rb,1),(qa,0),(sa,0)), 0.5*vmat[pb,qa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qa,1),(rb,0),(pb,0)), 0.5*vmat[rb,sa,pb,qa])
        #VV(VO) \ (VO)VV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,1),(ra,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sa,0),(qa,0),(pa,0)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,1),(rb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sb,0),(qb,0),(pb,0)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,1),(ra,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,1),(ra,0)), 0.5*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,0),(qb,0),(pa,0)), 0.5*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,0),(pa,0),(qb,0)), 0.5*vmat[ra,sb,qb,pa])
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sa,1),(rb,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sa,1),(rb,0)), 0.5*vmat[qb,pa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sa,0),(qb,0),(pa,0)), 0.5*vmat[rb,sa,pa,qb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sa,0),(pa,0),(qb,0)), 0.5*vmat[rb,sa,qb,pa])
        #VVVV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), 0.25*vmat[pa,qa,ra,sa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(n_b,n_orb):
                    rb = 2*r+1 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), 0.25*vmat[pb,qb,rb,sb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,0),(ra,0)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(ra,0),(sb,0)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(ra,0),(sb,0)), 0.25*vmat[qb,pa,sb,ra])
        return fermi_op

    def export_WN_ph2(self, config_a, config_b):
        """
        Builds the W_{N} operator in the particle-hole basis
        \hat{W}_{N} = \frac{1}{4} \sum_{pqrs} \langle pq || rs \rangle \{\hat{p}^{\dagger}\hat{q}^{\dagger}\hat{s}\hat{r}\}
        """
        n_orb = self.n_orb
        n_a = len(config_a)
        n_b = len(config_b)
        vmat = self.make_v()
        fermi_op = openfermion.FermionOperator()
        #OOOO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,0),(ra,0),(pa,1),(qa,1)), 0.25*vmat[pa,qa,ra,sa])
        for p in range(0,n_b):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,0),(rb,0),(pb,1),(qb,1)), 0.25*vmat[pb,qb,rb,sb])
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,0),(ra,0),(pa,1),(qb,1)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,0),(ra,0),(qb,1),(pa,1)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,0),(sb,0),(pa,1),(qb,1)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((ra,0),(sb,0),(qb,1),(pa,1)), 0.25*vmat[qb,pa,sb,ra])
        #OO(OV) / (VO)OO
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,0),(pa,1),(qa,1),(ra,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,1),(qa,0),(pa,0),(sa,1)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(0,n_b):
            pb = 2*p+1 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(n_b,n_orb):
                    rb = 2*r+1 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,0),(pb,1),(qb,1),(rb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(qb,0),(pb,0),(sb,1)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(0,n_a):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((sb,0),(pa,1),(qb,1),(ra,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,0),(qb,1),(pa,1),(ra,0)), 0.5*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(qb,0),(pa,0),(sb,1)), 0.5*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(pa,0),(qb,0),(sb,1)), 0.5*vmat[ra,sb,qb,pa])
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((sa,0),(pa,1),(qb,1),(rb,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,0),(qb,1),(pa,1),(rb,0)), 0.5*vmat[qb,pa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((rb,1),(qb,0),(pa,0),(sa,1)), 0.5*vmat[rb,sa,pa,qb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(pa,0),(qb,0),(sa,1)), 0.5*vmat[rb,sa,qb,pa])
        #VVOO / OOVV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), 0.25*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sa,1),(qa,0),(pa,0)), 0.25*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1 
                    for s in range(0,n_b):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), 0.25*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sb,1),(qb,0),(pb,0)), 0.25*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,0),(ra,0)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(ra,0),(sb,0)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(ra,0),(sb,0)), 0.25*vmat[qb,pa,sb,ra])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(qb,0),(pa,0)), 0.25*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(pa,0),(qb,0)), 0.25*vmat[ra,sb,qb,pa])
                        fermi_op += openfermion.FermionOperator(((sb,1),(ra,1),(qb,0),(pa,0)), 0.25*vmat[sb,ra,pa,qb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(ra,1),(pa,0),(qb,0)), 0.25*vmat[sb,ra,qb,pa])
        #VO(OV) / OV(VO)
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(ra,0),(qa,1),(sa,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qa,0),(ra,1),(pa,0)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(0,n_b):
                qb = 2*q+1 
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(rb,0),(qb,1),(sb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qb,0),(rb,1),(pb,0)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(0,n_b):
                qb = 2*q+1
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1 
                        fermi_op += openfermion.FermionOperator(((pa,1),(ra,0),(qb,1),(sb,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qb,0),(ra,1),(pa,0)), 0.5*vmat[ra,sb,pa,qb])
                for r in range(0,n_b):
                    rb = 2*r+1
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(rb,0),(qb,1),(sa,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qb,0),(rb,1),(pa,0)), 0.5*vmat[rb,sa,pa,qb])
        for p in range(n_b,n_orb):
            pb = 2*p+1
            for q in range(0,n_a):
                qa = 2*q 
                for r in range(0,n_a):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(ra,0),(qa,1),(sb,0)), 0.5*vmat[pb,qa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((sb,1),(qa,0),(ra,1),(pb,0)), 0.5*vmat[ra,sb,pb,qa])
                for r in range(0,n_b):
                    rb = 2*r+1 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pb,1),(rb,0),(qa,1),(sa,0)), 0.5*vmat[pb,qa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((sa,1),(qa,0),(rb,1),(pb,0)), 0.5*vmat[rb,sa,pb,qa])
        #VV(VO) \ (VO)VV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), 0.5*vmat[pa,qa,ra,sa])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sa,1),(qa,0),(pa,0)), 0.5*vmat[ra,sa,pa,qa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), 0.5*vmat[pb,qb,rb,sb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sb,1),(qb,0),(pb,0)), 0.5*vmat[rb,sb,pb,qb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(0,n_b):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), 0.5*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,0),(ra,0)), 0.5*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(qb,0),(pa,0)), 0.5*vmat[ra,sb,pa,qb])
                        fermi_op += openfermion.FermionOperator(((ra,1),(sb,1),(pa,0),(qb,0)), 0.5*vmat[ra,sb,qb,pa])
                for r in range(n_b,n_orb):
                    rb = 2*r+1
                    for s in range(0,n_a):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sa,0),(rb,0)), 0.5*vmat[pa,qb,rb,sa])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sa,0),(rb,0)), 0.5*vmat[qb,pa,rb,sa])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sa,1),(qb,0),(pa,0)), 0.5*vmat[rb,sa,pa,qb])
                        fermi_op += openfermion.FermionOperator(((rb,1),(sa,1),(pa,0),(qb,0)), 0.5*vmat[rb,sa,qb,pa])
        #VVVV
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_a,n_orb):
                qa = 2*q 
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(n_a,n_orb):
                        sa = 2*s 
                        fermi_op += openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), 0.25*vmat[pa,qa,ra,sa])
        for p in range(n_b,n_orb):
            pb = 2*p+1 
            for q in range(n_b,n_orb):
                qb = 2*q+1 
                for r in range(n_b,n_orb):
                    rb = 2*r+1 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), 0.25*vmat[pb,qb,rb,sb])
        for p in range(n_a,n_orb):
            pa = 2*p 
            for q in range(n_b,n_orb):
                qb = 2*q+1
                for r in range(n_a,n_orb):
                    ra = 2*r 
                    for s in range(n_b,n_orb):
                        sb = 2*s+1
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), 0.25*vmat[pa,qb,ra,sb])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(sb,0),(ra,0)), 0.25*vmat[qb,pa,ra,sb])
                        fermi_op += openfermion.FermionOperator(((pa,1),(qb,1),(ra,0),(sb,0)), 0.25*vmat[pa,qb,sb,ra])
                        fermi_op += openfermion.FermionOperator(((qb,1),(pa,1),(ra,0),(sb,0)), 0.25*vmat[qb,pa,sb,ra])
        return fermi_op


def init(molecule,charge,spin,basis,reference='rhf',n_frzn_occ=0, n_act=None, mo_order=None):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 8e3 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.symmetry = False
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_a, n_b = mol.nelec 
    n_el = n_a + n_b

    if n_act == None:
        n_act = n_orb

    #SCF
    if reference == "rhf": 
        mf = scf.RHF(mol)
    elif reference == "rohf":
        mf = scf.ROHF(mol)
    elif reference == "uhf":
        mf = scf.UHF(mol)
    else:
        print("Please specify a proper reference (rhf/rohf/uhf).")
    mf.conv_tol_grad = 1e-14
    mf.max_cycle = 1000
    mf.verbose = 4
    mf.init_guess = 'atom'
    #mf = scf.newton(mf).set(conv_tol=1e-12)

    hf_energy = mf.kernel()
    #internal stability check
    # print("Checking the internal stability of the SCF solution.")
    # new_mo = mf.stability()[0]
    # mo_diff = 0
    # j = 0
    # log = lib.logger.new_logger(mf)
    # if reference != "uhf":
    #     mo_diff = np.linalg.norm(mf.mo_coeff - new_mo)
    #     while (mo_diff > 1e-5) and (j < 3):
    #         print("Rotating orbitals to find stable solution: attempt %d."%(j+1))
    #         new_dm = mf.make_rdm1(new_mo,mf.mo_occ)
    #         mf.run(new_dm)
    #         new_mo = mf.stability()[0]
    #         mo_diff = np.linalg.norm(mf.mo_coeff - new_mo)
    #         j += 1
    #     if mo_diff > 1e-5:
    #         print("Unable to find a stable SCF solution after %d attempts."%(j+1))
    #     else:
    #         print("SCF solution is internally stable.")
    # else:
    #     mo_diff = np.linalg.norm(mf.mo_coeff[0] - new_mo[0]) + np.linalg.norm(mf.mo_coeff[1] - new_mo[1])
    #     while (mo_diff > 1e-5) and (j < 3):
    #         print("Rotating orbitals to find stable solution: attempt %d."%(j+1))
    #         new_dm = mf.make_rdm1(new_mo,mf.mo_occ)
    #         mf.run(new_dm)
    #         new_mo = mf.stability()[0]
    #         mo_diff = np.linalg.norm(mf.mo_coeff[0] - new_mo[0]) + np.linalg.norm(mf.mo_coeff[1] - new_mo[1])
    #         j += 1
    #     if mo_diff > 1e-5:
    #         print("Unable to find a stable SCF solution after %d attempts."%(j+1))
    #     else:
    #         print("SCF solution is internally stable.")

    hf_energy = mf.e_tot

    assert mf.converged == True
    mo_occ = mf.mo_occ
    C = mf.mo_coeff
    occ_a = 0
    occ_b = 0
    virt_a = 0
    virt_b = 0

    # dump orbitals for viewing 
    #molden.from_mo(mol, 'orbitals_canon.molden', C)

    if reference != "uhf":
        C_a = C_b = mf.mo_coeff
        mo_a = np.zeros(len(mo_occ))
        mo_b = np.zeros(len(mo_occ))
        for i in range(0, len(mo_occ)):
            if mo_occ[i] > 0:
                mo_a[i] = 1
                occ_a += 1
            else:
                virt_a += 1
            if mo_occ[i] > 1:
                mo_b[i] = 1
                occ_b += 1
            else:
                virt_b += 1

    else:
        C_a = mf.mo_coeff[0]
        C_b = mf.mo_coeff[1]
        mo_a = np.zeros(len(mo_occ[0]))
        mo_b = np.zeros(len(mo_occ[1]))
        for i in range(0, len(mo_occ[0])):
            if mo_occ[0][i] == 1:
                mo_a[i] = 1
                occ_a += 1
            else:
                virt_a += 1
        for i in range(0, len(mo_occ[1])):
            if mo_occ[1][i] == 1:
                mo_b[i] = 1
                occ_b += 1
            else:
                virt_b += 1

    P_a = np.diag(mo_a)
    P_b = np.diag(mo_b)

    E_nuc = mol.energy_nuc()

    #
    #if mo_order != None:
    #    print(len(mo_order) , mf.mo_coeff.shape[1])
    #    assert(len(mo_order) == mf.mo_coeff.shape[1])
    #    mf.mo_coeff = mf.mo_coeff[:,mo_order]
    


    #C = mf.mo_coeff #MO coeffs
    #S = mf.get_ovlp()

    ##READING INTEGRALS FROM PYSCF
    E_nuc = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    H_core = T + V
    S = mol.intor('int1e_ovlp_sph')
    I = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(n_el))
    print("Number of alpha electrons                      :%10i" %(n_a))
    print("Number of beta electrons                       :%10i" %(n_b))
    print("Nuclear Repulsion                              :%18.12f " %E_nuc)
    print("Electronic SCF energy                          :%18.12f " %(mf.e_tot-E_nuc))
    print("SCF Energy                                     :%21.15f"%(mf.e_tot))


    print(" AO->MO")
    #convert from AO to MO representation
    H_a = C_a.T.dot(H_core).dot(C_a)
    H_b = C_b.T.dot(H_core).dot(C_b)

    I_aa = np.einsum("pqrs,pi->iqrs",I,C_a)
    I_aa = np.einsum("iqrs,qj->ijrs",I_aa,C_a)
    I_aa = np.einsum("ijrs,rk->ijks",I_aa,C_a)
    I_aa = np.einsum("ijks,sl->ijkl",I_aa,C_a)
    
    I_ab = np.einsum("pqrs,pi->iqrs",I,C_a)
    I_ab = np.einsum("iqrs,qj->ijrs",I_ab,C_a)
    I_ab = np.einsum("ijrs,rk->ijks",I_ab,C_b)
    I_ab = np.einsum("ijks,sl->ijkl",I_ab,C_b)
    
    I_ba = np.einsum("pqrs,pi->iqrs",I,C_b)
    I_ba = np.einsum("iqrs,qj->ijrs",I_ba,C_b)
    I_ba = np.einsum("ijrs,rk->ijks",I_ba,C_a)
    I_ba = np.einsum("ijks,sl->ijkl",I_ba,C_a)

    I_bb = np.einsum("pqrs,pi->iqrs",I,C_b)
    I_bb = np.einsum("iqrs,qj->ijrs",I_bb,C_b)
    I_bb = np.einsum("ijrs,rk->ijks",I_bb,C_b)
    I_bb = np.einsum("ijks,sl->ijkl",I_bb,C_b)

    J_a = np.einsum("pqrs,rs->pq",I_aa,P_a) + np.einsum("pqrs,rs->pq",I_ab,P_b)
    J_b = np.einsum("pqrs,rs->pq",I_bb,P_b) + np.einsum("pqrs,rs->pq",I_ba,P_a)
    K_a = np.einsum("pqrs,rq->ps",I_aa,P_a)
    K_b = np.einsum("pqrs,rq->ps",I_bb,P_b)

    F_a = H_a + J_a - K_a
    F_b = H_b + J_b - K_b
    manual_energy = E_nuc + 0.5*np.einsum("pq,pq",H_a+F_a,P_a) + 0.5*np.einsum("pq,pq",H_b+F_b,P_b)
    print("Manual HF energy = %21.15f"%(manual_energy))
    onebody = np.einsum("pq,pq",H_a,P_a) + np.einsum("pq,pq",H_b,P_b)
    twobody = 0.5*np.einsum("pq,pq",J_a-K_a,P_a) + 0.5*np.einsum("pq,pq",J_b-K_b,P_b)
    #print("One-body energy = %16.10f"%onebody)
    #print("Two-body energy = %16.10f"%twobody)

    #group terms to be exported in {a0,b0,a1,b2,...} MO ordering
    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    AA = np.einsum("pq,rs->pqrs",A,A)
    AB = np.einsum("pq,rs->pqrs",A,B)
    BA = np.einsum("pq,rs->pqrs",B,A)
    BB = np.einsum("pq,rs->pqrs",B,B)

    h = np.kron(H_a,A) + np.kron(H_b,B)
    p = np.kron(P_a,A) + np.kron(P_b,B)
    #f = np.kron(F_a,A) + np.kron(F_b,B)
    C = np.kron(C_a,A) + np.kron(C_b,B)
    S = np.kron(S,A) + np.kron(S,B)
    g = np.kron(I_aa,AA) + np.kron(I_ab,AB) + np.kron(I_ba,BA) + np.kron(I_bb,BB)

    #print("(1a 1a | 1b 1b) = %16.10f"%(I_ab[1,1,1,1]))
    #print("(1a 1a | 1b 1b) = %16.10f"%(g[2,2,3,3]))
    #g -= np.einsum("pqrs->prsq",g)
    #g *= -0.25

    #FCI
    if False:
        cisolver = fci.FCI(mf)
        print("FCI energy = %21.15f"%cisolver.kernel()[0])

    return(n_orb, n_a, n_b, h, g, mol, E_nuc, mf.e_tot, C, S)

    #g = np.einsum("pqrs,pl->lqrs",g,C)
    #g = np.einsum("lqrs,qm->lmrs",g,C)
    #g = np.einsum("lmrs,rn->lmns",g,C)
    #g = np.einsum("lmns,so->lmno",g,C)

    #TODO:Implement frozen core
    comment="""
    assert(n_frzn_occ <= n_b)
    n_frzn_vir = n_orb - n_act - n_frzn_occ
    assert(n_frzn_vir >= 0)

    n_a   -= n_frzn_occ
    n_b   -= n_frzn_occ
    n_orb -= n_frzn_occ

    print(" NElectrons: %4i %4i" %(n_a, n_b))
    Cact = C[:,n_frzn_occ:n_frzn_occ+n_act]
    Cocc = C[:,0:n_frzn_occ]

    dm = Cocc @ Cocc.T
    j, k = scf.hf.get_jk(mol, dm)
 
    t = hcore + 2*j - k
    h = reduce(np.dot, (Cact.conj().T, hcore + 2*j - k, Cact))
    ecore = np.trace(2*dm @ (hcore + j - .5*k))
    print(" ecore: %12.8f" %ecore)

    E_nuc += ecore
    def view(h5file, dataname='eri_mo'):
        f5 = h5py.File(h5file)
        print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
        f5.close()
    
    eri_act = ao2mo.outcore.general_iofree(mol, (Cact,Cact,Cact,Cact), intor='int2e', aosym='s4', compact=True)
    #view('ints_occ.h5')
    #view('ints_act.h5')
    eri_act = ao2mo.restore('s1', eri_act, Cact.shape[1])
    print(" ERIs in the active-space:")
    print(eri_act.shape, " %12.8f Mb" %(eri_act.nbytes*1e-6))
    
    if False:
        #compute slater determinant energy
        e1 = 0
        e2 = 0
        config_a = range(n_a)
        config_b = range(n_b)
        print(config_a,config_b)
        for i in config_a:
            e1 += h[i,i]
        for i in config_b:
            e1 += h[i,i]
        for i in config_a:
            for j in config_a:
                if i>=j:
                    continue
                e2 += eri_act[i,i,j,j]
                e2 -= eri_act[i,j,j,i]
        for i in config_b:             
            for j in config_b:         
                if i>=j:                
                    continue           
                e2 += eri_act[i,i,j,j]
                e2 -= eri_act[i,j,j,i]
        for i in config_a:             
            for j in config_b:         
                e2 += eri_act[i,i,j,j]
        e = e1+e2
        print("*HF Energy: %12.8f" %(e+E_nuc))
    
    fci = 0
    #pyscf FCI
    if fci:
        print()
        print(" ----------------------")
        print(" PYSCF")
        mc = mcscf.CASCI(mf, n_act, (n_a,n_b),ncore=n_frzn_occ)
        #mc.fcisolver = pyscf.fci.solver(mf, singlet=True)
        #mc.fcisolver = pyscf.fci.direct_spin1.FCISolver(mol)
        efci, ci = mc.kernel()
        print(" PYSCF: FCI energy: %12.8f" %(efci))
        print()
    
    Cact = C[:,n_frzn_occ:n_frzn_occ+n_act]
    
    return(n_act, n_a, n_b, h, eri_act, mol, E_nuc ,mf.e_tot,Cact,S)
    """
# }}}





def init_pyscf(molecule,charge,spin,basis,orbitals):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b

    #SCF 
    mf = scf.RHF(mol).run()
    #mf = scf.ROHF(mol).run()
    C = mf.mo_coeff #MO coeffs
    S = mf.get_ovlp()

    print(" Orbs1:")
    print(C)
    Cl = cp.deepcopy(C)
    if orbitals == "boys":
        print("\nUsing Boys localised orbitals:\n")
        cl_o = lo.Boys(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.Boys(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))

    elif orbitals == "pipek":
        print("\nUsing Pipek-Mezey localised orbitals:\n")
        cl_o = lo.PM(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.PM(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "edmiston":
        print("\nUsing Edmiston-Ruedenberg localised orbitals:\n")
        cl_o = lo.ER(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.ER(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
#    elif orbitals == "svd":
#        print("\nUsing SVD localised orbitals:\n")
#        cl_o = cp.deepcopy(mf.mo_coeff[:,:n_a])
#        cl_v = cp.deepcopy(mf.mo_coeff[:,n_a:])
#
#        [U,s,V] = np.linalg.svd(cl_o)
#        cl_o = cl_o.dot(V.T)
#        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "canonical":
        print("\nUsing Canonical orbitals:\n")
        pass
    else:
        print("Error: Wrong orbital specification:")
        exit()

    
    print(" Overlap:")
    print(C.T.dot(S).dot(Cl))

    # sort by cluster
    blocks = [[0,1,2,3],[4,5,6,7]]
    O = Cl[:,:n_a] 
    V = Cl[:,n_a:] 
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,O)
    O = O[:,sorted_order]
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,V)
    V = V[:,sorted_order]
    Cl = np.column_stack((O,V)) 
    
    C = cp.deepcopy(Cl)
    # dump orbitals for viewing 
    molden.from_mo(mol, 'orbitals_canon.molden', C)
    molden.from_mo(mol, 'orbitals_local.molden', Cl)

    ##READING INTEGRALS FROM PYSCF
    E_nu = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nu)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nu))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))


    print(" AO->MO")
    g = np.einsum("pqrs,pl->lqrs",g,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    h = reduce(np.dot, (C.conj().T, hcore, C))

    
#    #mf = mf.density_fit(auxbasis='weigend')
#    #mf._eri = None
#    mcc = cc.UCCSD(mf)
#    eris = mcc.ao2mo()
#    eris.g = g
#    eris.focka = h
#    eris.fockb = h
#    
#    emp2, t1, t2 = mcc.init_amps(eris)
#    exit()
#    print(abs(t2).sum() - 4.9318753386922278)
#    print(emp2 - -0.20401737899811551)
#    t1, t2 = update_amps(mcc, t1, t2, eris)
#    print(abs(t1).sum() - 0.046961325647584914)
#    print(abs(t2).sum() - 5.378260578551683   )
#    
#    
#    exit()

    return(n_orb, n_a, n_b, h, g, mol, E_nu,mf.e_tot,C,S)
# }}}





def mulliken_clustering(clusters,mol,C):
    aos_per_atom = []
    aos_per_cluster = []

    #check n atoms
    max_atomid = 0
    tmp_natm = 0
    for c in clusters:
        for ci in c:
            max_atomid = max(max_atomid,ci)
            tmp_natm += 1

    assert(max_atomid == tmp_natm-1)
    assert(tmp_natm == mol.natm)
    #for d in dir(mol):
    #    print(d)
    for a in range(mol.natm):
        aos_per_atom.append([])
    for mu in range(C.shape[0]):
        aos_per_atom[mol.bas_atom(mu)].append(mu)

    mos_per_cluster = [[] for i in range(len(clusters))]
    print(" AOs per cluster")
    for c in clusters:
        tmp = []
        for ci in c:
            tmp.extend(aos_per_atom[ci])
        aos_per_cluster.append(tmp)
        print(tmp)

    S = mol.intor('int1e_ovlp_sph')
    for i in range(C.shape[1]):
        print(" Orbital %4i" %i)
        pi = np.einsum('a,b,bc->ac',C[:,i],C[:,i],S)
        
        pop_per_cluster = []

        for c in aos_per_cluster:
            pop_per_cluster.append(np.trace(pi[:,c][c]))
        mos_per_cluster[np.argmax(pop_per_cluster)].append(i)
    print(mos_per_cluster)
    sorted_orb_list = []
    cluster_sizes = []
    for c in mos_per_cluster:
        sorted_orb_list.extend(c)
        cluster_sizes.append(len(c))

    return(sorted_orb_list,cluster_sizes)



    



