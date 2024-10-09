import openfermion
import numpy as np
import copy as cp

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0
        self.gradient_print_thresh = 0

    def init(self,n_orb,
            n_occ_a=None,
            n_occ_b=None,
            n_vir_a=None,
            n_vir_b=None):
        self.n_orb = n_orb
        self.n_spin_orb = 2*self.n_orb

        if n_occ_a!=None and n_occ_b!=None:
            #assert(n_occ_a == n_occ_b)
            self.n_occ = n_occ_a
            self.n_occ_a = n_occ_a
            self.n_occ_b = n_occ_b
            self.n_vir = n_vir_a
            self.n_vir_a = n_vir_a
            self.n_vir_b = n_vir_b
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            self.spmat_ops.append(linalg.get_number_preserving_sparse_operator(op, self.n_spin_orb, self.n_occ_a+self.n_occ_b,spin_preserving=True))
        assert(len(self.spmat_ops) == self.n_ops)
        return

    def get_string_for_term(self,op):

        opstring = ""
        spins = ""
        for t in op.terms:


            opstring = "("
            for ti in t:
                opstring += str(int(ti[0]/2))
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("wrong")
                    exit()
                spins += str(ti[0]%2)

#            if self.fermi_ops[i].terms[t] > 0:
#                spins = "+"+spins
#            if self.fermi_ops[i].terms[t] < 0:
#                spins = "-"+spins
            opstring += ")"
            spins += " "
        opstring = " %18s : %s" %(opstring, spins)
        return opstring



    def compute_gradient_i(self,i,v,sig):
        """
        For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
        g(k) = 2Real<HA(k)>

        Note - this assumes A(k) is an antihermitian operator. If this is not the case, the derived class should
        reimplement this function. Of course, also assumes H is hermitian

        v   = current_state
        sig = H*v

        """
        opA = self.spmat_ops[i]
        gi = 2*(sig.transpose().conj().dot(opA.dot(v)))
        assert(gi.shape == (1,1))
        gi = gi[0,0]
        assert(np.isclose(gi.imag,0))
        gi = gi.real

        opstring = self.get_string_for_term(self.fermi_ops[i])

        if abs(gi) > self.gradient_print_thresh:
            print(" %4i %12.8f %s" %(i, gi, opstring) )

        return gi


class spin_complement_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        alpha_orbs = [2*i for i in range(self.n_orb)]
        beta_orbs = [2*i+1 for i in range(self.n_orb)]

        ops = []
        #aa
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>=q:
                    continue
                #if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
                #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
                #    continue
                one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
                one_elec += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
                ops.append(one_elec)
        #aa
        pq = 0
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>q:
                    continue
                rs = 0
                for r in alpha_orbs:
                    for s in alpha_orbs:
                        if r>s:
                            continue
                        if pq<rs:
                            continue
                        #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                            #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                            #continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        two_elec += openfermion.FermionOperator(((r+1,1),(p+1,0),(s+1,1),(q+1,0)))-openfermion.FermionOperator(((q+1,1),(s+1,0),(p+1,1),(r+1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1


        #ab
        pq = 0
        for p in alpha_orbs:
            for q in beta_orbs:
                rs = 0
                for r in alpha_orbs:
                    for s in beta_orbs:
                        if pq<rs:
                            continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        if p>q:
                            continue
                        two_elec += openfermion.FermionOperator(((s-1,1),(q-1,0),(r+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(r+1,0),(q-1,1),(s-1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1

        self.fermi_ops = ops
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}




class spin_complement_GSD2(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form spin-complemented GSD operators")

        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                if termA.many_body_order() > 0:
                    self.fermi_ops.append(termA)


        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))

                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
#                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
#
#                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
#                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
#
#                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
#                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        print()
#                        print(p,q,r,s)
#                        print(termA)
#                        print(termB)
#                        print(termC)
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
                        termC -= hermitian_conjugated(termC)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        termC = normal_ordered(termC)

                        if termA.many_body_order() > 0:
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            self.fermi_ops.append(termB)

                        if termC.many_body_order() > 0:
                            self.fermi_ops.append(termC)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}




class singlet_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)


        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

#                        oplist = []
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12)))
#
#                        print(p,q,r,s)
#                        for i in range(len(oplist)):
#                            oplist[i] -= hermitian_conjugated(oplist[i])
#                        for i in range(len(oplist)):
#                            for j in range(i+1,len(oplist)):
#                                opi = oplist[i]
#                                opj = oplist[i]
#                                opij = opi*opj - opj*opi
#                                opij = normal_ordered(opij)
#                                print(opij, end='')
#                        print()
                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t


                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}




class singlet_SD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet SD operators")
        self.fermi_ops = []

        assert(self.n_occ_a == self.n_occ_b)
        n_occ = self.n_occ
        n_vir = self.n_vir

        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)


        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1

                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))

                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t


                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
    # }}}



class unrestricted_SD(OperatorPool):
    def generate_SQ_operators(self):
        print("Forming unrestricted SD operators.")
        self.fermi_ops=[]
        n_occ_a = self.n_occ_a
        n_occ_b = self.n_occ_b
        n_vir_a = self.n_vir_a
        n_vir_b = self.n_vir_b
        
        #singles
        #aa
        for a in range(0,n_vir_a):
            a_a = 2*a + 2*n_occ_a
            for i in range(0,n_occ_a):
                i_a = 2*i

                termA = FermionOperator(((a_a,1),(i_a,0)),1)
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t*coeff_t
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
        #bb
        for a in range(0,n_vir_b):
            a_b = 2*a + 2*n_occ_b+1
            for i in range(0,n_occ_b):
                i_b = 2*i+1

                termB = FermionOperator(((a_b,1),(i_b,0)),1)
                termB -= hermitian_conjugated(termB)
                termB = normal_ordered(termB)
                #Normalize
                coeffB = 0
                for t in termsB.terms:
                    coeff_t = termB.terms[t]
                    coeffB += coeff_t * coeff_t
                if termB.many_body_order() > 0:
                    termB = termB/np.sqrt(coeffB)
                    self.fermi_ops.append(termB)

        #doubles
        #aaaa
        for a in range(0,n_vir_a):
            a_a = 2*a + 2*n_occ_a
            for b in range(a,n_vir_a):
                b_a = 2*b + 2*n_occ_a
                for i in range(0,n_vir_a):
                    i_a = 2*i
                    for j in range(i,n_vir_a):
                        j_a = 2*j

                        termAA = FermionOperator(((a_a,1),(b_a,1),(i_a,0),(j_a,0)),1)
                        termAA -= hermitian_conjugated(termAA)
                        termAA = normal_ordered(termAA)
                        #Normalize
                        coeffAA = 0
                        for t in termAA.terms:
                            coeff_t = termAA.terms[t]
                            coeffAA += coeff_t * coeff_t
                        if termAA.many_body_order() > 0:
                            termAA = termAA/np.sqrt(coeffAA)
                            self.fermi_ops.append(termAA)

        #abab
        for a in range(0,n_vir_a):
            a_a = 2*a + 2*n_occ_a
            for b in range(0,n_vir_b):
                b_b = 2*b +2*n_occ_b+1
                for i in range(0,n_occ_a):
                    i_a = 2*i
                    for j in range(0,n_occ_b):
                        j_b = 2*j+1

                        termAB = FermionOperator(((a_a,1),(b_b,1),(i_a,0),(j_b,0)),1)
                        termAB -= hermitian_conjugated(termAB)
                        termAB = normal_ordered(termAB)
                        #Normalize
                        coeffAB = 0
                        for t in termAB.terms:
                            coeff_t = termAB.terms[t]
                            coeffAB += coeff_t * coeff_t
                        if termAB.many_body_order() > 0:
                            termAB = termAB/np.sqrt(coeffAB)
                            self.fermi_ops.append(termAB)

        #bbbb
        for a in range(0,n_vir_b):
            a_b = 2*a + 2*n_occ_b+1
            for b in range(a,n_vir_b):
                b_b = 2*b + 2*n_occ_b+1
                for i in range(0,n_occ_b):
                    i_b = 2*i+1
                    for j in range(i,n_occ_b):
                        j_b = 2*j+1

                        termBB = FermionOperator(((a_b,1),(b_b,1),(i_b,0),(j_b,0)),1)
                        termBB -= hermitian_conjugated(termBB)
                        termBB = normal_ordered(termBB)
                        #Normalize
                        coeffBB = 0
                        for t in termBB.terms:
                            coeff_t = termBB.terms[t]
                            coeffBB = coeff_t * coeff_t
                        if termBB.many_body_order() > 0:
                            termBB = termBB/np.sqrt(coeffBB)
                            self.fermi_ops.append(termBB)

        self.n_ops = len(self.fermi_ops)
        print("Number of operators: %d"%self.n_ops)
        return

    def get_string_for_term(self,op):

        opstring = ""
        spins = ""

        for t in op.terms:
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0]/2))
                if ti[0]%2 == 0:
                    opstring += "a"
                elif ti[0]%2 == 1:
                    opstring += "b"
                else:
                    print("Problem getting string for term!")
                    exit()
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("Problem getting string for term!")
                    exit()
                spins += str(ti[0]%2)

            opstring += ")"
            spins += " "
        opstring = " %18s : %s"%(opstring, spins)
        return opstring

class unrestricted_GSD(OperatorPool):
    def generate_SQ_Operators(self):
        print("Forming unrestricted GSD operators.")
        
        self.fermi_ops=[]
        n_orb = self.n_orb
        
        #singles
        for p in range(0,n_orb):
            p_a = 2*p
            p_b = 2*p+1
            for q in range(p+1,n_orb):
                q_a = 2*q
                q_b = 2*q+1

                termA = FermionOperator(((p_a,1),(q_a,0)),1)
                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

                termB = FermionOperator(((p_b,1),(q_b,0)),1)
                termB -= hermitian_conjugated(termB)
                termB = normal_ordered(termB)
                #Normalize
                coeffB = 0
                for t in termB.terms:
                    coeff_t = termB.terms[t]
                    coeffB += coeff_t * coeff_t
                if termB.many_body_order() > 0:
                    termB = termB/np.sqrt(coeffB)
                    self.fermi_ops.append(termB)  

        #doubles
        pq = -1
        for p in range(0,n_orb):
            p_a = 2*p
            p_b = 2*p+1
            for q in range(0,n_orb):
                q_a = 2*q
                q_b = 2*q+1

                pq += 1
                rs = -1
                for r in range(0,n_orb):
                    r_a = 2*r
                    r_b = 2*r+1
                    for s in range(0,n_orb):
                        s_a = 2*s 
                        s_b = 2*s+1

                        rs += 1
                        if (pq < rs):
                            #abab
                            termAB = FermionOperator(((p_a,1),(q_b,1),(r_a,0),(s_b,0)),1)
                            termAB -= hermitian_conjugated(termAB)
                            termAB = normal_ordered(termAB)
                            #Normalize
                            coeffAB = 0
                            for t in termAB.terms:
                                coeff_t = termAB.terms[t]
                                coeffAB += coeff_t * coeff_t
                            if termAB.many_body_order() > 0:
                                termAB = termAB/np.sqrt(coeffAB)
                                self.fermi_ops.append(termAB)

                            if ((q > p) and (s > r)):
                                #aaaa
                                termAA = FermionOperator(((p_a,1),(q_a,1),(r_a,0),(s_a,0)),1)
                                termAA -= hermitian_conjugated(termAA)
                                termAA = normal_ordered(termAA)
                                #Normalize
                                coeffAA = 0
                                for t in termAA.terms:
                                    coeff_t = termAA.terms[t]
                                    coeffAA = coeff_t * coeff_t
                                if termAA.many_body_order() > 0:
                                    termAA = termAA/np.sqrt(coeffAA)
                                    self.fermi_ops.append(termAA)
                                #bbbb
                                termBB = FermionOperator(((p_b,1),(q_b,1),(r_b,0),(s_b,0)),1)
                                termBB -= hermitian_conjugated(termBB)
                                termBB = normal_ordered(termBB)
                                #Normalize
                                coeffBB = 0
                                for t in termBB.terms:
                                    coeff_t = termBB.terms[t]
                                    coeffBB = coeff_t * coeff_t
                                if termBB.many_body_order() > 0:
                                    termBB = termBB/np.sqrt(coeffBB)
                                    self.fermi_ops.append(termBB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

    def get_string_for_term(self,op):

        opstring = ""
        spins = ""

        for t in op.terms:
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0]/2))
                if ti[0]%2 == 0:
                    opstring += "a"
                elif ti[0]%2 == 1:
                    opstring += "b"
                else:
                    print("Problem getting string for term!")
                    exit()
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("Problem getting string for term!")
                    exit()
                spins += str(ti[0]%2)

            opstring += ")"
            spins += " "
        opstring = " %18s : %s"%(opstring, spins)
        return opstring


