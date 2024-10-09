import scipy
import vqe_methods 
import operator_pools
import pyscf_helper 

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
from pyscf.tools import molden
from pyscf.cc import ccsd

import openfermion as of 
from openfermion import *
from tVQE import *
    
r = 1.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]


charge = 0
spin = 0
basis = 'sto-3g'

[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

print(" n_orb: %4i" %n_orb)
print(" n_a  : %4i" %n_a)
print(" n_b  : %4i" %n_b)

sq_ham = pyscf_helper.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

fermi_ham  = sq_ham.export_FermionOperator()

hamiltonian = of.linalg.get_number_preserving_sparse_operator(fermi_ham,2*n_orb,n_a+n_b,spin_preserving=True)
print(hamiltonian.shape)
s2 = vqe_methods.Make_S2_unrestricted(n_orb,n_a+n_b,C=C,S=S)

#build reference configuration
occupied_list = []
for i in range(n_a):
    occupied_list.append(i*2)
for i in range(n_b):
    occupied_list.append(i*2+1)

print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
reference_ket = scipy.sparse.csc_matrix(of.jw_configuration_state(occupied_list, 2*n_orb)).transpose()

ref = scipy.sparse.csc_matrix(of.linalg.jw_sz_restrict_state(
    of.jw_configuration_state(occupied_list, 2*n_orb).transpose(),
    0,
    n_electrons=n_a+n_b,
    n_qubits=2*n_orb,
    up_index=lambda x : 2*x,
    down_index=lambda x : 2*x+1
)).transpose()
# print(ref)
[e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,1,which='SA',v0=ref.todense())
print(e[0]+E_nuc)

for ei in range(len(e)):
    S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
    print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))

fermi_ham += FermionOperator((),E_nuc)
pyscf.tools.molden.from_mo(mol, "full.molden", sq_ham.C)

#   Francesco, change this to singlet_GSD() if you want generalized singles and doubles
pool = operator_pools.singlet_GSD()
pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)

[e,v,params] = vqe_methods.adapt_vqe(fermi_ham, pool, ref, theta_thresh=1e-9)

print(" Final ADAPT-VQE energy: %12.8f" %e)
print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))

