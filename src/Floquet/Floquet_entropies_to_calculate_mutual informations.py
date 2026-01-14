import math
import numpy as np
import os
import pickle
import time
import uuid
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators
from scipy.linalg import expm


def create_NN_J_list(L, J, BC):
    if BC == "OBC":
        num_bonds = L - 1
    elif BC == "PBC":
        num_bonds = L
    else:
        raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
    return [[J, i, (i + 1) % L] for i in range(num_bonds)]


def calculate_bipartitions_EE(
    basis,
    psi,
    subsys_spin_idx_list,
):
    EE = np.zeros(len(subsys_spin_idx_list))
    psi_norm = psi / np.linalg.norm(psi)
    for i, subsys_spin_idx in enumerate(subsys_spin_idx_list):
        EE[i] = basis.ent_entropy(psi_norm, subsys_spin_idx, density=False)["Sent_A"]
    return EE


def calculate_bipartitions_EE_Floq_dynamics(
    basis,
    psi0,
    num_periods_array,
    fulle,
    fullv,
    fullv_inv,
    subsys_spin_idx_list,
):
    EE = np.zeros((len(num_periods_array), len(subsys_spin_idx_list)))
    utpsi = fullv_inv @ psi0.reshape(-1, 1)
    for i, n in enumerate(num_periods_array):
        psi = (fulle**n) * utpsi.reshape(-1)
        psi = fullv @ psi.reshape(-1, 1)
        EE[i] = calculate_bipartitions_EE(
            basis,
            psi=psi.reshape(-1),
            subsys_spin_idx_list=subsys_spin_idx_list,
        )
    return EE


L = 16
J_perp = 1.0
J_z = 1.0
T_0 = 1.0
T_1 = 2.5
W = 5.0
BC = "PBC"
num_particles = L // 2
pauli = False
sector_dim = math.comb(L, num_particles)
basis = spin_basis_1d(L, pauli=pauli, Nup=num_particles)
print("sector_dim==basis.Ns:\n", sector_dim == basis.Ns, "\n", sep="")  # data check
Jperp_list = create_NN_J_list(L, J=J_perp, BC=BC)
Jz_list = create_NN_J_list(L, J=J_z, BC=BC)
# Floquet
h_hamiltonian = hamiltonian(
    [["xx", Jperp_list], ["yy", Jperp_list]], [], basis=basis, dtype=np.float64
)
floq_1 = expm(-1j * T_1 * h_hamiltonian.toarray())


num_periods_list = [1, 3, 10, 100]


if BC == "OBC":
    num_mut_infos = L - 1
elif BC == "PBC":
    num_mut_infos = L // 2
else:
    raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
mut_info_related_subsys_spin_idx_list = (
    [(0,)]
    + [(i + 1,) for i in range(num_mut_infos)]
    + [(0, i + 1) for i in range(num_mut_infos)]
)


num_pre_samples = 50
print_update_delta_time = 3600.0
unique_id = uuid.uuid4()
seed = unique_id.int
_py_filename = os.path.splitext(os.path.basename(__file__))[
    0
]
filename = (
    _py_filename + f"_{seed:032x}"
)
print("filename=", filename)
psi0_vec_idx = []
disorder = []
entanglement_entropy = []
elapsed_time = 0.0
rng = np.random.default_rng(
    seed
)


###
variables_to_be_stored = [
    "L",
    "J_perp",
    "J_z",
    "T_0",
    "T_1",
    "W",
    "BC",
    "num_particles",
    "pauli",
    "sector_dim",
    "num_periods_list",
    "num_mut_infos",
    "mut_info_related_subsys_spin_idx_list",
    "num_pre_samples",
    "print_update_delta_time",
    "unique_id",
    "seed",
    "filename",
    "psi0_vec_idx",
    "disorder",
    "entanglement_entropy",
    "elapsed_time",
]
data = {name: globals()[name] for name in variables_to_be_stored}
start_time = time.perf_counter()
start_i = 0
for i in range(num_pre_samples):
    print(i)
    p0vi = rng.integers(low=0, high=sector_dim)
    hs = rng.uniform(low=-W, high=W, size=(L,))
    psi0_vector = np.zeros(sector_dim)
    psi0_vector[p0vi] = 1.0
    h_list = [[hs[ii], ii] for ii in range(L)]
    h_hamiltonian = hamiltonian(
        [["zz", Jz_list], ["z", h_list]], [], basis=basis, dtype=np.float64
    )
    Floq = np.diag(np.exp(-1j * T_0 * np.diag(h_hamiltonian.toarray()))) @ floq_1
    fulle, fullv = np.linalg.eig(Floq)
    fullv_inv = np.linalg.inv(fullv)
    EE = calculate_bipartitions_EE_Floq_dynamics(
        basis,
        psi0=psi0_vector,
        num_periods_array=num_periods_list,
        fulle=fulle,
        fullv=fullv,
        fullv_inv=fullv_inv,
        subsys_spin_idx_list=mut_info_related_subsys_spin_idx_list,
    )
    psi0_vec_idx.append(p0vi)
    disorder.append(hs)
    entanglement_entropy.append(EE)
    delta_time = time.perf_counter() - start_time
    if delta_time > print_update_delta_time or i == num_pre_samples - 1:
        elapsed_time += delta_time
        print(f"i={start_i}至i={i}共{i+1-start_i}组样本的运行时间：{delta_time:.2f}秒")
        print(f"i=0至i={i}累计运行时间：{elapsed_time:.2f}秒")
        print(
            f"由此预计约{elapsed_time/(i+1)*(num_pre_samples-i-1):.2f}秒后完全运行完毕"
        )
        start_time = time.perf_counter()
        start_i = i + 1
        data["elapsed_time"] = (
            elapsed_time
        )
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(data, f)
