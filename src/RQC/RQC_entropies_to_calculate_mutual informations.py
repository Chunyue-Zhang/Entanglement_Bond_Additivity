import math
import numpy as np
import os
import pickle
import time
import uuid
from quspin.basis import spin_basis_1d  # Hilbert space spin basis


def apply_2_qubits_gate(
    initial_vec,
    two_qubits_gate,
    bond_idx,
):
    num_qubits = round(np.log2(len(initial_vec)))
    initial_tsr = initial_vec.reshape([2] * num_qubits)
    gate_tsr = two_qubits_gate.reshape(2, 2, 2, 2)
    two_qubits_idxs = [bond_idx, (bond_idx + 1) % num_qubits]
    temp_tsr = np.tensordot(
        gate_tsr, initial_tsr, axes=([2, 3], two_qubits_idxs)
    )
    final_tsr = np.moveaxis(
        temp_tsr, [0, 1], two_qubits_idxs
    )
    return final_tsr.reshape(-1)


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


L = 16
alpha = 90.0
beta = 180.0
BC = "PBC"
num_particles = L // 2  # particle number
pauli = False
sector_dim = math.comb(L, num_particles)  # sector dimension
basis = spin_basis_1d(L, pauli=pauli, Nup=num_particles)
print("sector_dim==basis.Ns:\n", sector_dim == basis.Ns, "\n", sep="")  # data check
#
if BC == "OBC":
    num_bonds = L - 1
elif BC == "PBC":
    num_bonds = L
else:
    raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
#
U = np.array(
    [
        [np.exp(-1j * beta * np.pi / 180 / 4), 0, 0, 0],
        [
            0,
            np.exp(1j * beta * np.pi / 180 / 4) * np.cos(alpha * np.pi / 180 / 2),
            -1j * np.exp(1j * beta * np.pi / 180 / 4) * np.sin(alpha * np.pi / 180 / 2),
            0,
        ],
        [
            0,
            -1j * np.exp(1j * beta * np.pi / 180 / 4) * np.sin(alpha * np.pi / 180 / 2),
            np.exp(1j * beta * np.pi / 180 / 4) * np.cos(alpha * np.pi / 180 / 2),
            0,
        ],
        [0, 0, 0, np.exp(-1j * beta * np.pi / 180 / 4)],
    ]
)
#
basis_whole = spin_basis_1d(L, pauli=pauli)  # whole Hilbert space的basis
sector_index_in_whole = []
for i in range(basis.Ns):
    sector_index_in_whole.append(basis_whole.index(basis[i]))


D = 3000
selection_depth_list = [1, 5, 100, 2000, 2500, 3000]


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


num_pre_samples = 200
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
bond_idx_by_depth = []
entanglement_entropy = []
elapsed_time = 0.0
rng = np.random.default_rng(
    seed
)


###
variables_to_be_stored = [
    "L",
    "alpha",
    "beta",
    "BC",
    "num_particles",
    "pauli",
    "sector_dim",
    "D",
    "selection_depth_list",
    "num_mut_infos",
    "mut_info_related_subsys_spin_idx_list",
    "num_pre_samples",
    "print_update_delta_time",
    "unique_id",
    "seed",
    "filename",
    "psi0_vec_idx",
    "bond_idx_by_depth",
    "entanglement_entropy",
    "elapsed_time",
]
data = {name: globals()[name] for name in variables_to_be_stored}
start_time = time.perf_counter()
start_i = 0
for i in range(num_pre_samples):
    print(i)
    p0vi = rng.integers(low=0, high=sector_dim)
    bibd = rng.integers(low=0, high=num_bonds, size=D)
    psi = np.zeros(sector_dim)
    psi[p0vi] = 1.0
    EE = np.zeros(
        (len(selection_depth_list), len(mut_info_related_subsys_spin_idx_list))
    )
    psi_whole = np.zeros(basis_whole.Ns, dtype=np.complex128)
    psi_whole[sector_index_in_whole] = psi
    for ii in range(D):
        psi_whole = apply_2_qubits_gate(
            initial_vec=psi_whole,
            two_qubits_gate=U,
            bond_idx=bibd[
                ii
            ],
        )
        if ii + 1 in selection_depth_list:
            EE[selection_depth_list.index(ii + 1)] = calculate_bipartitions_EE(
                basis,
                psi=psi_whole[
                    sector_index_in_whole
                ],
                subsys_spin_idx_list=mut_info_related_subsys_spin_idx_list,
            )
    psi0_vec_idx.append(p0vi)
    bond_idx_by_depth.append(bibd)
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
