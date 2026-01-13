import math
import numpy as np
import os
import pickle
import time
import uuid
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators


def create_NN_J_list(L, J, BC):  # NN表示nearest neighbour
    if BC == "OBC":
        num_bonds = L - 1
    elif BC == "PBC":
        num_bonds = L
    else:
        raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
    return [[J, i, (i + 1) % L] for i in range(num_bonds)]


def create_NNN_J_list(L, J, BC):  # NNN表示next nearest neighbour
    if BC == "OBC":
        num_bonds = L - 2
    elif BC == "PBC":
        num_bonds = L
    else:
        raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
    return [[J, i, (i + 2) % L] for i in range(num_bonds)]


def calculate_EE(
    basis,
    psi,  # basis下的自旋链的态，1维numpy数组形式
    subsys_spin_idx,  # subsystem spin index，元组形式
):
    psi_norm = psi / np.linalg.norm(psi)
    EE = basis.ent_entropy(psi_norm, subsys_spin_idx, density=False)["Sent_A"]
    return EE


def calculate_EE_dynamics(
    basis,
    psi0,  # basis下的初态，1维numpy数组形式
    time_array,  # 演化时间，1维numpy数组形式
    fulle,
    fullv,
    subsys_spin_idx,  # subsystem spin index，元组形式
):
    EE = np.zeros(len(time_array))
    utpsi = fullv.conj().T @ psi0.reshape(-1, 1)
    for i, t in enumerate(time_array):
        psi = np.exp(-1j * fulle * t) * utpsi.reshape(-1)
        psi = fullv @ psi.reshape(-1, 1)
        EE[i] = calculate_EE(
            basis,
            psi=psi.reshape(-1),  # basis下的自旋链的态，1维numpy数组形式
            subsys_spin_idx=subsys_spin_idx,  ##subsystem spin index，以元组形式存在列表中
        )
    return EE


# 系统基本设置
# 基本参数
##整型----------
L = 16  # 偶数
##浮点型----------
gamma = 24 / 25
J_perp = 1.0
J_z = 0.5
W = 0.5
##其他（包括数据类型不确定的变量）----------
BC = "PBC"
num_particles = L // 2  # particle number
pauli = False
# 衍生变量：其中下划线开头的表示生成衍生变量时所需的临时中间变量，这些变量不需要保存到.pkl文件中
sector_dim = math.comb(L, num_particles)  # sector dimension
# --------------------------------------------------
# 主循环中需要用到的其他变量
basis = spin_basis_1d(L, pauli=pauli, Nup=num_particles)
print("sector_dim==basis.Ns:\n", sector_dim == basis.Ns, "\n", sep="")  # data check
Jperp_list = create_NN_J_list(  # NN表示nearest neighbour
    L, J=J_perp, BC=BC
) + create_NNN_J_list(  # NNN表示next nearest neighbour
    L, J=gamma * J_perp, BC=BC
)
Jz_list = create_NN_J_list(  # NN表示nearest neighbour
    L, J=J_z, BC=BC
) + create_NNN_J_list(  # NNN表示next nearest neighbour
    L, J=gamma * J_z, BC=BC
)


# 演化
# 基本参数
##整型----------

##浮点型----------

##其他（包括数据类型不确定的变量）----------

# 衍生变量：其中下划线开头的表示生成衍生变量时所需的临时中间变量，这些变量不需要保存到.pkl文件中
time_list = np.concatenate(
    (
        np.logspace(np.log10(1e-3), np.log10(1.0), 12 + 1),
        np.linspace(1.0, 10.0, 36 + 1),
        np.logspace(np.log10(10.0), np.log10(1e5), 16 + 1),
    )
)
# --------------------------------------------------
# 主循环中需要用到的其他变量


# bipartition
# 基本参数
##整型----------

##浮点型----------

##其他（包括数据类型不确定的变量）----------

# 衍生变量：其中下划线开头的表示生成衍生变量时所需的临时中间变量，这些变量不需要保存到.pkl文件中
HC_subsys_spin_idx = tuple(range(L // 2))
# --------------------------------------------------
# 主循环中需要用到的其他变量


# 主循环和文件输出
# 基本参数
##整型----------
num_pre_samples = 200  # 预抽样的数量
##浮点型----------
print_update_delta_time = 3600.0  # 打印进度和更新数据的最短时间间隔，以秒为单位
##其他（包括数据类型不确定的变量）----------

# 衍生变量：其中下划线开头的表示生成衍生变量时所需的临时中间变量，这些变量不需要保存到.pkl文件中
unique_id = uuid.uuid4()  # 生成一个唯一的UUID用于设置随机数种子和命名输出的文件
seed = unique_id.int  # UUID转换成的128二进制位的整数，作为随机数种子
_py_filename = os.path.splitext(os.path.basename(__file__))[
    0
]  # 用程序文件名和UUID去掉横杠后的十六进制整数（比十进制简短些）构建唯一的输出文件名（前缀）
filename = (
    _py_filename + f"_{seed:032x}"
)  # 使用f-string将十进制的seed整数格式化为32位的、带前导零的十六进制字符串
print("filename=", filename)
# 存数据的列表和程序运行时间的初始值
psi0_vec_idx = []
disorder = []  # 用于存储disorder构型
entanglement_entropy = []
elapsed_time = 0.0
# --------------------------------------------------
# 主循环中需要用到的其他变量
rng = np.random.default_rng(
    seed
)  # 使用现代、安全的方式创建RNG实例，利用其背后的SeedSequence机制处理并充分利用二进制位数高达128的整数种子


###
variables_to_be_stored = [
    "L",
    "gamma",
    "J_perp",
    "J_z",
    "W",
    "BC",
    "num_particles",
    "pauli",
    "sector_dim",
    "time_list",
    "HC_subsys_spin_idx",
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
    # 随机构型
    p0vi = rng.integers(low=0, high=sector_dim)
    hs = rng.uniform(low=-W, high=W, size=(L,))
    # 计算
    psi0_vector = np.zeros(sector_dim)
    psi0_vector[p0vi] = 1.0
    h_list = [[hs[ii], ii] for ii in range(L)]
    h_hamiltonian = hamiltonian(
        [["xx", Jperp_list], ["yy", Jperp_list], ["zz", Jz_list], ["z", h_list]],
        [],
        basis=basis,
        dtype=np.float64,
    )
    fulle, fullv = h_hamiltonian.eigh()
    EE = calculate_EE_dynamics(
        basis,
        psi0=psi0_vector,  # basis下的初态，1维numpy数组形式
        time_array=time_list,  # 演化时间，1维numpy数组形式
        fulle=fulle,
        fullv=fullv,
        subsys_spin_idx=HC_subsys_spin_idx,  # subsystem spin index，元组形式
    )
    # 收集数据
    psi0_vec_idx.append(p0vi)
    disorder.append(hs)
    entanglement_entropy.append(EE)
    # 打印进度和更新数据
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
            elapsed_time  # elapsed_time是不可变对象，字典data中存储的值是其副本而不是引用，需要手动更新
        )
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(data, f)
