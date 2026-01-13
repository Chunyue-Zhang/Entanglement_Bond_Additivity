import math
import numpy as np
import os
import pickle
import time
import uuid
from quspin.basis import spin_basis_1d  # Hilbert space spin basis


def apply_2_qubits_gate(
    initial_vec,  # 全Hilbert空间中的初态，1维numpy数组形式
    two_qubits_gate,  # 所施加的双量子比特酉门，2维numpy数组表示的矩阵
    bond_idx,  # 双量子比特酉门所施加的bond的index，0,1,2,……,num_qubits-1
):
    num_qubits = round(np.log2(len(initial_vec)))
    initial_tsr = initial_vec.reshape([2] * num_qubits)
    gate_tsr = two_qubits_gate.reshape(2, 2, 2, 2)
    two_qubits_idxs = [bond_idx, (bond_idx + 1) % num_qubits]
    temp_tsr = np.tensordot(
        gate_tsr, initial_tsr, axes=([2, 3], two_qubits_idxs)
    )  # 通过张量缩并实现门的作用
    final_tsr = np.moveaxis(
        temp_tsr, [0, 1], two_qubits_idxs
    )  # 缩并后产生的新轴会在最前面的位置，需要移动到它们原来的位置
    return final_tsr.reshape(-1)  # 输出全Hilbert空间中的末态，1维numpy数组形式


def calculate_EE(
    basis,
    psi,  # basis下的自旋链的态，1维numpy数组形式
    subsys_spin_idx,  # subsystem spin index，元组形式
):
    psi_norm = psi / np.linalg.norm(psi)
    EE = basis.ent_entropy(psi_norm, subsys_spin_idx, density=False)["Sent_A"]
    return EE


# 系统基本设置
# 基本参数
##整型----------
L = 16  # 偶数
##浮点型----------
alpha = 90.0
beta = 180.0
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


# 演化
# 基本参数
##整型----------
D = 3000  # circuit的总深度
##浮点型----------

##其他（包括数据类型不确定的变量）----------
selection_depth_list = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
    2300,
    2400,
    2500,
    2600,
    2700,
    2800,
    2900,
    3000,
]
# 衍生变量：其中下划线开头的表示生成衍生变量时所需的临时中间变量，这些变量不需要保存到.pkl文件中

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
bond_idx_by_depth = []  # 用于存储random circuit构型
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
    "alpha",
    "beta",
    "BC",
    "num_particles",
    "pauli",
    "sector_dim",
    "D",
    "selection_depth_list",
    "HC_subsys_spin_idx",
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
"""
关于用高级索引(Advanced Indexing)进行
赋值操作：
a[indices] = b
和
切片操作：
b = a[indices]
时a与b相互独立的demo代码：
import numpy as np

indices = [0, 2, 4]

a = np.arange(5)
b = np.array([11, 22, 33])
a[indices] = b#赋值操作
print('a=\n',a,'\n',sep='')
print('b=\n',b,'\n',sep='')
b[0] = 99
print('a=\n',a,'\n',sep='')
print('b=\n',b,'\n',sep='')

a = np.arange(5)
indices = [0, 2, 4]
b = a[indices]#切片操作
print('a=\n',a,'\n',sep='')
print('b=\n',b,'\n',sep='')
b[0] = 99
print('a=\n',a,'\n',sep='')
print('b=\n',b,'\n',sep='')
"""
for i in range(num_pre_samples):
    print(i)
    # 随机构型
    p0vi = rng.integers(low=0, high=sector_dim)
    bibd = rng.integers(low=0, high=num_bonds, size=D)
    # 计算
    psi = np.zeros(sector_dim)
    psi[p0vi] = 1.0
    EE = np.zeros(len(selection_depth_list))
    psi_whole = np.zeros(basis_whole.Ns, dtype=np.complex128)
    psi_whole[sector_index_in_whole] = psi
    for ii in range(D):
        psi_whole = apply_2_qubits_gate(
            initial_vec=psi_whole,  # 全Hilbert空间中的初态，1维numpy数组形式
            two_qubits_gate=U,  # 所施加的双量子比特酉门，2维numpy数组表示的矩阵
            bond_idx=bibd[
                ii
            ],  # 双量子比特酉门所施加的bond的index，0,1,2,……,num_qubits-1
        )
        if ii + 1 in selection_depth_list:
            EE[selection_depth_list.index(ii + 1)] = calculate_EE(
                basis,
                psi=psi_whole[
                    sector_index_in_whole
                ],  # basis下的自旋链的态，1维numpy数组形式
                subsys_spin_idx=HC_subsys_spin_idx,  # subsystem spin index，元组形式
            )
    # 收集数据
    psi0_vec_idx.append(p0vi)
    bond_idx_by_depth.append(bibd)
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
