import math
import matplotlib.pyplot as plt
import os
from itertools import combinations
import textwrap


def print_and_save_params(
    params_to_be_printed,
    filename,  # 输出文件名（前缀）
):
    print_content_lines = []
    wrap_width = 72  # 定义每行的最大字符数，一个典型的等宽字符大约占0.1英寸
    for name in params_to_be_printed:
        value = globals()[name]
        initial_line = f"{name}={value}"  # 构建待打印变量的初始行
        if len(initial_line) > wrap_width:
            prefix = f"{name}="
            wrapped_lines = textwrap.wrap(  # textwrap.wrap会将一个长字符串分割成一系列符合最大宽度要求的短字符串，然后放进一个列表
                str(
                    value
                ),  # 当value是列表时，str将其转换成字符串时会自动在逗号与下一个元素之间加一个空格
                width=wrap_width,
                # break_long_words=False,
                initial_indent=prefix,  # 第一行以"name="开始
                subsequent_indent=" "
                * len(prefix),  # 后续行以相同长度的空格开始，实现对齐
            )
            print_content_lines.extend(wrapped_lines)
        else:
            print_content_lines.append(initial_line)  # 如果行本身不长，就直接添加
    final_print_string = "\n".join(
        print_content_lines
    )  # 将所有行合并成一个最终的字符串
    fig = plt.figure(figsize=(10.0, 100.0))  # 尺寸设得足够大，保存时会裁边
    fig.text(
        0.01,
        0.99,
        final_print_string,
        fontsize=10.0,
        verticalalignment="top",
        horizontalalignment="left",
        fontname="Courier New",
    )
    fig.savefig(
        f"{filename}_params.pdf",
        bbox_inches="tight",  # 裁边。bbox是Bounding Box的缩写，意为“边界框”
        # bbox_inches='tight'是裁边用的，那么，当绘制的内容超出fig的范围时，bbox_inches='tight'能增加fig的范围吗？
        # 问得非常好！这是一个在 matplotlib 使用中非常关键且容易混淆的点。
        # 简短的回答是：是的，bbox_inches='tight' 的核心作用就是调整最终输出文件的边界框（bounding box），
        # 以确保所有绘制的内容（matplotlib称之为 "artists"）都被完整地包含进来，即使这些内容超出了原始 figure 对象定义的尺寸范围。
        # 所以，它并不是在Python脚本中“增加 fig 的范围”（fig.get_size_inches() 的返回值不会变），
        # 而是在保存文件的那一刻，计算一个“恰好”能包住所有内容的新的边界框，并以此为准来生成最终的PDF、PNG等文件。
        pad_inches=0.2,  # 文本内容与页面边界的距离
    )


def create_bipartitions_by_subsys_size(L):
    num_bipartitions_by_size = []
    subsys_spin_idx_list = []  # Store in a list in the form of tuples
    for i in range(L // 2):
        subsys_size = i + 1
        num_bipartitions = (
            math.comb(L, subsys_size) // 2
            if subsys_size == L / 2
            else math.comb(L, subsys_size)
        )
        num_bipartitions_by_size.append(num_bipartitions)
        subsys_spin_idx_list.extend(
            list(combinations(range(L), subsys_size))[:num_bipartitions]
        )
    # data check
    print(
        "sum(num_bipartitions_by_size)==2**L/2-1:\n",
        sum(num_bipartitions_by_size) == 2**L / 2 - 1,
        "\n",
        sep="",
    )
    print(
        "len(subsys_spin_idx_list)==2**L/2-1:\n",
        len(subsys_spin_idx_list) == 2**L / 2 - 1,
        "\n",
        sep="",
    )
    return num_bipartitions_by_size, subsys_spin_idx_list


def subsys_spin_idx_to_set(
    L,
    subsys_spin_idx,  # 元组形式
):
    return {frozenset(subsys_spin_idx), frozenset(set(range(L)) - set(subsys_spin_idx))}


def operate_set_form_bipartition_by_parity(L, set_form_bipartition):
    subsys_spin_idx = list(
        next(iter(set_form_bipartition))
    )  # 从set_form_bipartition中任取一个子系统（iter()创建迭代器，next()从迭代器中获取下一个元素）
    for i, idx in enumerate(subsys_spin_idx):
        subsys_spin_idx[i] = L - 1 - idx
    return {frozenset(subsys_spin_idx), frozenset(set(range(L)) - set(subsys_spin_idx))}


def operate_set_form_bipartition_by_translation(
    L, delta_idx, set_form_bipartition  # 平移的距离
):
    subsys_spin_idx = list(
        next(iter(set_form_bipartition))
    )  # 从set_form_bipartition中任取一个子系统（iter()创建迭代器，next()从迭代器中获取下一个元素）
    for i, idx in enumerate(subsys_spin_idx):
        subsys_spin_idx[i] = (idx + delta_idx) % L
    return {frozenset(subsys_spin_idx), frozenset(set(range(L)) - set(subsys_spin_idx))}


def classify_bipartition_by_symmetry(
    L,
    BC,
    subsys_spin_idx_list,
):
    # 将subsys_spin_idx_list中用子系统表示的bipartition转换成{frozenset,frozenset}的形式
    # 然后将可以通过宇称和平移操作联系起来的bipartition归为同一类，打包成集合，存到列表中
    # 每个对称类中取一个代表性的bipartition，将其subsys_spin_idx按同样的顺序存到列表中
    bipartition_list_by_symmetry = []
    representative_subsys_spin_idx_list = []
    if BC == "OBC":
        for subsys_spin_idx in subsys_spin_idx_list:
            set_form_bipartition = subsys_spin_idx_to_set(L, subsys_spin_idx)
            generated_bipartitions_set = set(
                [
                    frozenset(set_form_bipartition),
                    frozenset(
                        operate_set_form_bipartition_by_parity(L, set_form_bipartition)
                    ),
                ]
            )  # 通过宇称生成bipartition对称类
            if generated_bipartitions_set not in bipartition_list_by_symmetry:
                bipartition_list_by_symmetry.append(generated_bipartitions_set)
                representative_subsys_spin_idx_list.append(subsys_spin_idx)
    elif BC == "PBC":
        for subsys_spin_idx in subsys_spin_idx_list:
            set_form_bipartition = subsys_spin_idx_to_set(L, subsys_spin_idx)
            generated_bipartitions_set = set(
                [
                    frozenset(
                        operate_set_form_bipartition_by_translation(
                            L, delta_idx, set_form_bipartition
                        )
                    )
                    for delta_idx in range(L)
                ]
                + [
                    frozenset(
                        operate_set_form_bipartition_by_translation(
                            L,
                            delta_idx,
                            operate_set_form_bipartition_by_parity(
                                L, set_form_bipartition
                            ),
                        )
                    )
                    for delta_idx in range(L)
                ]
            )  # 通过平移和宇称生成bipartition对称类
            if generated_bipartitions_set not in bipartition_list_by_symmetry:
                bipartition_list_by_symmetry.append(generated_bipartitions_set)
                representative_subsys_spin_idx_list.append(subsys_spin_idx)
    else:
        raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
    return bipartition_list_by_symmetry, representative_subsys_spin_idx_list


def calculate_bipartition_n_array(
    L,
    BC,
    subsys_spin_idx,  # 元组形式
):
    # 诸n_i的定义详见"C:\Users\chuny\Desktop\桌面文件夹和压缩包\笔记本latex（含目录）\科研日记\paper_2\note\paper_2.pdf"
    if BC == "OBC":
        n_array = [0] * L  # \left(n_0,n_1,n_2,\cdots,n_{L-2},n_{L-1}\right)
        n_array[0] = len(subsys_spin_idx)
        for i in range(1, L):  # i充当bond所连接的两自旋间的距离
            for ii in range(L - i):  # L-i是n_i可达的最大值
                if (ii in subsys_spin_idx) != (
                    ii + i in subsys_spin_idx
                ):  # 判断bond所连接的spin分别是否在sssi里
                    n_array[i] += 1
    elif BC == "PBC":
        n_array = [0] * (
            L // 2 + 1
        )  # \left(n_0,n_1,n_2,\cdots,n_{(L-1)/2-1},n_{(L-1)/2}\right) for odd L
        # or \left(n_0,n_1,n_2,\cdots,n_{L/2-1},n_{L/2}\right) for even L
        n_array[0] = len(subsys_spin_idx)
        for i in range(1, L // 2 + 1):  # i充当bond所连接的两自旋间的距离
            for ii in range(
                L // 2 if i == L / 2 else L
            ):  # (L//2 if i==L/2 else L)是n_i可达的最大值
                if (ii in subsys_spin_idx) != (
                    (ii + i) % L in subsys_spin_idx
                ):  # 判断bond所连接的spin分别是否在sssi里
                    n_array[i] += 1
    else:
        raise ValueError(f"Invalid BC value: '{BC}'. Must be 'OBC' or 'PBC'.")
    return tuple(n_array)


# 基本参数
##整型----------
L = 16
##浮点型----------

##其他（包括数据类型不确定的变量）----------
BC = "PBC"


num_bipartitions_by_size, subsys_spin_idx_list = create_bipartitions_by_subsys_size(L)
set_form_bipartition_list = [
    subsys_spin_idx_to_set(L, subsys_spin_idx)
    for subsys_spin_idx in subsys_spin_idx_list
]
# data check
print(
    "len(set_form_bipartition_list)==2**L/2-1:\n",
    len(set_form_bipartition_list) == 2**L / 2 - 1,
    "\n",
    sep="",
)


bipartition_list_by_symmetry, representative_subsys_spin_idx_list = (
    classify_bipartition_by_symmetry(
        L,
        BC,
        subsys_spin_idx_list,
    )
)
# data check
print(
    "sum([len(bipartitions_set) for bipartitions_set in bipartition_list_by_symmetry])==2**L/2-1:\n",
    sum([len(bipartitions_set) for bipartitions_set in bipartition_list_by_symmetry])
    == 2**L / 2 - 1,
    "\n",
    sep="",
)


representative_subsys_spin_idx_dict_by_n_array = {}
for subsys_spin_idx in representative_subsys_spin_idx_list:
    n_array = calculate_bipartition_n_array(L, BC, subsys_spin_idx)
    if (
        n_array in representative_subsys_spin_idx_dict_by_n_array
    ):  # 判断n_array是否是字典representative_subsys_spin_idx_dict_by_n_array中的一个键
        representative_subsys_spin_idx_dict_by_n_array[n_array].append(subsys_spin_idx)
    else:
        representative_subsys_spin_idx_dict_by_n_array[n_array] = [subsys_spin_idx]


N_n_0 = [0] * (L // 2)
M_n_0 = [0] * (L // 2)
degeneracy = [[] for _ in range(L // 2)]  # 不用[0]*(L//2)，避免子列表不相互独立
for (
    n_array
) in (
    representative_subsys_spin_idx_dict_by_n_array
):  # 遍历字典representative_subsys_spin_idx_dict_by_n_array中的所有键
    N_n_0[n_array[0] - 1] += len(
        representative_subsys_spin_idx_dict_by_n_array[n_array]
    )
    M_n_0[n_array[0] - 1] += 1
    degeneracy[n_array[0] - 1].append(
        len(representative_subsys_spin_idx_dict_by_n_array[n_array])
    )


filename = os.path.splitext(os.path.basename(__file__))[0]

params_to_be_printed = [
    "N_n_0",
    "M_n_0",
    "degeneracy",
]
print_and_save_params(params_to_be_printed, filename)
