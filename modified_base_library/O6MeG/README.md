---
title: O6MeG OL21 MD Topology Library
date: 2026-10-05
tags: [modified-DNA, AMBER, RESP, topology]
status: initial-analogy-topology
related:
  - "[[05 Code/modified_base_library/README|非标准碱基库]]"
  - "[[O6MeG Gaussian Parameterization]]"
  - "[[DNA Workflow GUI]]"
---

# O6MeG：可加载的 OL21 初版 MD 拓扑库

本条目已经包含完整的残基模板和力场补充文件，能够生成 AMBER/GROMACS 拓扑。真实 RESP、电荷/化学图检查、短链连接、能量最小化、同构型能量转换以及 OPC 兼容性检查均已完成。

这是 **OL21 主干＋真实 RESP 电荷＋AMBER/GAFF2 局部类比补参** 的初版，不是新做完整QM势能拟合或经实验验证的损伤DNA力场。`MD_topology_ready=true` 表示技术上可用；`production_MD_approved=false` 保留尚未进行长时间/物理精度验证的边界。

## 实际文件

- `O6MeG.lib`：四种残基的原子类型、电荷、共价连接和链连接点。
- `O6MeG.frcmod`：12项局部键、键角、proper/improper补参，不覆盖现有OL21精确参数。
- `leaprc.O6MeG`：加载上述文件及5′/3′端命名映射。
- `atom_types.csv`、`parameter_sources.json`：逐原子类型和逐参数类比来源。
- `entry.json`：适用母体、模板、电荷、验证结果及科学验证状态。
- `base_charges.csv`、`internal_residue_charges.csv`：原始RESP电荷仍保留，未替换成AM1-BCC。

## 四种残基名称

| 名称 | 使用位置 | 原子数 | 实际残基电荷/e | 链连接 |
|---|---|---:|---:|---|
| `OG6` | DNA内部 | 36 | −1.000001 | 前端P，后端O3′ |
| `OG5` | 5′-OH端 | 34 | −0.307901 | 后端O3′ |
| `OG3` | 3′-OH端 | 37 | −0.692101 | 前端P |
| `OGN` | 孤立中性核苷 | 35 | −0.000001 | 无 |

端残基的非整数电荷来自AMBER母体DG5/DG3的电荷分配惯例，不能分别强制改成0或−1。整个5′/3′-OH线性DNA链仍有正确总电荷。六位RESP输出造成约10⁻⁶ e舍入差，未人为修改单原子电荷。5′/3′磷酸化或其他端修饰不在这四个模板范围内。

O6MeG的N1没有氢，O6为C6–O6–CM醚。与DG相比，C2、N1、C6、O6以及新甲基采用对应AMBER类型；所有糖/磷酸类型及电荷均继承同位置OL21母体。使用GAFF2的局部键/角/扭转数值不等于把整个DNA换成GAFF2；LJ和1–4缩放仍由AMBER类型和OL21体系提供。

## 本机准备一个修饰PDB

输入须是干燥、单构型、标准AMBER命名的DNA PDB，不含水/离子。支持单链、双链；指定一个DG位点，选择的是PDB链名和残基编号，不是全局原子编号。程序保留DNA主干，用已优化B构象的碱基按N9/C4/C8对齐插入；它只给MD初始结构，不预测修饰后的主导构象。

现有27bp正常双链的A14示例（WSL中执行）：

```bash
PY=/home/chenz/.local/share/dna-amber/env/bin/python
CODE='/mnt/d/data/obsidian/computation/05 Code/modified_base_library'

"$PY" "$CODE/prepare_o6meg_pdb.py" \
  --input '/mnt/d/data/DNA DoS/DNA rask/test1/dna_27bp/amber_input.pdb' \
  --output '/mnt/d/data/DNA DoS/computation/O6MeG_parameterization_20261004/gaussian/O6MeG_A14_input.pdb' \
  --chain A --resid 14
```

输入不改，输出若存在会拒绝覆盖。原子名O1P/O2P规范化为OP1/OP2；C5M规范化为C7。脚本会依据所选位点的链内位置选择OG6/OG5/OG3/OGN。尚不支持一次修改多位点或直接导入已有其他非标准残基。

## AMBER加载方法

在你的建模工作目录准备以下 `build.in` 内容（PDB改成实际文件名）：

```text
source leaprc.DNA.OL21
source leaprc.O6MeG
source leaprc.water.opc
dna = loadPdb O6MeG_A14_input.pdb
check dna
charge dna
saveAmberParm dna dna.prmtop dna.inpcrd
savePdb dna dna.pdb
quit
```

本机WSL AmberTools加载方式：

```bash
export AMBERHOME=/home/chenz/.local/share/dna-amber/env
MODLIB='/mnt/d/data/obsidian/computation/05 Code/modified_base_library/O6MeG'

"$AMBERHOME/bin/teLeap" \
  -I"$AMBERHOME/dat/leap/cmd" \
  -I"$AMBERHOME/dat/leap/lib" \
  -I"$AMBERHOME/dat/leap/parm" \
  -I"$AMBERHOME/dat/leap/prep" \
  -I"$MODLIB" -f build.in
```

使用teLeap避免本机包装脚本对空格路径的拆分问题。若在别的机器，只需改AMBERHOME和MODLIB，不需要重做RESP或修改lib/frcmod。OPC溶剂化、0.15M盐、EM/NVT/NPT和生产MD仍按原有AMBER流程安排；本条目不自动执行这些生产阶段。

当前仅验证 `leaprc.DNA.OL21`。不把本库自动标记为已验证的OL24拓扑，也不与CHARMM参数混用。已接入DNA工作台GUI：“非标准碱基→O6MeG”，指定链和从5′端计数的位置，选择OL21 + OPC后创建新任务。任务会复制库文件快照，并保留修饰直至Gaussian导出。上面的文件方式仍可使用。

## 已完成的实际验证

| 检查 | 结果 |
|---|---|
| OG6、OG5、OG3、OGN模板加载 | 全部通过 |
| 三核苷酸前后连接和参数完整性 | 全部通过，无缺失键/角/二面角 |
| 四种模型各300步AMBER气相EM | 全部能量下降，共价键长合理 |
| 同构型AMBER/GROMACS总能量 | 最大绝对差0.0594 kJ/mol |
| 27bp双链A14-O6MeG:C干拓扑 | 54残基，1716原子，52条链内磷酸二酯连接，总电荷约−52 e |
| OPC＋中和K⁺短链 | 742个四位点OPC水，3070原子；GROMACS 2025.4成功生成TPR并完成25步EM |
| 自动回归测试 | 3项通过 |

四种短模型的气相EM用于拓扑/转换检查，不是DNA溶液中的结构平衡。OPC的25步EM同样只验证技术兼容性，不代表充分最小化或100ns生产采样。

完整证据在 `validation/`。其中 `duplex27/dna.prmtop`、`dna.inpcrd`、`dna.top`、`dna.gro` 是可查验的目标干体系拓扑；`opc_trimer/em.tpr` 是短链OPC兼容性测试，**不是27bp体系的生产TPR**。

## 原始方法与局部补参来源

RESP输入、两阶段原始日志、Gaussian模型保持在原参数化目录。母体模板和参数来自本机AmberTools：`ff-nucleic-OL21.lib`、`parm10.dat`、`ff-nucleic-OL21.frcmod`；芳香醚局部类比项来自`gaff2.dat`。没有采用parmchk2输出中的ATTN零参数。

每项类比的实际类型组合和数值保存在 `parameter_sources.json`，后续如进行QM精度升级，可以只替换相应局部项，不重做整条DNA主干。

- [AMBER官方：非标准共价残基模板和补参](https://ambermd.org/tutorials/basic/tutorial5/index.php)
- [AMBER官方：固定封端、多构象RESP](https://ambermd.org/tutorials/advanced/tutorial1/section1.php)
