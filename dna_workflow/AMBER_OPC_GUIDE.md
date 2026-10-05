---
title: DNA Workflow AMBER OL21 OL24 OPC 分支
date: 2026-09-22
tags: [computation, workflow, molecular-dynamics]
status: validation-in-progress
---

# AMBER OL21 / OL24 + OPC

本分支增加 AmberTools 建模入口，保留原有 CHARMM36 / CHARMM TIP3P 路径。旧任务不自动改力场；切换模型必须创建新任务，不复用旧轨迹或检查点。DoS 数学和绘图代码不变。

## 本机环境

- WSL：Ubuntu-24.04；GROMACS `/opt/gromacs/2025.4/bin/gmx`。
- AmberTools 26.0、ParmEd 4.3.1、Python 3.11：`/home/chenz/.local/share/dna-amber/env`。
- GUI 的 Amber Python：上述目录的 `bin/python`；tleap：`bin/tleap`。
- 参数直接来自该环境的 `dat/leap/cmd/leaprc.DNA.OL21`、`leaprc.DNA.OL24` 和 `leaprc.water.opc`，不手动拼接其他力场参数。
- 当前 OPC 配套离子来自 `frcmod.ionslm_126_opc`（12-6）。没有加入 12-6-4 参数。

## 使用顺序

1. 重启工作台，在新任务的“计算设置”中选择模型，检查 WSL 环境。
2. 显式选择 Amber 支持的盒型；本版本不支持 Amber 菱形十二面体，不能静默替换。可选截角八面体 `octahedron` 或正交盒。界面 padding 单位是 nm，tleap 输入转换为 Å。
3. 生成 DNA 拓扑，再生成 OPC 水盒及 NaCl/KCl。中和离子与指定浓度盐对分别计算，不把“中和成功”当作盐浓度设置完成。
4. 本地仅准备 `em.tpr`。超算完成 EM 后回传，再依次准备 NVT、NPT、MD 的 TPR；每一步仍依赖上一阶段真实坐标/检查点。
5. 轨迹处理、VMD 主导构型选择继续沿用原入口。主导构型再溶剂化继承父任务的 DNA 模型、OPC 和离子模型，再生成独立 EM 子任务。
6. EM 回传后去水去离子，按 DNA manifest 验证原子身份与顺序，再输出 Gaussian 输入。

支持标准 A/C/G/T 单链或双链；现新增 O6MeG 修饰入口，当前仅兼容 OL21 + OPC。其他氧化/甲基化碱基尚未入库，不能用标准 G 代替未支持的修饰。

## O6MeG 修饰入口

1. 在“序列与任务”保留原始A/C/G/T序列，选择“非标准碱基 → O6MeG”。
2. 指定A/B链和链内从5′端开始计数的位置；该位置必须是G。单链只能选择A链。
3. 在计算设置中明确选择OL21 + OPC及兼容盒型，创建新任务。CHARMM、OL24、非G位置、越界位置会被拦截。
4. 构建PDB时插入真实O6MeG化学结构，N1无氢，O6连接甲基；不是把普通G改一个残基名。
5. 新任务在`workflow.json`保存`modifications`，并复制库文件到任务的`modified_base_library/O6MeG`。拓扑、溶剂化、主导构型再溶剂化、轨迹身份恢复和Gaussian导出均保留修饰。

数据库位于`../modified_base_library`；服务端从`catalog.json`读取可用条目。GUI当前编辑一个修饰位点，后端配置支持不重复的多位点列表。参数为真实RESP电荷＋OL21主干＋AMBER/GAFF2局部类比补参，技术检查通过但未完成长期物理精度验证；界面会显示此边界。

不要在A链输入框直接写`[O6MeG]`标签；请使用显式修饰入口，配对链不会自动突变。位于5′/3′OH端时会使用OG5/OG3，内部位点使用OG6。

## 科学边界

- OPC 必须保留四位点、电荷、虚拟位点及刚性水定义；不能用 `spc216.gro` 或 TIP3P 拓扑代替。
- 新分支的 MDP 使用独立 LJ 设置，不继承 CHARMM force-switch；具体参数写入每个任务 MDP，仍需根据体系开展平衡和生产验证。
- 成功 `grompp` 仅说明输入可编译，不证明 AMBER/GROMACS 能量、力或系综完全一致。
- 未完成跨引擎单点能量/力对照和长期稳定性验证前，不将此分支标为生产验证完成；更换力场也不保证消除物理端部 fraying。

## 代码入口

- `dna_workflow.py`：模型配置、MDP、DNA 身份校验和 Gaussian 导出。
- `amber_backend.py`：tleap 建模、OPC 溶剂化、ParmEd 转换。
- `md_pipeline.py`：准备/回传阶段调度。
- `md_extensions.py`：主导构型再溶剂化和 EM 子任务。
- `gui_server.py`、`gui/index.html`：GUI 模型选择及环境检查。

## 验证记录

- 既有/新分支流程测试通过；另外显式执行 `test_workflow.py`，标准单双链、CHARMM 重原子/末端、PDB 回读和 MDP 单位测试通过；5 项 DoS 回归测试通过。
- 真实 OL21 6 bp 测例：379 个 DNA 原子，电荷 −10，10 条链内磷酸骨架连接，Gaussian 输入导出通过。
- GUI 模型切换浏览器检查无 JavaScript 错误。
- 修饰入口：44项单元/流程测试通过；真实3nt单链及3bp双链均通过O6MeG拓扑、OPC溶剂化和EM TPR编译，未执行mdrun；Gaussian输入导出保留修饰。实际浏览器验证A13非G拦截和A14新任务创建。
- 真实 OL21、OL24 6 bp 测例均通过 OPC 溶剂化及 EM、带 POSRES 的 NVT TPR 编译；NVT 仅用于编译测试，不作为平衡后的生产输入，没有跳过用户作业的 EM 回传条件。
- OL24 水盒：5114 个四位点 OPC 水，26 个 Na+、16 个 Cl−，共 20877 位点；10 个 Na+ 用于中和，16 对盐对应盒体积离散化后的目标 0.15 M。
- OL24 代表构型 EM 子任务已通过再次溶剂化和无位置约束 EM TPR 编译。没有执行生产 MD、Gaussian 或大型 DoS 计算。

原始 CHARMM 基线保留在 Git 标签 `baseline-charmm36-20260922`，开发位于 `feature/amber-ol-opc`，尚不合并主分支。
