---
title: Modified DNA Base RESP Library
date: 2026-10-05
tags:
  - modified-DNA
  - AMBER
  - RESP
status: initial-MD-topology-library
related:
  - "[[O6MeG Gaussian Parameterization]]"
  - "[[O6MeG AMBER Parameterization]]"
  - "[[DNA Workflow GUI]]"
---

# 非标准 DNA 碱基 RESP 与 MD 拓扑库

当前已有的真实条目是 **O6MeG**。在已完成的AmberTools RESP电荷基础上，现已补齐OL21兼容的残基模板、局部类比参数及技术验证，并接入工作台GUI修饰入口。没有追加QM扫描或超算提交脚本。其他修饰完成各自电荷和拓扑构建后再加入`catalog.json`，不预填虚构参数。

## 当前条目

| 修饰 | 电荷来源 | 主干来源 | 状态 |
|---|---|---|---|
| O6MeG，O⁶-甲基鸟嘌呤 | 两个已计算构象共同进行两阶段 AmberTools RESP | AMBER DNA OL21 DG/DG5/DG3/DGN | 四种MD模板完成，技术验证通过；类比参数初版 |

入口是 `catalog.json`；具体记录在 `O6MeG/entry.json`。JSON 中保留化学身份、命名、键连接、拟合方法、构象来源、误差、源结果路径及适用范围。

完整MD库文件及实际使用方法见 [[05 Code/modified_base_library/O6MeG/README|O6MeG OL21 MD拓扑库]]。可加载文件为`O6MeG.lib`、`O6MeG.frcmod`和`leaprc.O6MeG`，另有`O6MeG_OL21_MD_LIBRARY.zip`便于拷贝。

## O6MeG 可直接查用的文件

| 文件 | 内容 |
|---|---|
| `O6MeG/base_charges.csv` | 修饰碱基组18个原子的拟合电荷 |
| `O6MeG/internal_residue_charges.csv` | 恢复 OL21 糖/磷酸主干后，内部 DNA 残基36个原子的候选电荷 |
| `O6MeG/nucleoside_charges.csv` | 原始中性封端核苷35个原子的完整拟合结果；含封端OH氢 |
| `O6MeG/dG_control_comparison.csv` | 正常dG控制的RESP结果与OL21碱基电荷比较 |
| `O6MeG/fit_quality.json` | A、B和dG的电势拟合误差 |
| `O6MeG/nucleoside_A.xyz`、`nucleoside_B.xyz` | 两个实际优化核苷几何；不是内部DNA残基的完整结构模板 |
| `O6MeG/stage1.*`、`stage2.*`、`initial.qin` | 联合RESP输入、原始日志和实际qout；第二阶段读取第一阶段qout |

这些是带原子名的电荷表，导入时按名称匹配，不能按其他软件的原子行顺序直接套用。

## 本次拟合结果

Gaussian电势是气相 HF/6-31G(d)、6D/10F、原生MK；优化用 B3LYP-D3BJ/6-31G(d,p)。真实 AmberTools `espgen` 和 `resp` 已运行，不是合成测试结果。

固定 DGN 糖/OH 17个原子，A/B共享同一套电荷，氨基和甲基氢施加等价约束。电荷约束、两阶段收敛和导出检查通过。

| 项目 | 结果 |
|---|---:|
| dG控制 ESP relative RMS | 0.12334 |
| O6MeG A ESP relative RMS | 0.14620 |
| O6MeG B ESP relative RMS | 0.17495 |
| O6MeG 合并 ESP relative RMS | 0.16038 |
| O6MeG 合并电势点数 | 2886 |
| 碱基组实际电荷 | −0.088801 e |
| 恢复DG主干后的内部残基实际电荷 | −1.000001 e |

误差由实际ESP点按 `V=sum(q/r)` 重新计算，距离单位为bohr，合并relative RMS与原生RESP日志一致。保存误差是记录拟合质量，不将约束通过等同于完整力场或MD验证。

保留六位小数RESP结果，未为了强行凑整数而修改单个原子的电荷。−1.000001 e与目标−1 e之间的差异为输出舍入。

## 电荷表的适用范围

内部残基电荷表删除中性核苷的 `HO5'`、`HO3'`，恢复完整DG糖/磷酸电荷，只转移修饰碱基的RESP电荷。O6MeG的N1无氢，O6为C6–O6–CM醚连接。

上述CSV仍是内部DNA位点电荷表。现新增OG6/OG5/OG3/OGN四种MD模板，分别用于内部、5′OH、3′OH、孤立核苷，`MD_topology_ready=true`。局部AMBER/GAFF2类比补参、母体继承、短链连接、实际EM及GROMACS转换已验证；尚未进行新的QM二面角拟合、长期MD或实验精度验证，`production_MD_approved=false`。工作台GUI现可指定链、位置和O6MeG类型，当前只开放OL21兼容入口。

## 原始结果与复用入口

原始Gaussian和完整RESP拟合结果保留于：

`D:\data\DNA DoS\computation\O6MeG_parameterization_20261004\gaussian`

其中 `fits/dG_reference` 是正常对照，`fits/O6MeG_A_000__O6MeG_B_180` 是联合拟合。`check_gaussian.py` 已修正Gaussian日志的 `数字 Fit 电势值` 识别；`test_check_gaussian.py` 是实际日志回归测试。

`export_resp_library.py` 从真实拟合结果导出本库，拒绝覆盖已有O6MeG条目。相关的 [[O6MeG Gaussian Parameterization]] 和 [[O6MeG AMBER Parameterization]] 保留原先起步记录；本文件是本次真实电荷完成后的入库说明。
