---
title: Modified DNA Base RESP Library
date: 2026-10-12
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

当前有 **O6MeG及七种新修饰**。新条目使用已回传的真实Gaussian HF/MK结果进行受约束RESP拟合，继承各自OL21糖/磷酸主干，配合有来源的局部参数。八种条目均已接入工作台修饰入口，包含内部、5′OH、3′OH和孤立核苷四种模板。没有追加QM扫描或超算提交脚本。

## 当前条目

| 修饰 | 电荷来源 | 主干来源 | 状态 |
|---|---|---|---|
| O6MeG，O⁶-甲基鸟嘌呤 | 两个已计算构象共同进行两阶段 AmberTools RESP | AMBER DNA OL21 DG/DG5/DG3/DGN | 四种MD模板完成，技术验证通过；类比参数初版 |
| m5dC，5-甲基胞嘧啶 | 实际HF/MK与两阶段RESP | OL21 DC系列 | MC6/MC5/MC3/MCN |
| hm5dC，5-羟甲基胞嘧啶 | 两个构象共同RESP | OL21 DC系列 | HC6/HC5/HC3/HCN |
| Cl5dC，5-氯胞嘧啶 | 实际HF/MK与两阶段RESP | OL21 DC系列 | LC6/LC5/LC3/LCN；保留Cl元素 |
| dI，脱氧肌苷 | 实际HF/MK与两阶段RESP | OL21 DA系列 | DI6/DI5/DI3/DIN |
| m6dA，N6-甲基腺嘌呤 | 两个构象共同RESP | OL21 DA系列 | MA6/MA5/MA3/MAN |
| oxo8dG，8-氧代鸟嘌呤 | anti/syn共同RESP | OL21 DG系列及公开8OG局部参数 | XG6/XG5/XG3/XGN |
| O6CMdG，O6-羧甲基鸟嘌呤 | 两个构象共同RESP，核苷净电荷−1 | OL21 DG系列 | CG6/CG5/CG3/CGN；内部残基−2e |

新条目均通过真实坐标端基/连接、电荷、27bp双链拓扑、短EM、糖手性、OPC和AMBER/GROMACS同构型能量检查。模板坐标选一个合格构象；不同构象的ESP共同拟合一套共享电荷，并非分别作为数据库条目。

正常dA的QM/RESP对照尚未完成，按用户决定留待后续补齐；dI、m6dA条目已明确记录该限制。`MD_topology_ready=true`仅表示技术可运行，所有条目的`production_MD_approved=false`，不能据此宣称长期构象或实验精度已验证。

工作台按`entry.json`的母体、四模板、对齐锚点和几何文件建模，任务中保存所用库副本。O6-CM的额外−1e参与DNA电荷、离子中和及Gaussian输入，不能再将总电荷等同于磷酸骨架连接数。5′磷酸化、交联及自由基仍不支持。

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
