# Experiments 实验脚本入口（PRISM 主线 + 7 方法对比 + 论证 PPT 证据）

本目录按业务分 6 子目录组织（参照 LA_perfect 风格）。每个脚本对应一个明确的实验场景，可直接从仓库根目录运行：`python experiments/<subdir>/run_xxx.py --help`。

主线方法 **PRISM**（Physics-Regularized Iterative Spectral Mixing，详见 [`../docs/prism_method.md`](../docs/prism_method.md)），其他方法（NNLS / OLS / FCLS / NMF / MCR-ALS）作为对比基线。**NNLS 是 PRISM 在 `weight_mode=uniform, λ_L2=0, tv_iters=0` 时的退化形式。**

---

## 按业务分组的子目录索引

### `unmixing_runs/` — 解混入口（4 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_real_unmixing_single.py` | 单张真实面扫图上跑 OLS/NNLS/FCLS/NMF | `outputs/real_unmixing_single/` |
| `run_real_method_comparison.py` | 单张图上四方法详细对比（含图） | `outputs/real_method_comparison/` |
| `run_batch_method_comparison.py` | 多张典型图四方法批量对比汇总 | `outputs/batch_method_comparison/` |
| `run_synthetic_method_comparison.py` | 合成真值数据上跑四方法定量对比 | `outputs/synthetic_method_comparison/` |

### `preprocessing_runs/` — 预处理协议对比（2 个，论点② ALS+L2 选型）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_real_preprocessing_comparison.py` | 单张图上对比三协议（als_l2/als_max/none_l2） | `outputs/real_preprocessing_comparison/` |
| `run_batch_preprocessing_comparison.py` | 多张图上批量对比三协议 | `outputs/batch_preprocessing_comparison/` |

### `diagnostics/` — 诊断/汇总（5 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_endmember_fingerprint_plot.py` | 三端元纯谱叠加 + 文献指纹峰标注（物理基础页） | `outputs/experiments/endmember_fingerprint/` |
| `run_method_constraint_diagnostics.py` | 逐像素负丰度率 / NMF 端元 SAM / NNLS 稀疏度 | `outputs/experiments/method_constraint_diagnostics/` |
| `run_protocol_consistency_analysis.py` | 三协议下逐像素 CV + 指纹峰保留率 | `outputs/experiments/protocol_consistency/` |
| `run_mcr_als_check.py` | hard-constrained（端元锁死 = NNLS）+ semi-blind（端元漂移到 Pearson r ≈ 0） | `outputs/experiments/mcr_als_check/` |
| `run_overall_summary.py` | **7 方法 × 9 指标 × 多数据集** 横向总表（论文 §3 引用单一事实源） | `outputs/showcase/method_comparison/method_overall_summary.csv` |

### `prism_tuning/` — PRISM 调参/消融/真实样本检查（7 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_prism_quick_check.py` | 合成数据上 PRISM vs NNLS 快速验证 | `outputs/experiments/prism_quick_check/` |
| `run_prism_param_sweep.py` | 34 配置超参网格扫描（`lambda_l2 × lambda_tv × tv_iters × weight_mode`） | `outputs/experiments/prism_param_sweep/` |
| `run_prism_ablation.py` | PRISM 2^3 三组件消融脚本（波段加权 / L2 / TV anchor 单独和两两组合） | `outputs/showcase/method_comparison/prism_ablation_summary.csv` |
| `run_prism_synth_std_vs_uni.py` | 加权策略消融（uniform vs endmember_std） | `outputs/experiments/prism_synth_std_vs_uni/` |
| `run_prism_real_check.py` | 3 个真实样本上 PRISM vs NNLS 对比 + 丰度图 | `outputs/experiments/prism_real_check/` |
| `run_prism_absent_check.py` | absent_load 物理一致性测试（"不应有 PE"的假阳性率） | `outputs/experiments/prism_absent_check/` |
| `run_prism_abundance_viz.py` | 合成数据 PRISM vs NNLS 丰度图阵列可视化 | `outputs/experiments/prism_abundance_viz/` |

### `generalization/` — 跨淀粉源泛化（1 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_generalization_batch.py` | 跨淀粉来源（展艺/新良/甘汁园）泛化批量评估 | `outputs/generalization_batch/` |

### `plotting/` — 纯绘图入口（4 个，读 csv → 调 visualization 函数）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_synthetic_metric_plot.py` | 合成真值 MAE/RMSE/R² 三联子图 | `outputs/showcase/synthetic_method_comparison/` |
| `run_prism_ablation_plot.py` | PRISM 2³ 消融三联子图（MAE/Pearson r/spatial TV） | `outputs/showcase/method_comparison/prism_ablation_bars.png` |
| `run_absent_load_plot.py` | absent-load 假阳性柱状图（NNLS vs PRISM） | `outputs/showcase/method_comparison/absent_load_false_positive_bars.png` |
| `run_prism_convergence_plot.py` | tv_iters trade-off 收敛曲线 | `outputs/showcase/prism_convergence/` |

---

## 推荐运行顺序（论文写作场景）

```text
1. 单图调试   → unmixing_runs/run_real_unmixing_single.py / run_real_method_comparison.py
2. 多图批量   → unmixing_runs/run_batch_method_comparison.py
3. 合成真值   → unmixing_runs/run_synthetic_method_comparison.py + plotting/run_synthetic_metric_plot.py
4. PRISM 主线 → prism_tuning/run_prism_quick_check.py → run_prism_param_sweep.py → run_prism_real_check.py
5. PRISM 物理 → prism_tuning/run_prism_absent_check.py + run_prism_synth_std_vs_uni.py + plotting/run_prism_convergence_plot.py
6. MCR-ALS    → diagnostics/run_mcr_als_check.py（含 hard / semi-blind 双跑）
7. 泛化       → generalization/run_generalization_batch.py
8. PPT 证据   → diagnostics/run_endmember_fingerprint_plot.py / run_method_constraint_diagnostics.py / run_protocol_consistency_analysis.py
9. 汇总总表   → diagnostics/run_overall_summary.py（论文 §3 表 1 数据来源）
```

---

## import 约定

所有脚本在子目录里通过 `Path(__file__).resolve().parents[2]` 把仓库根目录加入 `sys.path`。脚本内部统一使用顶层职责包：

- `preprocessing.*`
- `synthetic.*`
- `unmixing.*`（含 `prism_unmix_spectra` / `mcr_als_unmix_spectra` / `unmix_spectra` / `blind_nmf_unmix_spectra`）
- `visualization.*`（含 `apply_paper_style` / `save_figure` / `COLORS` / 各 `plot_*` 函数）
- `utils.*`
