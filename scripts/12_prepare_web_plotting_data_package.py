#!/usr/bin/env python3
"""Prepare a clean web-plotting data package for manuscript figures."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"

FIGDATA = PROJECT / "results" / "07_figure_data"
DEEP = PROJECT / "results" / "08_deep_mechanism_annotation"
ENRICHED = PROJECT / "results" / "11_enriched_clinical_error_analyses"
MODEL = PROJECT / "models" / "ipm_mem_unified_public_only"
OUT = PROJECT / "results" / "12_plotting_data_package"


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "main": OUT / "main_figures",
        "supp": OUT / "supplementary_figures",
        "tables": OUT / "tables_for_reference",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_csv(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_csv(frame: pd.DataFrame, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dst, index=False)
    return dst


def add_manifest(
    rows: list[dict[str, str]],
    figure: str,
    panel: str,
    title: str,
    plot_type: str,
    data_file: Path,
    source_file: str,
    key_message: str,
    destination: str,
) -> None:
    rows.append(
        {
            "destination": destination,
            "figure": figure,
            "panel": panel,
            "panel_title": title,
            "recommended_plot_type": plot_type,
            "data_file": str(data_file.relative_to(OUT)).replace("\\", "/"),
            "source_file": source_file,
            "key_message": key_message,
        }
    )


def study_flow_tables(dirs: dict[str, Path], manifest: list[dict[str, str]]) -> None:
    nodes = pd.DataFrame(
        [
            {"node_id": "public_training", "label": "Public training genomes", "stage": "training", "order": 1},
            {"node_id": "unified_model", "label": "Unified IPM/MEM model", "stage": "model", "order": 2},
            {"node_id": "locked_public_external", "label": "Locked public external validation", "stage": "validation", "order": 3},
            {"node_id": "locked_local_validation", "label": "Locked local clinical validation", "stage": "validation", "order": 4},
            {"node_id": "mechanism_annotation", "label": "High-confidence mechanism annotation", "stage": "mechanism", "order": 5},
            {"node_id": "clinical_warning", "label": "Clinical warning strata", "stage": "clinical", "order": 6},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "public_training", "target": "unified_model", "edge_label": "train only"},
            {"source": "unified_model", "target": "locked_public_external", "edge_label": "locked evaluation"},
            {"source": "unified_model", "target": "locked_local_validation", "edge_label": "locked evaluation"},
            {"source": "locked_local_validation", "target": "mechanism_annotation", "edge_label": "paired IPM/MEM + WGS"},
            {"source": "mechanism_annotation", "target": "clinical_warning", "edge_label": "safety interpretation"},
        ]
    )
    node_path = write_csv(nodes, dirs["main"] / "Figure1" / "Fig1A_study_flow_nodes.csv")
    edge_path = write_csv(edges, dirs["main"] / "Figure1" / "Fig1A_study_flow_edges.csv")
    add_manifest(
        manifest,
        "Figure 1",
        "Fig 1A",
        "Study design flow",
        "flow diagram / sankey-like workflow",
        node_path,
        "derived manually from study design",
        "Public-training-only model evaluated in locked public and local validation cohorts.",
        "main",
    )
    add_manifest(
        manifest,
        "Figure 1",
        "Fig 1A edges",
        "Study design flow edges",
        "flow diagram edge table",
        edge_path,
        "derived manually from study design",
        "Shows independence between training and locked validation cohorts.",
        "main",
    )


def mechanism_framework_table(dirs: dict[str, Path], manifest: list[dict[str, str]]) -> Path:
    frame = pd.DataFrame(
        [
            {
                "step_order": 1,
                "framework_layer": "Evidence inputs",
                "item": "OprD severe loss",
                "definition": "oprD truncation, disruption, large indel, or short OprD length",
                "interpretation_role": "primary high-confidence carbapenem-resistance driver",
                "assigned_subtype_if_present": "OprD-loss or high-confidence composite",
            },
            {
                "step_order": 2,
                "framework_layer": "Evidence inputs",
                "item": "Acquired carbapenemase",
                "definition": "strict acquired carbapenemase gene signal such as VIM/IMP/NDM/KPC/SPM/GIM or carbapenemase-associated GES",
                "interpretation_role": "composite driver when co-occurring with OprD loss or another high-confidence class",
                "assigned_subtype_if_present": "High-confidence composite",
            },
            {
                "step_order": 3,
                "framework_layer": "Evidence inputs",
                "item": "AmpC-axis disruptive driver",
                "definition": "high-confidence disruptive variant in core AmpC regulatory/cell-wall axis genes",
                "interpretation_role": "exploratory high-confidence driver because sample size is small",
                "assigned_subtype_if_present": "AmpC-axis disruptive or high-confidence composite",
            },
            {
                "step_order": 4,
                "framework_layer": "Evidence inputs",
                "item": "Efflux-driver disruptive genotype",
                "definition": "strict disruptive signal in selected efflux regulatory genes",
                "interpretation_role": "composite driver when combined with OprD loss or another high-confidence class",
                "assigned_subtype_if_present": "High-confidence composite",
            },
            {
                "step_order": 5,
                "framework_layer": "Priority assignment",
                "item": "Subtype hierarchy",
                "definition": "high-confidence composite > OprD-loss > AmpC-axis disruptive > no high-confidence driver",
                "interpretation_role": "prevents broad PAO1-relative missense burden from defining final driver labels",
                "assigned_subtype_if_present": "Final high-confidence mechanism subtype",
            },
            {
                "step_order": 6,
                "framework_layer": "Prediction interpretation",
                "item": "Drug-specific trust/warning strata",
                "definition": "apply the same public-training model to IPM and MEM, then interpret errors by high-confidence mechanism subtype",
                "interpretation_role": "separate predictable IPM backgrounds from MEM underprediction warning backgrounds",
                "assigned_subtype_if_present": "Clinical interpretation rule",
            },
        ]
    )
    dst = write_csv(frame, dirs["main"] / "Figure2" / "Fig2D_mechanism_framework.csv")
    add_manifest(
        manifest,
        "Figure 2",
        "Fig 2D",
        "High-confidence mechanism framework",
        "workflow/table schematic",
        dst,
        "derived manually from locked local mechanism annotation rules",
        "Defines the evidence layers and priority rules used before subtype-specific IPM/MEM error analysis.",
        "main",
    )
    return dst


def target_gene_burden(dirs: dict[str, Path], manifest: list[dict[str, str]]) -> Path:
    src = DEEP / f"target_gene_variant_summary_{DATE}.tsv"
    frame = pd.read_csv(src, sep="\t", low_memory=False)
    frame["has_any_variant"] = frame["variant_count"].gt(0)
    frame["has_non_syn_or_regulatory"] = frame["n_non_syn_or_regulatory"].gt(0)
    frame["has_disruptive_or_uncertain"] = frame["severity"].isin(
        ["disruptive", "coding_indel_uncertain"]
    )
    burden = (
        frame.groupby(["axis", "gene", "locus_tag", "strict_driver"], as_index=False)
        .agg(
            isolates=("genome_id", "nunique"),
            isolates_with_any_variant=("has_any_variant", "sum"),
            isolates_with_non_syn_or_regulatory=("has_non_syn_or_regulatory", "sum"),
            isolates_with_disruptive_or_uncertain=("has_disruptive_or_uncertain", "sum"),
            total_variants=("variant_count", "sum"),
        )
        .sort_values(["axis", "gene"])
    )
    for col in [
        "isolates_with_any_variant",
        "isolates_with_non_syn_or_regulatory",
        "isolates_with_disruptive_or_uncertain",
    ]:
        burden[f"{col}_pct"] = burden[col] / burden["isolates"] * 100.0
    dst = write_csv(burden, dirs["main"] / "Figure2" / "Fig2G_target_gene_variant_burden.csv")
    add_manifest(
        manifest,
        "Figure 2",
        "Fig 2G",
        "Target-gene PAO1-relative variation burden",
        "grouped bar / dot plot",
        dst,
        str(src.relative_to(PROJECT)).replace("\\", "/"),
        "Broad PAO1-relative missense variation is too common to define final mechanism drivers.",
        "main",
    )
    return dst


def build_package() -> pd.DataFrame:
    dirs = ensure_dirs()
    manifest: list[dict[str, str]] = []

    mechanism_framework_table(dirs, manifest)

    main_copies = [
        ("Figure 2", "Fig 2A", "Model-development and locked-evaluation cohorts", "grouped bar", "fig1a_dataset_readiness", "Figure2/Fig2A_dataset_readiness.csv", "Shows public-training, locked public external, and local clinical validation phenotype availability."),
        ("Figure 2", "Fig 2B", "Locked model performance", "grouped bar + point labels", "fig1b_unified_ipm_mem_performance", "Figure2/Fig2B_locked_model_performance.csv", "MEM performance is poorer than IPM in locked local validation."),
        ("Figure 2", "Fig 2C", "Local IPM/MEM phenotype groups", "bar chart", "fig1c_local_ipm_mem_pair_groups", "Figure2/Fig2C_local_ipm_mem_phenotype_groups.csv", "Defines paired local phenotype context."),
        ("Figure 2", "Fig 2E", "High-confidence mechanism counts", "bar chart", "fig2d_local_high_confidence_subtype_counts", "Figure2/Fig2E_high_confidence_subtype_counts.csv", "High-confidence OprD-loss and composite strata define main mechanism groups."),
        ("Figure 2", "Fig 2F", "High-confidence mechanism evidence matrix", "heatmap / oncoprint", "fig2e_local_high_confidence_evidence_matrix", "Figure2/Fig2F_high_confidence_evidence_matrix.csv", "Per-isolate mechanism evidence supporting high-confidence subtype calls."),
        ("Figure 3", "Fig 3A/B", "High-confidence predictability metrics", "bar/dot plot by drug and subtype", "fig3c_high_confidence_predictability_metrics", "Figure3/Fig3AB_high_confidence_predictability_metrics.csv", "IPM is predictable but MEM is underpredicted in OprD-loss/composite backgrounds."),
        ("Figure 3", "Fig 3C/D", "High-confidence error points", "strip/scatter plot", "fig3d_high_confidence_error_points", "Figure3/Fig3CD_high_confidence_error_points.csv", "Point-level errors show MEM underprediction in warning strata."),
        ("Figure 3", "Fig 3E", "Observed versus predicted MIC shift", "dumbbell / paired point plot", "fig3e_high_confidence_observed_predicted_mic_shift", "Figure3/Fig3E_observed_predicted_mic_shift.csv", "MEM OprD-loss and composite strata show a large downward predicted-versus-observed MIC shift."),
        ("Figure 3", "Fig 3F", "Large-error direction by subtype", "stacked bar / diverging bar plot", "fig3f_high_confidence_large_error_direction", "Figure3/Fig3F_large_error_direction.csv", "Large MEM errors are dominated by underprediction in mechanism-warning strata."),
        ("Figure 4", "Fig 4A", "Paired high-confidence IPM/MEM error points", "paired dot/line plot", "fig4f_high_confidence_paired_error_points", "Figure4/Fig4A_paired_ipm_mem_error_points.csv", "Within-isolate MEM errors exceed IPM errors."),
        ("Figure 4", "Fig 4B", "Paired high-confidence error summary", "bar/dot plot", "fig4g_high_confidence_paired_error_summary", "Figure4/Fig4B_paired_error_summary.csv", "OprD-loss and composite groups show consistently larger MEM errors."),
        ("Figure 4", "Fig 4C", "Local ST counts", "ranked bar chart", "fig4c_local_st_counts", "Figure4/Fig4C_local_st_counts.csv", "Documents clone/ST composition."),
        ("Figure 4", "Fig 4D", "Leave-one-ST-out sensitivity", "point range/table plot", "fig4d_leave_one_st_out_sensitivity", "Figure4/Fig4D_leave_one_st_out_sensitivity.csv", "MEM-worse signal is not driven by one dominant ST."),
        ("Figure 4", "Fig 4E", "ST-adjusted models", "model coefficient/summary plot", "fig4e_st_adjusted_models", "Figure4/Fig4E_st_adjusted_models.csv", "Mechanism signal persists beyond sparse ST grouping."),
        ("Figure 4", "Fig 4F", "ST-level paired error consistency", "bubble / lollipop plot", "fig4h_st_level_paired_error_consistency", "Figure4/Fig4F_st_level_paired_error_consistency.csv", "Multiple STs show positive MEM-minus-IPM paired error, supporting a non-clonal MEM-worse signal."),
        ("Figure 5", "Fig 5A", "Clinical warning rule table", "tile/table plot", "fig5a_clinical_warning_rule_table", "Figure5/Fig5A_clinical_warning_rule_table.csv", "Translates mechanism strata into trustworthy/warning labels."),
        ("Figure 5", "Fig 5B", "Rule-level performance", "bar/dot plot", "fig5b_clinical_rule_level_performance", "Figure5/Fig5B_rule_level_performance.csv", "MEM warning strata carry high standard VME and underprediction risk."),
        ("Figure 5", "Fig 5C", "MEM large-underprediction enrichment", "forest/dot plot", "fig5c_mem_warning_enrichment", "Figure5/Fig5C_mem_warning_enrichment.csv", "OprD disruptive status enriches MEM large underprediction; standard VME is broadly high among true non-susceptible MEM isolates."),
        ("Figure 5", "Fig 5D", "Breakpoint-zone errors", "stacked bar / dot plot", "fig6c_breakpoint_zone_error_summary", "Figure5/Fig5D_breakpoint_zone_error_summary.csv", "MEM VME is often far above the breakpoint."),
        ("Figure 5", "Fig 5E", "Error taxonomy", "stacked bar / heatmap", "fig6e_error_taxonomy_summary", "Figure5/Fig5E_error_taxonomy_summary.csv", "Dangerous far-from-breakpoint false-susceptible errors dominate MEM failures."),
        ("Figure 5", "Fig 5F optional", "Safety gate evaluation", "coverage vs VME avoided plot", "fig6f_safety_gate_evaluation", "Figure5/Fig5F_safety_gate_evaluation_optional.csv", "Mechanism-warning gate avoids VME but reduces coverage."),
    ]

    for figure, panel, title, plot_type, stem, rel_out, key in main_copies:
        src = FIGDATA / f"{stem}_{DATE}.csv"
        dst = copy_csv(src, dirs["main"] / rel_out)
        add_manifest(
            manifest,
            figure,
            panel,
            title,
            plot_type,
            dst,
            str(src.relative_to(PROJECT)).replace("\\", "/"),
            key,
            "main",
        )

    target_gene_burden(dirs, manifest)
    manifest.sort(
        key=lambda row: (
            row["destination"],
            row["figure"],
            row["panel"],
            row["data_file"],
        )
    )

    supplementary_copies = [
        ("Supplementary Figure S1", "S1A", "Unified model policy", "table plot", MODEL / f"ipm_mem_unified_public_only_policy_{DATE}.csv", "SFig1/S1A_model_policy.csv", "Complete endpoint definitions and shared policy."),
        ("Supplementary Figure S1", "S1B", "Public-training apparent performance", "bar/table plot", MODEL / f"ipm_mem_unified_public_only_training_apparent_summary_{DATE}.csv", "SFig1/S1B_training_apparent_summary.csv", "Training apparent metrics for smoke-check context."),
        ("Supplementary Figure S2", "S2A", "First-pass strict subtype counts", "bar chart", FIGDATA / f"fig2a_local_strict_subtype_counts_{DATE}.csv", "SFig2/S2A_strict_subtype_counts.csv", "First-pass strict mechanism landscape."),
        ("Supplementary Figure S2", "S2B", "Pair group by strict subtype", "stacked bar", FIGDATA / f"fig2b_local_pair_group_by_subtype_{DATE}.csv", "SFig2/S2B_pair_group_by_strict_subtype.csv", "Phenotype groups by first-pass subtype."),
        ("Supplementary Figure S2", "S2C", "Strict evidence matrix", "heatmap / oncoprint", FIGDATA / f"fig2c_local_mechanism_evidence_matrix_{DATE}.csv", "SFig2/S2C_strict_evidence_matrix.csv", "Auditable first-pass strict evidence."),
        ("Supplementary Figure S3", "S3A", "Target-gene variant summary", "gene burden / heatmap source", DEEP / f"target_gene_variant_summary_{DATE}.tsv", "SFig3/S3A_target_gene_variant_summary.tsv", "Full isolate x gene variant severity table."),
        ("Supplementary Figure S3", "S3B", "Target-region variant rows", "variant detail table", DEEP / f"target_variant_rows_{DATE}.tsv", "SFig3/S3B_target_variant_rows.tsv", "Full target-region variant rows."),
        ("Supplementary Figure S4", "S4A", "Strict subtype error points", "strip/scatter plot", FIGDATA / f"fig3b_local_error_points_by_subtype_{DATE}.csv", "SFig4/S4A_strict_subtype_error_points.csv", "Full strict subtype point-level errors."),
        ("Supplementary Figure S4", "S4B", "High-confidence error points", "strip/scatter plot", FIGDATA / f"fig3d_high_confidence_error_points_{DATE}.csv", "SFig4/S4B_high_confidence_error_points.csv", "Full high-confidence point-level errors."),
        ("Supplementary Figure S5", "S5A", "Local ST counts", "ranked bar chart", FIGDATA / f"fig4c_local_st_counts_{DATE}.csv", "SFig5/S5A_local_st_counts.csv", "MLST composition."),
        ("Supplementary Figure S5", "S5B", "Leave-one-ST-out", "point plot", FIGDATA / f"fig4d_leave_one_st_out_sensitivity_{DATE}.csv", "SFig5/S5B_leave_one_st_out.csv", "ST sensitivity analysis."),
        ("Supplementary Figure S5", "S5C", "ST-adjusted models", "model summary plot", FIGDATA / f"fig4e_st_adjusted_models_{DATE}.csv", "SFig5/S5C_st_adjusted_models.csv", "Mechanism vs ST explanatory models."),
        ("Supplementary Figure S6", "S6A", "Calibration summary", "dot/bar plot", FIGDATA / f"fig6a_calibration_summary_{DATE}.csv", "SFig6/S6A_calibration_summary.csv", "Subtype-level probability calibration."),
        ("Supplementary Figure S6", "S6B", "Calibration by probability bin", "calibration plot", ENRICHED / f"calibration_by_probability_bin_{DATE}.csv", "SFig6/S6B_calibration_by_probability_bin.csv", "Binned predicted probability vs observed NS rate."),
        ("Supplementary Figure S6", "S6C", "Susceptible-call risk", "bar plot", FIGDATA / f"fig6b_susceptible_call_risk_{DATE}.csv", "SFig6/S6C_susceptible_call_risk.csv", "False-susceptible risk among predicted susceptible calls."),
        ("Supplementary Figure S7", "S7A", "Safety gate evaluation", "coverage vs VME avoided", FIGDATA / f"fig6f_safety_gate_evaluation_{DATE}.csv", "SFig7/S7A_safety_gate_evaluation.csv", "All gate scenarios."),
        ("Supplementary Figure S7", "S7B", "Error taxonomy summary", "stacked bar / heatmap", FIGDATA / f"fig6e_error_taxonomy_summary_{DATE}.csv", "SFig7/S7B_error_taxonomy_summary.csv", "Full error taxonomy breakdown."),
        ("Supplementary Figure S7", "S7C", "False-susceptible cases", "case table / heatmap source", FIGDATA / f"fig5d_false_susceptible_cases_{DATE}.csv", "SFig7/S7C_false_susceptible_cases.csv", "All VME cases with mechanism context."),
    ]

    for figure, panel, title, plot_type, src, rel_out, key in supplementary_copies:
        dst = copy_csv(src, dirs["supp"] / rel_out)
        add_manifest(
            manifest,
            figure,
            panel,
            title,
            plot_type,
            dst,
            str(src.relative_to(PROJECT)).replace("\\", "/"),
            key,
            "supplementary",
        )

    table_copies = [
        ("Table 1", "cohort_and_locked_evaluation_dataset", FIGDATA / f"fig1a_dataset_readiness_{DATE}.csv"),
        ("Table 1", "cohort_and_locked_evaluation_performance", FIGDATA / f"fig1b_unified_ipm_mem_performance_{DATE}.csv"),
        ("Table 2", "clinical_interpretation_rule_table", FIGDATA / f"fig5a_clinical_warning_rule_table_{DATE}.csv"),
    ]
    for table, label, src in table_copies:
        dst = copy_csv(src, dirs["tables"] / f"{table.replace(' ', '_')}_{label}.csv")
        add_manifest(
            manifest,
            table,
            label,
            label.replace("_", " "),
            "manuscript table source",
            dst,
            str(src.relative_to(PROJECT)).replace("\\", "/"),
            "Reference data for manuscript tables.",
            "tables",
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_path = write_csv(manifest_df, OUT / f"plotting_manifest_{DATE}.csv")
    write_readme(manifest_df, manifest_path)
    return manifest_df


def write_readme(manifest: pd.DataFrame, manifest_path: Path) -> None:
    lines = [
        "# IPM-GPT 网页版绘图数据包",
        "",
        f"日期：{DATE}",
        "",
        "这个目录把正文图和补充图需要的数据从各个分析结果目录中重新整理、重命名，方便网页版直接画图。",
        "",
        "## 目录结构",
        "",
        "- `main_figures/`：推荐正文 Figure 1-5 的 panel 数据。",
        "- `supplementary_figures/`：推荐补充 Figure S1-S7 的 panel 数据。",
        "- `tables_for_reference/`：正文 Table 1-2 的参考数据。",
        f"- `plotting_manifest_{DATE}.csv`：每个 panel 的数据文件、推荐图形类型和 key message。",
        "",
        "## 推荐正文图",
        "",
            "- Figure 1：研究设计流程图。",
            "- Figure 2：数据构成、统一 IPM/MEM 模型整体表现和高可信机制分型框架。",
        "- Figure 3：机制依赖的 IPM/MEM MIC 可预测性。",
        "- Figure 4：同菌株 IPM-vs-MEM 配对误差和 ST 敏感性。",
        "- Figure 5：临床预警规则、VME/ME、安全边界。",
        "",
        "## 使用建议",
        "",
        "1. 先打开 `plotting_manifest_2026-06-05.csv`，按 figure/panel 找对应 CSV。",
        "2. 正文图优先使用 `main_figures/` 下的数据。",
        "3. 如果某个 panel 太拥挤，把该 panel 移到补充图，不要再新增分析。",
        "4. Figure 5F safety gate 是可选正文 panel；如果版面紧，建议放 Supplementary Figure S7。",
        "",
        "## Panel 数量",
        "",
        f"- main panel data files: {int((manifest['destination'] == 'main').sum())}",
        f"- supplementary panel data files: {int((manifest['destination'] == 'supplementary').sum())}",
        f"- table reference files: {int((manifest['destination'] == 'tables').sum())}",
        "",
        "## 注意",
        "",
        "- CSV/TSV 均为画图数据，不是最终图片。",
        "- 对机制小样本结果，例如 AmpC-axis disruptive 和 strict efflux disruptive，图中建议标注 exploratory/small n。",
        "- No high-confidence driver 应写作 mechanism-unresolved，不要写成 no mechanism。",
    ]
    (OUT / f"README_web_plotting_data_package_{DATE}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_package()
    print(f"Wrote {len(manifest)} plotting data entries to {OUT}")


if __name__ == "__main__":
    main()
