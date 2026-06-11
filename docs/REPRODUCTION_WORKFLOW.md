# Reproduction Workflow

The numbered scripts in `scripts/` were used in sequence during manuscript preparation.

## Main Pipeline

1. `00_clean_ipm_metadata.py`
   Clean local metadata and prepare cohort-level availability tables.

2. `01_audit_existing_annotation_coverage.py`
   Audit available genome annotation sources.

3. `02_train_eval_ipm_mem_unified_predictability.py`
   Build and evaluate the locked public-training-only IPM/MEM prediction layer.

4. `03_build_ipm_mem_mechanism_predictability_tables.py`
   Join prediction errors with first-pass mechanism strata.

5. `04_analyze_ipm_mem_predictability_statistics.py`
   Perform subtype-level predictability statistics.

6. `05_assign_local_mlst_from_pubmlst.py`
   Assign local ST labels using PubMLST-compatible loci.

7. `06_st_sensitivity_predictability_analysis.py`
   Evaluate whether paired IPM/MEM error patterns are dominated by sequence type.

8. `07_export_predictability_figure_data.py`
   Export main figure data.

9. `08_deepen_regulator_ampc_variant_annotation.py`
   Generate deep OprD, efflux-regulator, and AmpC-axis annotations.

10. `09_high_confidence_mechanism_statistics.py`
    Build high-confidence subtype performance summaries.

11. `10_clinical_warning_rule_analysis.py`
    Derive clinical warning-rule summaries.

12. `11_enrich_clinical_error_analyses.py`
    Generate calibration, false-susceptible, and safety-gate analyses.

13. `12_prepare_web_plotting_data_package.py` and `13_prepare_figure_excel_data_package.py`
    Prepare figure-source data packages.

14. `15_draw_main_figures_editable_pdf.py` and `16_draw_supplementary_figures_editable_pdf.py`
    Draw editable PDF figure drafts.

15. `17_prepare_manuscript_tables.py`, `18_write_methods_results_manuscript_docs.py`, and `19_write_nc_style_supplementary_material.py`
    Generate manuscript and supplementary table drafts.

16. `20_prepare_full_feature_web_assets.py`
    Prepare assets for the full-feature web deployment.

## Notes

The scripts assume the original project data layout and local/private inputs. Raw local clinical files and local assemblies are not bundled in this code package.
