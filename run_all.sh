#!/usr/bin/env bash
# November pipeline (JCF manuscript): modulator-untreated, age > 5 y (no upper limit), spirometry in t-1, t, t+1.
set -euo pipefail
cd "$(dirname "$0")/scripts"
PY=$(grep ^PY ../.pyenv | cut -d= -f2); RS=$(grep ^RS ../.pyenv | cut -d= -f2)
L=../results/logs; mkdir -p $L
run(){ echo ">> $*"; $PY "$@" > "$L/$(basename $1 .py)${2:+_$3}.log" 2>&1; }
run cftr_variant_map.py
run 00_build_cohort.py
run 00b_data_audit.py
run 21_model_development.py           # index-free clinical model (CLIN) -> used by everything below
run 29_culture_window.py              # co-author review: anchor-year vs two-year windows of P. aeruginosa and MRSA
run 25_candidate_screen.py            # exploratory: added value of single candidates over the four core predictors
for s in 01_table1 03_classification 04_calibration 05_dca 07_gee_associations 08_subgroup 10_shap 11_temporal_validation \
         12_lr5_model_spec 13_missingness 14_comorbidity 22_fev1_forecast 19_extra_validation 20_consistency_checks 24_loco_fev1; do run $s.py; done
run 26_pa_vs_genotype.py             # authors' question: CFTR class genotype instead of P. aeruginosa
run 27_ml_experiments.py             # nested-CV machine-learning experiments (~12 min)
run 28_ml_temporal.py                # 2024 check of those experiments
run 03_classification.py --window 3
run 15_build_tables.py; run 16_figure_data.py; run 23_number_sheet.py
$RS ../R/figures.R
[ -d ../../manuscript ] || { echo "all done (no manuscript folder)"; exit 0; }
cd ../../manuscript && $PY render_jcf.py && $PY export_docx.py MANUSCRIPT_JCF_v1.md SUPPLEMENT_JCF_v1.md ../reports/01_kasim_yanitlar_ve_analizler.md ../reports/02_kasim_kararlar_ve_ml_denemeleri.md ../reports/03_kasim_klinik_makale.md ../reports/05_kultur_penceresi_yanit.md
$PY tripod_checklist.py && $PY export_docx.py TRIPOD_AI_checklist.md   # page/line numbers from the line-numbered manuscript
echo "all done"
