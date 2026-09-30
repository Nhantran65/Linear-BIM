"""Deterministically export compact, aggregate paper evidence from frozen reports.

Run only in the full research workspace. No raw epochs or model weights are copied.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'review_evidence' / 'paper_v1.json'
SOURCES = {
    's20': 'artifacts/frozen/sleep_ulw_author_literal_bim/reports/author-literal-tenfold-2026-08-29/report.json',
    's78': 'artifacts/frozen/sleep_ulw_s78_author/reports/s78-author-full-2026-08-30/report.json',
    's3': 'artifacts/frozen/sleep_cvan_matched_benchmark/reports/cvan-github-matched-2026-09-03/linear-bim/report.json',
    'explain_s20': 'artifacts/frozen/sleep_ulw_bim_explain_88/reports/full-leaky-2026-08-30/report.json',
    'explain_s78': 'artifacts/frozen/sleep_ulw_s78_author/reports/s78-author-full-2026-08-30/bim-explain/report.json',
    'attrib_s20': 'artifacts/frozen/sleep_ulw_bim_explain_88/reports/full-leaky-2026-08-30/attributions.npz',
    'attrib_s78': 'artifacts/frozen/sleep_ulw_s78_author/reports/s78-author-full-2026-08-30/bim-explain/attributions.npz',
    'figure': 'paper/figures/fig2-exact-attribution/matplotlib-v9-serif/figure-data.json',
    'table1_source': 'paper/tables/table-i-main-results-2026-09-08/table-i-main-results.tex',
    'table2_source': 'paper/tables/table-ii-ablation-2026-09-08/table-ii-ablation.tex',
    'paper_pdf': 'Linear_BIM_Paper.pdf',
}


def read(name):
    return json.loads((ROOT / SOURCES[name]).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(row):
    return row['confusion_matrix_true_rows']


def subject_diffs(path):
    with np.load(ROOT / path) as a:
        subjects = a['subjects']
        transitions = a['transition_mask']
        values = a['weighted_innovation_share']
        return [float(values[(subjects == s) & transitions].mean() - values[(subjects == s) & ~transitions].mean())
                for s in np.unique(subjects) if ((subjects == s) & transitions).any() and ((subjects == s) & ~transitions).any()]


def main():
    data = {'schema': 1, 'scope': 'aggregate derived evidence; held-out-selected, Sleep-EDF subject-overlapping',
            'sources': {k: {'path': v, 'sha256': sha(ROOT / v)} for k, v in sorted(SOURCES.items())}}
    datasets = {}
    for name, parent, bim in [('s20', 'ULW_SELECTED', 'BIM_SELECTED'), ('s78', 'U0', 'B0'), ('s3', 'U0', 'B0')]:
        report = read(name)
        folds = []
        for fold in report['folds']:
            key = 'heldout_metrics' if name == 's3' else 'metrics'
            folds.append({'fold': fold['fold'], 'n_epochs': fold['n_heldout_epochs'],
                          'parent_correct': sum(fold[parent][key]['confusion_matrix_true_rows'][i][i] for i in range(5)),
                          'bim_correct': sum(fold[bim][key]['confusion_matrix_true_rows'][i][i] for i in range(5))})
        datasets[name] = {'parent_cm': metrics(report['pooled'][parent]), 'bim_cm': metrics(report['pooled'][bim]),
                          'folds': sorted(folds, key=lambda f: f['fold']),
                          'parameter_ledger': report['parameter_ledger'], 'operation_ledger': report['operation_ledger']}
    data['datasets'] = datasets
    analyses = {}
    for name in ('s20', 's78'):
        report = read('explain_' + name)
        analyses[name] = {
            'ablation_cm': {**{'remove_' + k: metrics(v['metrics']) for k, v in report['pure_counterfactuals'].items() if k in ('all_true_history', 'past_true_history', 'future_true_history')},
                           **{'mask_' + k: metrics(v['metrics']) for k, v in report['coordinate_counterfactuals'].items() if k in ('memory_all', 'innovation_all', 'scale_0', 'scale_1', 'scale_2')}},
            'subject_weighted_transition_differences': subject_diffs(SOURCES['attrib_' + name]),
            'bootstrap_claim': report['bootstrap']['transition_weighted_innovation_difference'],
            'share_claims': {k: report['mechanism'][k] for k in ('correction_true_history_share_median', 'harmful_true_history_share_median')},
            'correction_share_bootstrap_claim': report['bootstrap']['correction_true_history_share'],
            'subject_correction_share_medians': [v['correction_history_share_median'] for _, v in sorted(report['strata_attribution']['subject'].items(), key=lambda kv: int(kv[0])) if v['correction_history_share_median'] is not None],
            'fidelity_claim': report['fidelity']['residual_completeness_max_abs_error'],
            'tau_minutes_by_fold': [row['tau_minutes'] for row in report['time_constants']],
            'counts': report['counts'],
        }
    data['analyses'] = analyses
    fig = read('figure')
    data['figure2'] = {'baseline_margin': fig['baseline_margin'], 'full_margin': fig['full_margin'],
                       'removed_margin': fig['removed_margin'], 'contributions': fig['example']['pure_margin_contributions']}
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'{OUT.relative_to(ROOT)}: {OUT.stat().st_size} bytes, sha256 {sha(OUT)}')


if __name__ == '__main__':
    main()
