"""Check compact paper evidence without raw data, checkpoints or frozen results.

Requires numpy only for matching the original PCG64 subject bootstrap exactly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'review_evidence' / 'paper_v1.json'
CLAIMS = {
    's20': {'parent': (87.07, 80.91, .823), 'bim': (88.00, 82.59, .836), 'gain': (.93, 1.68, 5.08), 'folds': 9},
    's78': {'parent': (82.18, 75.22, .751), 'bim': (84.04, 78.31, .777), 'gain': (1.86, 3.09, 7.66), 'folds': 10},
    's3': {'parent': (82.08, 80.80, .770), 'bim': (83.12, 81.77, .783), 'gain': (1.04, .97, .92), 'folds': 6},
}
ABLATION = {
    'remove_all_true_history': (85.91, 80.36, 47.86),
    'remove_past_true_history': (86.19, 80.59, 47.50),
    'remove_future_true_history': (87.35, 81.86, 51.64),
    'mask_memory_all': (87.78, 82.15, 51.33),
    'mask_innovation_all': (87.37, 81.56, 49.97),
    'mask_scale_0': (87.73, 82.33, 52.19),
    'mask_scale_1': (87.79, 82.24, 51.68),
    'mask_scale_2': (87.76, 81.96, 50.92),
}


def check(cond, message):
    if not cond:
        raise AssertionError(message)


def close(a, b, tol=.00501):
    check(abs(a-b) <= tol, f'{a} != {b} (tolerance {tol})')


def measures(cm):
    check(len(cm) == 5 and all(len(row) == 5 for row in cm), 'bad confusion matrix')
    check(all(isinstance(v, int) and v >= 0 for row in cm for v in row), 'bad count')
    n = sum(map(sum, cm))
    rows = list(map(sum, cm))
    cols = [sum(cm[i][j] for i in range(5)) for j in range(5)]
    correct = sum(cm[i][i] for i in range(5))
    f1 = [2*cm[i][i]/(rows[i]+cols[i]) for i in range(5)]
    pe = sum(rows[i]*cols[i] for i in range(5))/n**2
    kappa = (correct/n-pe)/(1-pe)
    return {'n': n, 'correct': correct, 'acc': 100*correct/n, 'mf1': 20*sum(f1), 'n1': 100*f1[1], 'kappa': kappa}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sources', action='store_true', help='verify hashes for source files present locally')
    args = parser.parse_args()
    data = json.loads(BUNDLE.read_text())
    check(data['schema'] == 1, 'schema mismatch')
    for name, claim in CLAIMS.items():
        row = data['datasets'][name]
        parent, bim = measures(row['parent_cm']), measures(row['bim_cm'])
        for key, expected in [('parent', parent), ('bim', bim)]:
            for actual, paper in zip((expected['acc'], expected['mf1'], expected['kappa']), claim[key]):
                close(actual, paper, .00501 if paper > 1 else .00051)
        for actual, paper in zip((bim['acc']-parent['acc'], bim['mf1']-parent['mf1'], bim['n1']-parent['n1']), claim['gain']):
            close(actual, paper)
        folds = row['folds']
        check(len(folds) == 10 and [f['fold'] for f in folds] == list(range(10)), name+' fold ids')
        check(sum(f['n_epochs'] for f in folds) == bim['n'], name+' fold total')
        check(sum(f['parent_correct'] for f in folds) == parent['correct'], name+' parent fold reconciliation')
        check(sum(f['bim_correct'] for f in folds) == bim['correct'], name+' bim fold reconciliation')
        positive = sum(f['bim_correct'] > f['parent_correct'] for f in folds)
        check(positive == claim['folds'], name+' positive folds')
        ledger = row['parameter_ledger']
        parent_params = ledger.get('ULW', ledger.get('U0_complete'))
        bim_params = ledger.get('BIM_complete', ledger.get('B0_complete'))
        check(parent_params + ledger['BIM_head'] == bim_params, name+' parameter ledger')
        print(f'{name}: n={bim["n"]}, ACC {parent["acc"]:.2f}->{bim["acc"]:.2f}, MF1 {parent["mf1"]:.2f}->{bim["mf1"]:.2f}, positive folds {positive}/10')
    s20 = data['analyses']['s20']
    full = measures(data['datasets']['s20']['bim_cm'])
    for key, expected in ABLATION.items():
        m = measures(s20['ablation_cm'][key])
        check(m['n'] == full['n'], key+' count')
        for actual, paper in zip((m['acc'], m['mf1'], m['n1']), expected):
            close(actual, paper)
    close(full['acc']-measures(s20['ablation_cm']['remove_all_true_history'])['acc'], 2.09)
    for name, point, interval, share, harm, err in [
        ('s20', .1001, (.0751, .1244), .7581, .6542, 1.3322676295501878e-15),
        ('s78', .1180, (.1059, .1303), .8192, .7800, 8.881784197001252e-16),
    ]:
        a = data['analyses'][name]
        values = np.asarray(a['subject_weighted_transition_differences'])
        spec = a['bootstrap_claim']
        check(len(values) == spec['n_subjects'], name+' subject count')
        close(float(values.mean()), point, .000051)
        rng = np.random.default_rng(spec['seed'])
        draws = values[rng.integers(0, len(values), size=(spec['samples'], len(values)))].mean(axis=1)
        ci = np.quantile(draws, [.025, .975])
        for actual, reported, paper in zip(ci, spec['ci95'], interval):
            close(float(actual), reported, 1e-12)
            close(float(actual), paper, .000051)
        shares = np.asarray(a['subject_correction_share_medians'])
        share_spec = a['correction_share_bootstrap_claim']
        check(len(shares) == share_spec['n_subjects'], name+' correction share subjects')
        share_rng = np.random.default_rng(share_spec['seed'])
        share_draws = shares[share_rng.integers(0, len(shares), size=(share_spec['samples'], len(shares)))].mean(axis=1)
        close(float(shares.mean()), share_spec['estimate'], 1e-12)
        for actual, reported in zip(np.quantile(share_draws, [.025, .975]), share_spec['ci95']):
            close(float(actual), reported, 1e-12)
        # The paper's pooled event median cannot be rebuilt from subject medians.
        close(a['share_claims']['correction_true_history_share_median'], share, .000051)
        close(a['share_claims']['harmful_true_history_share_median'], harm, .000051)
        check(a['fidelity_claim'] <= err + 1e-18, name+' fidelity limit')
        check(len(a['tau_minutes_by_fold']) == 10, name+' time constants')
        print(f'{name}: transition difference {values.mean():.4f}, 95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}]')
    fig = data['figure2']
    c = fig['contributions']
    past = sum(c['past_history_'+str(i)] for i in range(3))
    future = sum(c['future_history_'+str(i)] for i in range(3))
    close(fig['baseline_margin'] + sum(c.values()), fig['full_margin'], 2e-7)
    close(fig['full_margin']-past-future, fig['removed_margin'], 2e-7)
    for actual, paper in [(fig['baseline_margin'], -.243), (fig['full_margin'], 1.422),
                          (fig['removed_margin'], -.627), (past, .810), (future, 1.239),
                          (c['local_total'], -.457), (c['bias'], .072)]:
        close(actual, paper, .00051)
    print('Table II, Fig. 2, parameter totals and attribution precision: PASS')
    if args.check_sources:
        found = 0
        for source in data['sources'].values():
            path = ROOT / source['path']
            if path.is_file():
                found += 1
                check(hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], str(path)+' SHA-256')
        print(f'Source SHA-256: {found}/{len(data["sources"])} files available and matched')
    print('PASS — numerical review only; this does not rerun training or evaluation.')


if __name__ == '__main__':
    main()
