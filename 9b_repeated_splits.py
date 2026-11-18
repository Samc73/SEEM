"""Step 9b: held-out comparisons with REFITTING noise included, two schemes.

Input : out/joint_events.npz, out/marginals.json (the same main set as step 9);
        for the bootstrap also out/dependence_law_scoreboard<suffix>.json (warm starts)
Output: out/replicates/<scheme>_<ranks>_<start>_<stop>.json   (one per chunk)
        out/dependence_law_replicates.json                     (merge)

    python 9b_repeated_splits.py halving   ranks=voxel_rate start=0 stop=50
    python 9b_repeated_splits.py split80   ranks=voxel_rate start=0 stop=50
    python 9b_repeated_splits.py bootstrap ranks=voxel_rate start=0 stop=100
    python 9b_repeated_splits.py merge

halving    (the DECIDING evidence) each replicate takes half of the trajectories
           of EACH cooling rate (stratified) at random, refits everything on
           them -- ranks, every form with full starting points, both voxel
           tables -- and scores every model on the other half.  No trajectory
           appears twice, so the many-parameter tables are not handicapped.
           Halves overlap between replicates: the spread is a stability guide
           (mean, sd, fraction won), not a confidence interval.
split80    as halving, with 80% of each rate's trajectories for training and
           20% held out (the training share of 5 folds)
bootstrap  each replicate resamples trajectories with replacement within each
           rate, then redoes step 9's 2-fold fit (warm starts from step 9's
           in-sample fit) and an in-sample refit for coefficient intervals.
           Duplicated trajectories handicap the voxel tables (a resample keeps
           ~63% of distinct trajectories), so model comparisons from it lean
           toward the small models; reported, not used to decide.
Every replicate's random draws depend only on (scheme, ranks, replicate), so
chunks can run in parallel and give the same result as one run.
Ranks: as step 9 (ranks=voxel_rate | voxel_rate3 | voxel).
"""
import glob
import json
import os
import re
import sys
import time
import warnings
import numpy as np
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
MIN_PAIRED = 100          # as step 9
SEED = 2026
SCHEMES = ('halving', 'bootstrap', 'split80')       # order fixes the seeds; append only
TRAIN_FRACTION = dict(halving=0.5, split80=0.8)
RANK_MODES = ('voxel_rate', 'voxel_rate3', 'voxel')
SUFFIX = dict(voxel_rate='', voxel_rate3='_voxel_rate3', voxel='_pooled')
args = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
SCHEME = sys.argv[1] if len(sys.argv) > 1 else 'halving'
RANKS = args.get('ranks', 'voxel_rate')


def load():
    events = np.load('out/joint_events.npz')
    with open('out/marginals.json') as handle:
        xmin = json.load(handle)['xmin']
    n_bins = int(events['n_bins'])
    du, dtau, cell_all = events['du'], events['dtau'], events['cell']
    paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
    counts = np.bincount(cell_all[paired], minlength=n_bins * n_bins)
    keep = paired & (counts[cell_all] >= MIN_PAIRED)
    rates = np.unique(events['rate'])
    data = dict(x=-du[keep], y=-dtau[keep], u=events['u'][keep], tau=events['tau'][keep],
                cell=cell_all[keep], traj=events['traj'][keep], rate=events['rate'][keep])
    data['rate_index'] = np.searchsorted(rates, data['rate'])
    data['ln_rate'] = np.log(data['rate'] / rates[len(rates) // 2])        # as step 3 and step 9
    return data


def rank_levels(data, ranks):
    c, r = data['cell'], data['rate_index']
    return dict(voxel_rate=[c * 9 + r, c * 3 + r // 3, c], voxel_rate3=[c * 3 + r // 3, c], voxel=[c])[ranks]


def stratified_trajectories(data):
    ids, first = np.unique(data['traj'], return_index=True)
    return {r: ids[data['rate'][first] == r] for r in np.unique(data['rate'][first])}


def replicate(data, scheme, ranks, s, warm=None):
    rng = np.random.default_rng([SEED, SCHEMES.index(scheme), RANK_MODES.index(ranks), s])
    strata = stratified_trajectories(data)
    if scheme in TRAIN_FRACTION:
        train_ids = np.concatenate([rng.choice(ids, int(round(TRAIN_FRACTION[scheme] * len(ids))), replace=False)
                                    for ids in strata.values()])
        fold_id = np.where(np.isin(data['traj'], train_ids), 0, 1)       # fold 1 is held out
        held, folds = dl.cross_validate(data['x'], data['y'], data['u'], data['tau'], data['cell'], fold_id, rng,
                                        held_out_folds=[1], ln_rate=data['ln_rate'],
                                        rank_levels=rank_levels(data, ranks))
        test = fold_id == 1
        fd = folds[0]
        return dict(replicate=s, n_test=int(test.sum()),
                    totals={m: float(held[m][test].sum()) for m in held},
                    betas={f: r['beta'] for f, r in fd['forms'].items() if 'beta' in r},
                    g_table_rate=fd['forms'][dl.TABLE_RATE]['g'],
                    flags=['%s: %s' % (f, x) for f, r in fd['forms'].items() for x in r.get('flags', [])],
                    nesting_violations=fd['nesting_violations'], rank_level_events=fd['rank_level_events'],
                    rank_fallback=fd['rank_fallback_events'])
    # bootstrap: resample trajectories within each rate, refit the 2-fold split and in-sample
    members = {t: np.flatnonzero(data['traj'] == t) for t in np.unique(data['traj'])}
    pick = np.concatenate([rng.choice(ids, len(ids)) for ids in strata.values()])
    index = np.concatenate([members[t] for t in pick if t in members])
    d = {k: v[index] for k, v in data.items()}
    held, folds = dl.cross_validate(d['x'], d['y'], d['u'], d['tau'], d['cell'], d['traj'] % 2, rng, warm=warm,
                                    ln_rate=d['ln_rate'], rank_levels=rank_levels(d, ranks))
    fit = dl.fit_in_sample(d['x'], d['y'], d['u'], d['tau'], d['cell'], rng, ln_rate=d['ln_rate'],
                           rank_levels=rank_levels(d, ranks), warm=warm)
    return dict(replicate=s, n_events=int(len(index)), totals={m: float(held[m].sum()) for m in held},
                betas={f: r['beta'].tolist() for f, r in fit['fits'].items()},
                g_table_rate=fit['table_rate']['g'],
                flags=['fold %d %s: %s' % (fd['fold'], f, x) for fd in folds for f, r in fd['forms'].items()
                       for x in r.get('flags', [])],
                nesting_violations=[v for fd in folds for v in fd['nesting_violations']] + fit['nesting_violations'])


def warm_starts(ranks):
    with open('out/dependence_law_scoreboard%s.json' % SUFFIX[ranks]) as handle:
        ins = json.load(handle)['in_sample']
    warm = {f: np.array(ins[f]['beta']) for f in dl.FORMS + dl.RATE_FORMS}
    warm[dl.TABLE_RATE] = ins[dl.TABLE_RATE]['g']
    return warm


# --- the comparisons summarised for every scheme and rank mode ------------------------------------
def comparisons():
    pairs = [(m, 'constant') for m in dl.MODELS_WITH_RATE if m != 'constant']
    pairs += [(f, dl.TABLE) for f in dl.FORMS] + [(f, dl.TABLE_RATE) for f in dl.RATE_FORMS]
    pairs += [(f + dl.RATE, f) for f in dl.FORMS] + [(dl.TABLE_RATE, dl.TABLE)]
    for suffix in ('', dl.RATE):
        pairs += [('quadratic' + suffix, 'interaction' + suffix), ('quadratic' + suffix, 'product' + suffix),
                  ('interaction' + suffix, 'product' + suffix), ('interaction' + suffix, 'linear' + suffix)]
    return pairs


def summarise(rows, scheme):
    out = {}
    for a, b in comparisons():
        d = np.array([r['totals'][a] - r['totals'][b] for r in rows])
        entry = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), median=float(np.median(d)),
                     percentiles=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     wins=int((d > 0).sum()), fraction_won=float(np.mean(d > 0)), n=len(rows),
                     differences=d.tolist())
        if scheme in TRAIN_FRACTION:
            per_k = 1000 * d / np.array([r['n_test'] for r in rows])
            entry.update(mean_per_1000=float(per_k.mean()), sd_per_1000=float(per_k.std(ddof=1)),
                         percentiles_per_1000=[float(np.percentile(per_k, 2.5)), float(np.percentile(per_k, 97.5))],
                         per_1000=per_k.tolist())
        out['%s - %s' % (a, b)] = entry
    coef = {}
    for f in dl.FORMS + dl.RATE_FORMS:
        betas = np.array([r['betas'][f] for r in rows])
        coef[f] = [dict(name=nm, median=float(np.median(betas[:, i])), sd=float(betas[:, i].std(ddof=1)),
                        percentiles=[float(np.percentile(betas[:, i], 2.5)), float(np.percentile(betas[:, i], 97.5))])
                   for i, nm in enumerate(dl.TERMS[f])]
    g = np.array([r['g_table_rate'] for r in rows])
    coef[dl.TABLE_RATE] = [dict(name='g', median=float(np.median(g)), sd=float(g.std(ddof=1)),
                                percentiles=[float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))])]
    return out, coef


if SCHEME == 'merge':
    merged = dict(seed=SEED, note_halving='halves overlap between replicates; the spread is a stability '
                                          'guide, not a confidence interval',
                  note_bootstrap='duplicated trajectories handicap the voxel tables: comparisons lean '
                                 'toward small models', results={})
    for scheme in SCHEMES:
        for ranks in RANK_MODES:
            pattern = re.compile(r'%s_%s_\d{3}_\d{3}\.json$' % (scheme, ranks))     # exact: voxel != voxel_rate
            files = sorted(f for f in glob.glob('out/replicates/*.json') if pattern.match(os.path.basename(f)))
            if not files:
                continue
            rows = sorted([r for fn in files for r in json.load(open(fn))['rows']], key=lambda r: r['replicate'])
            assert len({r['replicate'] for r in rows}) == len(rows), 'duplicate replicates in %s' % files
            summary, coef = summarise(rows, scheme)
            flags = [(r['replicate'], f) for r in rows for f in r['flags']]
            nesting = [(r['replicate'], v) for r in rows for v in r['nesting_violations']]
            merged['results']['%s/%s' % (scheme, ranks)] = dict(
                n=len(rows), comparisons=summary, coefficients=coef, flags=flags, nesting_violations=nesting,
                rank_level_events=[r.get('rank_level_events') for r in rows], replicates=rows)
            print('\n%s, ranks %s: %d replicates; flags %d; nesting violations %d'
                  % (scheme, ranks, len(rows), len(flags), len(nesting)))
            if scheme in TRAIN_FRACTION:
                lev = [r['rank_level_events'] for r in rows]
                tr = np.mean([l['train'] for l in lev], axis=0)
                te = np.mean([l['test'] for l in lev], axis=0)
                merged['results']['%s/%s' % (scheme, ranks)]['mean_rank_level_events'] = dict(
                    train=tr.tolist(), test=te.tolist(),
                    train_fraction_fallback=float(1 - tr[0] / tr.sum()), test_fraction_fallback=float(1 - te[0] / te.sum()))
                print('  mean events per rank level (finest first): training %s (%.1f%% fall back), held out %s '
                      '(%.1f%% fall back); mean held-out events %d'
                      % (np.round(tr).astype(int).tolist(), 100 * (1 - tr[0] / tr.sum()),
                         np.round(te).astype(int).tolist(), 100 * (1 - te[0] / te.sum()),
                         np.mean([r['n_test'] for r in rows])))
            for key, e in summary.items():
                if scheme in TRAIN_FRACTION:
                    print('  %-36s mean %+8.1f  sd %6.1f  [%+8.1f, %+8.1f]  won %2d / %d   per 1,000: %+6.2f +- %.2f'
                          % (key, e['mean'], e['sd'], *e['percentiles'], e['wins'], e['n'],
                             e['mean_per_1000'], e['sd_per_1000']))
                else:
                    print('  %-36s median %+8.1f  [%+8.1f, %+8.1f]  P(> 0) %.2f'
                          % (key, e['median'], *e['percentiles'], e['fraction_won']))
    with open('out/dependence_law_replicates.json', 'w') as handle:
        json.dump(merged, handle, indent=1)
    print('\nwrote out/dependence_law_replicates.json')
    sys.exit(0)

start, stop = int(args.get('start', 0)), int(args.get('stop', 50))
assert SCHEME in SCHEMES and RANKS in RANK_MODES
t0 = time.time()
data = load()
warm = warm_starts(RANKS) if SCHEME == 'bootstrap' else None
rows = []
for s in range(start, stop):
    rows.append(replicate(data, SCHEME, RANKS, s, warm))
    print('%s %s replicate %d done (%.0f s)' % (SCHEME, RANKS, s, time.time() - t0), flush=True)
os.makedirs('out/replicates', exist_ok=True)
with open('out/replicates/%s_%s_%03d_%03d.json' % (SCHEME, RANKS, start, stop), 'w') as handle:
    json.dump(dict(scheme=SCHEME, ranks=RANKS, start=start, stop=stop, min_rank_train=dl.MIN_RANK_TRAIN,
                   rows=rows), handle)
print('wrote chunk %d-%d (%.0f s)' % (start, stop, time.time() - t0))
