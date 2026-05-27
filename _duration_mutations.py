"""Notebook patches: per-algo wall-clock timing + BCO mutation-counts viz."""
import json
import ast

PATH = 'examples/route_generator/final_experiments.ipynb'
nb = json.load(open(PATH, 'r', encoding='utf-8'))


# ---- 0. imports: add `import time` for perf_counter timing ----
for c in nb['cells']:
    if c.get('id') != 'imports':
        continue
    src = ''.join(c['source'])
    old = 'import math\nimport gc'
    new = 'import math\nimport gc\nimport time'
    assert old in src and 'import time' not in src.splitlines(), \
        'imports anchor missing OR `import time` already present'
    src = src.replace(old, new, 1)
    ast.parse(src); compile(src, '<imports>', 'exec')
    c['source'] = src.splitlines(keepends=True)
    c['outputs'] = []; c['execution_count'] = None
    print('imports: added `import time`')
    break


# ---- 1. a-helpers: time each branch + capture mutation_counts ----
for c in nb['cells']:
    if c.get('id') != 'a-helpers':
        continue
    src = ''.join(c['source'])

    # Insert `mutation_counts = None` default and start wall-clock timer right
    # after the `cost_history = None` line.
    old = 'cost_history = None   # overridden in sa/ga/hh/bco branches'
    new = ('cost_history = None   # overridden in sa/ga/hh/bco branches\n'
           '    mutation_counts = None  # set in bco branch; others stay None\n'
           '    _t_start = time.perf_counter()')
    assert old in src, 'a-helpers default-init anchor missing'
    src = src.replace(old, new, 1)

    # BCO branch already binds `_mc`; pipe it into `mutation_counts`.
    old_bco = ('        _, metrics, _, routes, _mc = run_bco(\n'
               '            cfg, init_routes, tensors=tensors, run_name_scope=run_scope,\n'
               '            cost_history_out=_ch_out)\n'
               '        _h = _ch_out.get("history")')
    new_bco = ('        _, metrics, _, routes, _mc = run_bco(\n'
               '            cfg, init_routes, tensors=tensors, run_name_scope=run_scope,\n'
               '            cost_history_out=_ch_out)\n'
               '        mutation_counts = _mc\n'
               '        _h = _ch_out.get("history")')
    assert old_bco in src, 'a-helpers bco _mc anchor missing'
    src = src.replace(old_bco, new_bco, 1)

    # Stop timer and add duration_s to the row; extend return tuple.
    old_row = ('    row = {"city": city, "alpha": alpha, "method": method_label,\n'
               '           "ATT": float(metric_value(metrics, "ATT")),\n'
               '           "RTT": float(metric_value(metrics, "RTT")),\n'
               '           "d_0": float(metric_value(metrics, "$d_0$")),\n'
               '           "d_1": float(metric_value(metrics, "$d_1$")),\n'
               '           "d_2": float(metric_value(metrics, "$d_2$")),\n'
               '           "d_un": float(metric_value(metrics, "$d_{un}$")),\n'
               '           "cost": float(metric_value(metrics, "cost"))}\n'
               '    return row, routes, cost_history')
    new_row = ('    _duration_s = time.perf_counter() - _t_start\n'
               '    row = {"city": city, "alpha": alpha, "method": method_label,\n'
               '           "ATT": float(metric_value(metrics, "ATT")),\n'
               '           "RTT": float(metric_value(metrics, "RTT")),\n'
               '           "d_0": float(metric_value(metrics, "$d_0$")),\n'
               '           "d_1": float(metric_value(metrics, "$d_1$")),\n'
               '           "d_2": float(metric_value(metrics, "$d_2$")),\n'
               '           "d_un": float(metric_value(metrics, "$d_{un}$")),\n'
               '           "cost": float(metric_value(metrics, "cost")),\n'
               '           "duration_s": float(_duration_s)}\n'
               '    return row, routes, cost_history, mutation_counts')
    assert old_row in src, 'a-helpers row-return anchor missing'
    src = src.replace(old_row, new_row, 1)

    ast.parse(src); compile(src, '<a-helpers>', 'exec')
    c['source'] = src.splitlines(keepends=True)
    c['outputs'] = []; c['execution_count'] = None
    print('a-helpers: timing + mutation_counts capture + 4-tuple return')
    break


# ---- 2. a-sweep: capture mutation counts + propagate duration in fail row ----
for c in nb['cells']:
    if c.get('id') != 'a-sweep':
        continue
    src = ''.join(c['source'])

    old_init = ('final_a_histories = {}   # (city, alpha, method) -> '
                'cost-per-iter or None\n'
                '    city_to_spec_a = {s["city"]: s for s in BENCHMARK_SPECS}')
    new_init = ('final_a_histories = {}   # (city, alpha, method) -> '
                'cost-per-iter or None\n'
                '    final_a_mutation_counts = {}  # '
                '(city, alpha, method) -> BCO mutation_counts dict or None\n'
                '    city_to_spec_a = {s["city"]: s for s in BENCHMARK_SPECS}')
    assert old_init in src, 'a-sweep init anchor missing'
    src = src.replace(old_init, new_init, 1)

    old_unpack = ('row, _routes, _history = _run_one(\n'
                  '                        city, alpha, label, kind, spec, tensors, init_routes)\n'
                  '                    final_a_rows.append(row)\n'
                  '                    final_a_histories[(city, alpha, label)] = _history')
    new_unpack = ('row, _routes, _history, _mut = _run_one(\n'
                  '                        city, alpha, label, kind, spec, tensors, init_routes)\n'
                  '                    final_a_rows.append(row)\n'
                  '                    final_a_histories[(city, alpha, label)] = _history\n'
                  '                    final_a_mutation_counts[(city, alpha, label)] = _mut')
    assert old_unpack in src, 'a-sweep unpack anchor missing'
    src = src.replace(old_unpack, new_unpack, 1)

    # Failure row: add duration_s=nan so the column stays uniform.
    old_fail = ('"cost": float("nan")})\n'
                '                finally:')
    new_fail = ('"cost": float("nan"),\n'
                '                                         "duration_s": float("nan")})\n'
                '                finally:')
    assert old_fail in src, 'a-sweep fail-row anchor missing'
    src = src.replace(old_fail, new_fail, 1)

    ast.parse(src); compile(src, '<a-sweep>', 'exec')
    c['source'] = src.splitlines(keepends=True)
    c['outputs'] = []; c['execution_count'] = None
    print('a-sweep: mutation-counts capture + duration_s in fail row')
    break


# ---- 3. a-tables: include duration_s column ----
for c in nb['cells']:
    if c.get('id') != 'a-tables':
        continue
    src = ''.join(c['source'])
    old = 'METRIC_COLS_ALPHA = ["ATT", "RTT", "d_0", "d_1", "d_2", "d_un", "cost"]'
    new = ('METRIC_COLS_ALPHA = ["ATT", "RTT", "d_0", "d_1", "d_2", "d_un", '
           '"cost", "duration_s"]')
    assert old in src, 'a-tables METRIC_COLS anchor missing'
    src = src.replace(old, new, 1)
    ast.parse(src); compile(src, '<a-tables>', 'exec')
    c['source'] = src.splitlines(keepends=True)
    c['outputs'] = []; c['execution_count'] = None
    print('a-tables: duration_s column added')
    break


# ---- 4. Insert a-mutations-md + a-mutations-fig cells after a-history-fig ----
MUTATIONS_MD = '''### Section A: accepted vs rejected mutations (BCO only)

Each BCO iteration proposes one mutation per bee from a heterogeneous mix of
mutator types (`type1` = SP rebuild, `type2` = local extend/shorten,
`type3` = path-combiner rebuild, `type4-7` = neural variants). The
`mutation_counts` dict captured in the sweep tracks per-type:

* **attempted** -- how many times a bee of that type proposed a candidate.
* **accepted** -- proposals that survived selection (strictly better cost
  OR worse-accepted under the temperature OR force-accepted by trim-grace).
* **worse_accepted** -- subset of `accepted` where the proposal was worse
  than the current network but accepted under the worse-accept softmax.

Two views below:

* **Top panel**: per-method grouped bars showing the totals (sum across all
  `(city, alpha)` runs) for the three non-empty types.
* **Bottom**: per-method **accept rate** = `accepted / attempted` per type.
  This is the more interpretable number -- higher rate = the type is
  *productive*, lower rate = the proposals are wasted compute.

Non-BCO algorithms (SA / GA / HH / NSGA-II / RL) do not currently surface
comparable per-mutation-type counts and are omitted. Wall-clock duration
per run is captured separately in `duration_s` (see the §A tables above).'''


MUTATIONS_CODE = '''import numpy as _np_mut

# Collect BCO methods that produced a non-None mutation_counts dict.
_bco_methods = sorted({m for (_, _, m), mc in final_a_mutation_counts.items()
                         if isinstance(mc, dict)})
print(f"BCO methods with mutation_counts: {_bco_methods}")
if not _bco_methods:
    print("[note] No BCO runs captured -- skipping the mutation comparison.")

if _bco_methods:
    # Sum attempted / accepted / worse_accepted per (method, type) across runs.
    _type_keys = ["type1", "type2", "type3", "type4", "type5", "type6", "type7"]
    _aggregate = {m: {bucket: {t: 0 for t in _type_keys}
                       for bucket in ("attempted", "accepted", "worse_accepted")}
                   for m in _bco_methods}
    for (_, _, m), mc in final_a_mutation_counts.items():
        if not isinstance(mc, dict):
            continue
        for bucket in ("attempted", "accepted", "worse_accepted"):
            sub = mc.get(bucket, {})
            if isinstance(sub, dict):
                for t in _type_keys:
                    _aggregate[m][bucket][t] += int(sub.get(t, 0))

    # Drop type columns that are zero everywhere (keeps the plot uncluttered).
    _types_with_data = [t for t in _type_keys
                        if any(_aggregate[m]["attempted"][t] > 0
                               for m in _bco_methods)]
    print(f"Mutation types with non-zero attempts: {_types_with_data}")

    if _types_with_data:
        n_types = len(_types_with_data)
        n_methods = len(_bco_methods)
        bar_w = 0.8 / 3                           # 3 bars per type group
        x = _np_mut.arange(n_types)

        fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.5 * n_methods * n_types), 10))

        # ---- Top panel: raw totals ----
        ax = axes[0]
        cmap = plt.get_cmap("tab10")
        method_colors = {m: cmap(i % 10) for i, m in enumerate(_bco_methods)}
        group_w = 0.8
        sub_w = group_w / n_methods
        for mi, method in enumerate(_bco_methods):
            offsets = -group_w / 2 + sub_w * (mi + 0.5)
            attempted = [_aggregate[method]["attempted"][t] for t in _types_with_data]
            accepted = [_aggregate[method]["accepted"][t] for t in _types_with_data]
            worse = [_aggregate[method]["worse_accepted"][t] for t in _types_with_data]
            # Stack: rejected at bottom, accepted-(worse_accept) middle, worse_accept top.
            rejected = [a - acc for a, acc in zip(attempted, accepted)]
            accepted_strict = [acc - w for acc, w in zip(accepted, worse)]
            ax.bar(x + offsets, rejected, sub_w,
                    color=method_colors[method], alpha=0.25,
                    label=f"{method} (rejected)" if mi == 0 else None)
            ax.bar(x + offsets, accepted_strict, sub_w, bottom=rejected,
                    color=method_colors[method], alpha=0.75,
                    label=f"{method}")
            ax.bar(x + offsets, worse, sub_w,
                    bottom=[r + a for r, a in zip(rejected, accepted_strict)],
                    color=method_colors[method], alpha=1.0, hatch="//",
                    edgecolor="black", linewidth=0.3,
                    label=f"{method} (worse-accept)" if mi == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(_types_with_data)
        ax.set_ylabel("count (sum across runs)")
        ax.set_title("Attempted mutations per type per method\\n"
                      "(faded = rejected, solid = accepted-strict, hatched = worse-accepted)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="best")

        # ---- Bottom panel: accept rate ----
        ax = axes[1]
        for mi, method in enumerate(_bco_methods):
            offsets = -group_w / 2 + sub_w * (mi + 0.5)
            rates = []
            for t in _types_with_data:
                a = _aggregate[method]["attempted"][t]
                acc = _aggregate[method]["accepted"][t]
                rates.append(acc / a if a > 0 else 0.0)
            ax.bar(x + offsets, rates, sub_w, color=method_colors[method],
                    label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(_types_with_data)
        ax.set_ylabel("accept rate (accepted / attempted)")
        ax.set_title("Per-type accept rate per method")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="black", alpha=0.2, linewidth=0.6)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="best")

        fig.tight_layout()
        plt.show()

    # ---- Summary table: totals + accept-rate per method ----
    _summary_rows = []
    for method in _bco_methods:
        att = sum(_aggregate[method]["attempted"].values())
        acc = sum(_aggregate[method]["accepted"].values())
        wor = sum(_aggregate[method]["worse_accepted"].values())
        _summary_rows.append({
            "method": method,
            "attempted": att,
            "accepted": acc,
            "rejected": att - acc,
            "worse_accepted": wor,
            "accept_rate": acc / att if att > 0 else 0.0,
            "worse_share_of_accept": wor / acc if acc > 0 else 0.0,
        })
    _mut_summary_df = pd.DataFrame(_summary_rows).sort_values(
        "accept_rate", ascending=False).reset_index(drop=True)
    save_table(_mut_summary_df, "a_mutation_summary",
                subdir=FINAL_RESULTS_SUBDIR)
    display(_mut_summary_df.round(3))

    # ---- Wall-clock comparison: total seconds per method (summed across runs) ----
    if "duration_s" in final_a_df.columns:
        _dur = (final_a_df.dropna(subset=["duration_s"])
                .groupby("method", as_index=False)["duration_s"].sum()
                .sort_values("duration_s", ascending=True))
        fig2, ax = plt.subplots(figsize=(10, 0.4 * max(4, len(_dur)) + 2))
        ax.barh(_dur["method"], _dur["duration_s"], color="steelblue")
        ax.set_xlabel("total wall-clock seconds across all (city, alpha) runs")
        ax.set_title("Section A wall-clock cost per method")
        ax.grid(axis="x", alpha=0.25)
        fig2.tight_layout()
        plt.show()'''


def md(cid, text):
    lines = text.split('\n')
    return {'cell_type': 'markdown', 'id': cid, 'metadata': {},
            'source': [ln + '\n' for ln in lines[:-1]] + [lines[-1]]}


def code(cid, text):
    lines = text.split('\n')
    return {'cell_type': 'code', 'id': cid, 'execution_count': None,
            'metadata': {}, 'outputs': [],
            'source': [ln + '\n' for ln in lines[:-1]] + [lines[-1]]}


ast.parse(MUTATIONS_CODE); compile(MUTATIONS_CODE, '<a-mutations-fig>', 'exec')

hist_idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'a-history-fig')
nb['cells'] = (nb['cells'][:hist_idx + 1]
               + [md('a-mutations-md', MUTATIONS_MD),
                  code('a-mutations-fig', MUTATIONS_CODE)]
               + nb['cells'][hist_idx + 1:])
print(f'inserted a-mutations-md + a-mutations-fig after a-history-fig (idx {hist_idx})')


with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False); f.write('\n')
json.load(open(PATH, 'r', encoding='utf-8'))
print('\nJSON OK')
