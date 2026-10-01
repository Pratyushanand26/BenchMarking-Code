"""Turn raw per-clip rows + per-frame traces into the comparison tables.

    python -m fallbench.summarize [results_dir]

Outputs (in results_dir):
    summary.md          human-readable table (median [IQR] over repeats, % vs full30)
    summary.csv         the same numbers, machine-readable
    per_repeat.csv      one row per (strategy, repeat) before the median is taken
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from .config import STRATEGIES


def _pooled_latency(results_dir, strategy, rep):
    p = os.path.join(results_dir, "traces", f"{strategy}_rep{rep}.npz")
    if not os.path.exists(p):
        return {}
    t = np.load(p)
    steady = t["steady"]
    sel = steady & t["arrival"]            # frames the camera actually delivers
    out = {}
    fr = t["frame_ns"][sel] / 1e6
    if fr.size:
        out.update(lat_median_ms=np.median(fr), lat_p95_ms=np.percentile(fr, 95),
                   lat_p99_ms=np.percentile(fr, 99), lat_max_ms=fr.max())
    lag = t["lag_ns"][sel] / 1e6
    if lag.size:
        out.update(lag_p95_ms=np.percentile(lag, 95), lag_max_ms=lag.max())
    if steady.any():
        out["amortized_ms_per_video_frame"] = t["frame_ns"][steady].mean() / 1e6
    for name, key in (("pose", "pose_ns"), ("prep", "prep_ns"),
                      ("recon", "recon_ns"), ("fwd", "fwd_ns")):
        a = t[key]
        a = a[a > 0] / 1e6
        out[f"{name}_ms_when_run"] = a.mean() if a.size else np.nan
    out["decode_ms_mean"] = t["decode_ns"].mean() / 1e6
    return out


def per_repeat(df, cost, results_dir):
    gmacs = cost.get("stgcn_gmacs_per_pass")
    rows = []
    for (strat, rep), g in df.groupby(["strategy", "repeat"]):
        dur = g["duration_s"].sum()
        pipe = g["pipeline_s"].sum()
        r = {
            "strategy": strat, "repeat": rep, "clips": len(g),
            "invalid_clips": int(g["invalid"].sum()),
            # time
            "rtf": g["steady_pipeline_s"].sum() / g["steady_video_s"].sum(),
            "rtf_whole_clip": pipe / dur,
            "pipeline_fps": g["steady_video_frames"].sum() / g["steady_pipeline_s"].sum(),
            "over_budget_pct": 100 * g["over_budget_frames"].sum() / max(g["steady_frames"].sum(), 1),
            # work + compute
            "pose_per_video_s": g["pose_calls"].sum() / dur,
            "fwd_per_video_s": g["forward_passes"].sum() / dur,
            "cpu_ms_per_video_s": 1000 * g["cpu_s"].sum() / dur,
            "cpu_ms_per_frame": 1000 * g["cpu_s"].sum() / g["frames"].sum(),
            "core_util": g["cpu_s"].sum() / pipe,
            "sys_cpu_pct": g["sys_cpu_pct_mean"].mean(),
            # memory / thermal
            "rss_baseline_mb": g["rss_after_model_load_mb"].iloc[0],
            "peak_rss_mb": g["peak_rss_mb"].max(),
            "temp_max_c": g["temp_max_c"].max(),
            "freq_min_mhz": g["freq_min_mhz"].min(),
        }
        if gmacs:
            r["stgcn_gmacs_per_video_s"] = r["fwd_per_video_s"] * gmacs
        r.update(_pooled_latency(results_dir, strat, rep))

        lab = g[g["gt_fall"].astype(str).isin(["0", "1", "0.0", "1.0"])]
        if len(lab):
            pos, neg = lab[lab["gt_fall"].astype(float) == 1], lab[lab["gt_fall"].astype(float) == 0]
            r["fall_recall"] = pos["fall_detected"].mean() if len(pos) else np.nan
            r["false_alarm_rate"] = neg["fall_detected"].mean() if len(neg) else np.nan
            r["accuracy"] = lab["correct"].astype(float).mean()
        rows.append(r)
    out = pd.DataFrame(rows)

    # latency to first flag, relative to full30 on the same clips (deterministic)
    ref = (df[df.strategy == "full30"].drop_duplicates("clip_id")
           .set_index("clip_id")["first_fall_time_s"])
    ref = pd.to_numeric(ref, errors="coerce")
    for i, r in out.iterrows():
        g = df[(df.strategy == r.strategy) & (df.repeat == r.repeat)].set_index("clip_id")
        t = pd.to_numeric(g["first_fall_time_s"], errors="coerce")
        both = pd.concat([t, ref], axis=1, keys=["s", "f"]).dropna()
        out.loc[i, "flag_delay_vs_full30_s_median"] = (both.s - both.f).median() if len(both) else np.nan
        out.loc[i, "falls_missed_vs_full30"] = int(((ref.notna()) & (t.reindex(ref.index).isna())).sum())
    return out


def _iqr(x):
    x = np.asarray(x, dtype=float)
    if np.isnan(x).all():
        return np.nan, np.nan, np.nan
    return np.nanmedian(x), np.nanpercentile(x, 25), np.nanpercentile(x, 75)


def aggregate(pr):
    metrics = [c for c in pr.columns if c not in ("strategy", "repeat")]
    rows = []
    for s in [s for s in STRATEGIES if s in set(pr.strategy)]:
        g = pr[pr.strategy == s]
        r = {"strategy": s, "n_repeats": len(g)}
        for m in metrics:
            med, q1, q3 = _iqr(g[m].astype(float))
            r[m], r[m + "_q1"], r[m + "_q3"] = med, q1, q3
        rows.append(r)
    return pd.DataFrame(rows)


def _cell(r, m, nd=2):
    if m not in r.index:
        return "–"
    v, a, b = r[m], r[m + "_q1"], r[m + "_q3"]
    if pd.isna(v):
        return "–"
    if abs(a - b) < 10 ** -(nd + 1):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{a:.{nd}f}–{b:.{nd}f}]"


def _pct(summary, m):
    base = summary.loc[summary.strategy == "full30", m]
    if base.empty or not base.iloc[0]:
        return {s: "" for s in summary.strategy}
    b = base.iloc[0]
    return {r.strategy: ("baseline" if r.strategy == "full30" else f"{100 * (r[m] - b) / b:+.0f}%")
            for _, r in summary.iterrows()}


def markdown(summary, env, cost):
    L = []
    L.append("# fall-bench results\n")
    L.append(f"Device: **{env.get('device_model')}** · {env.get('os')} · {env.get('arch')} · "
             f"governor `{env.get('governor')}` · torch threads {env.get('threads')} · "
             f"k={env.get('k')} · {env.get('n_clips')} clips · {env.get('repeats')} repeats\n")
    L.append("Cells are **median [IQR]** across repeats. Pipeline time = pose + reconstruction + "
             "preprocessing + ST-GCN forward (video decoding excluded). RTF < 1 means the "
             "pipeline keeps up with real time. A frame's deadline is the time to the next camera "
             "frame (33 ms at 30 fps; 200 ms for the 5 fps strategies). Live lag = queueing delay "
             "if frames arrive in real time and one worker processes them in order; it stays small "
             "when the pipeline keeps up and keeps growing when it cannot.\n")

    def table(title, cols, delta=None):
        L.append(f"## {title}\n")
        head = ["strategy"] + [c[1] for c in cols] + ([f"Δ {delta[1]} vs full30"] if delta else [])
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        d = _pct(summary, delta[0]) if delta else {}
        for _, r in summary.iterrows():
            cells = [r.strategy] + [_cell(r, c[0], c[2] if len(c) > 2 else 2) for c in cols]
            if delta:
                cells.append(d[r.strategy])
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    table("Time", [("rtf", "RTF (steady state)", 3), ("pipeline_fps", "pipeline FPS", 1),
                   ("amortized_ms_per_video_frame", "amortized ms / video frame", 1),
                   ("lat_median_ms", "median ms / processed frame", 1),
                   ("lat_p95_ms", "p95 ms", 1), ("lat_p99_ms", "p99 ms", 1),
                   ("lat_max_ms", "max ms", 1),
                   ("over_budget_pct", "% frames missing their deadline", 1),
                   ("lag_p95_ms", "live lag p95 ms", 1), ("lag_max_ms", "live lag max ms", 0)],
          ("rtf", "RTF"))
    table("Per-stage cost (ms, only on frames where the stage ran)",
          [("pose_ms_when_run", "pose", 1), ("recon_ms_when_run", "reconstruct", 2),
           ("prep_ms_when_run", "preprocess", 2), ("fwd_ms_when_run", "ST-GCN fwd", 1),
           ("decode_ms_mean", "decode (excluded)", 1)])
    table("Compute", [("pose_per_video_s", "pose calls / video-s", 1),
                      ("fwd_per_video_s", "ST-GCN passes / video-s", 1),
                      ("cpu_ms_per_video_s", "CPU-ms / video-s", 0),
                      ("cpu_ms_per_frame", "CPU-ms / frame", 1),
                      ("core_util", "cores busy", 2),
                      ("stgcn_gmacs_per_video_s", "ST-GCN GMACs / video-s", 1)],
          ("cpu_ms_per_video_s", "CPU-ms/video-s"))
    table("Memory & thermals", [("rss_baseline_mb", "RSS after model load (MB)", 0),
                                ("peak_rss_mb", "peak RSS (MB)", 0),
                                ("temp_max_c", "max temp (°C)", 1),
                                ("freq_min_mhz", "min CPU clock (MHz)", 0),
                                ("invalid_clips", "throttled clips", 0)])
    if "accuracy" in summary.columns:
        table("Detection quality (a cheaper strategy that misses falls is not a win)",
              [("fall_recall", "fall recall", 3), ("false_alarm_rate", "false-alarm rate", 3),
               ("accuracy", "clip accuracy", 3),
               ("flag_delay_vs_full30_s_median", "median flag delay vs full30 (s)", 3),
               ("falls_missed_vs_full30", "falls full30 caught but this missed", 0)])
    if cost:
        L.append(f"ST-GCN: {cost['stgcn_gmacs_per_pass']:.3f} GMACs / "
                 f"{cost['stgcn_gflops_per_pass']:.3f} GFLOPs per forward pass, "
                 f"{cost['stgcn_params']:,} parameters. MediaPipe (TFLite) is reported in "
                 f"time / CPU-ms only.\n")
    return "\n".join(L)


def main(results_dir="results"):
    runs = os.path.join(results_dir, "benchmark_runs.csv")
    if not os.path.exists(runs):
        sys.exit(f"{runs} not found - run benchmark.py first")
    df = pd.read_csv(runs)
    cost, env = {}, {}
    for name, target in (("model_cost.json", "cost"), ("env.json", "env")):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                (cost if target == "cost" else env).update(json.load(f))

    pr = per_repeat(df, cost, results_dir)
    pr.to_csv(os.path.join(results_dir, "per_repeat.csv"), index=False)
    summary = aggregate(pr)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    md = markdown(summary, env, cost)
    with open(os.path.join(results_dir, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\nwrote {results_dir}/summary.md, summary.csv, per_repeat.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
