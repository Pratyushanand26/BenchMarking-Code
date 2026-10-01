# fall-bench

Benchmarks five ST-GCN fall-detection strategies on a Raspberry Pi: **time** and **compute cost**
(plus detection quality, so a cheap strategy that misses falls doesn't look like a win).

| strategy | what it does | pose calls / video-s* | ST-GCN passes / video-s* |
|---|---|---|---|
| `full30` | native fps, 30-frame window, inference every frame (baseline) | 30 | 30 |
| `fps_hold` | simulates a **5 fps camera**; zero-order-hold rebuilds 30 frames | 5 | 5 |
| `fps_interp` | simulates a 5 fps camera; linear interpolation rebuilds 30 frames | 5 | 5 |
| `cost_skip` | MediaPipe every *k* frames, stale pose reused, ST-GCN every frame | 30/k | 30 |
| `cost_stride` | MediaPipe every frame, ST-GCN every *k* frames | 30 | 30/k |

\*at 30 fps, k = 3 → 10. `python tests/smoke_test.py` asserts these counts.

## Layout

```
requirements.txt        pinned for aarch64 (mediapipe 0.10.14 = last build with aarch64 wheels AND mp.solutions)
benchmark.py            orchestrator: 5 strategies x R repeats, shuffled, cooled, throttle-checked, resumable
fallbench/              model.py classifier.py pose.py strategies.py runner.py telemetry.py summarize.py testing.py ...
scripts/                setup_pi.sh  governor.sh  run_benchmark.sh
tests/smoke_test.py     no videos / weights needed
weights/                put tsstg-model.pth here           (gitignored)
videos/                 put your clips here                (gitignored)
results/                everything the benchmark writes    (gitignored)
```

## Run it on the Pi — exact steps

**0. Get the repo onto the Pi** (pick one)
```bash
# from your laptop, if you unzip/push it yourself:
scp fall-bench.zip pi@<PI_IP>:~            # then on the Pi:  unzip fall-bench.zip
# or push to GitHub and on the Pi:         git clone <your-repo-url> fall-bench
```

**1. SSH in and enter the folder**
```bash
ssh pi@<PI_IP>
cd ~/fall-bench
```

**2. One-time setup** (installs system libs + a venv + pinned Python packages; takes a few minutes)
```bash
bash scripts/setup_pi.sh
```
Needs a **64-bit OS** (`uname -m` must print `aarch64`) and **Python 3.9–3.12**
(Raspberry Pi OS Bookworm ships 3.11 — fine. On 3.13 the script prints a `uv` workaround.)

**3. Copy the model weights and the videos** (run these on your *laptop*, not the Pi)
```bash
scp path/to/tsstg-model.pth        pi@<PI_IP>:~/fall-bench/weights/
scp -r path/to/your_clips/*.mp4    pi@<PI_IP>:~/fall-bench/videos/
```
The weights are **not** in this repo (the upstream project doesn't ship them either); use the
`tsstg-model.pth` you already ran locally. Videos stay on the Pi's local disk so SD/network
I/O doesn't pollute timings.

Ground truth: clip names starting `fall` count as falls, `adl` as non-falls. For anything else add
`videos/labels.csv` with header `clip_id,fall[,fall_onset_sec]`. Unlabeled clips still get timing
numbers; only the accuracy columns are blank.

**4. Make the Pi stable for benchmarking**
```bash
bash scripts/governor.sh performance      # lock CPU clock
```
Use the official power supply and a heatsink/fan — under-voltage or thermal throttling invalidates
a run (the benchmark detects it, discards the run and retries).

**5. Sanity check (30 s, no videos needed)**
```bash
source .venv/bin/activate
python tests/smoke_test.py                # must end with: all strategies produce the expected work counts
```

**6. Dry run on 3 clips** — confirms weights, MediaPipe and videos all work
```bash
python benchmark.py --repeats 1 --limit 3
```

**7. Full benchmark** — inside `tmux` so an SSH drop doesn't kill it
```bash
tmux new -s bench
bash scripts/run_benchmark.sh             # 5 strategies x 5 repeats, all clips
# detach: Ctrl-b then d      reattach later: tmux attach -t bench
```
It is **resumable**: if interrupted, re-run the same command and finished runs are skipped.
Rough duration ≈ `repeats × Σ(RTF of each strategy) × total video length` + cool-downs.
Useful flags: `--repeats 3`, `--k 5`, `--threads 4`, `--cpus 0-3` (taskset pinning), `--cool-temp 50`.

**8. Read the results, then restore the governor**
```bash
cat results/summary.md
bash scripts/governor.sh ondemand
# from your laptop:
scp -r pi@<PI_IP>:~/fall-bench/results ./results_pi
```

## What is measured (and how)

**Time** — `time.perf_counter_ns` around each stage: pose extraction, window reconstruction,
preprocessing, ST-GCN forward. *Pipeline time* = their sum per frame; **video decoding is timed
separately and excluded** (a real camera hands you frames). Reported: mean / median / p95 / p99 / max
ms per processed frame (pooled over all steady-state frames), **RTF** = pipeline seconds ÷ video
seconds (<1 keeps up with real time), sustained pipeline FPS, and the % of frames that miss their
deadline. A frame's deadline is the time to the next camera frame: 33 ms at 30 fps, **200 ms for the
5 fps strategies** (only every 6th frame is a real camera frame there). **Live lag** simulates frames
arriving in real time into a single worker: it stays small if the pipeline keeps up and grows without
bound if it can't. *Steady state* = frames from the first inference onward (RTF is steady-state; the
whole-clip value is kept as `rtf_whole_clip`). Warm-up (model load, 10 dummy forwards, dummy pose
frames) happens before anything is timed.

**Compute cost**
- **CPU-ms per frame / per video-second** from `time.process_time_ns` (user+sys, all threads, this
  process only — other processes can't pollute it), and **cores busy** = CPU time ÷ wall time.
  System-wide CPU% at 1 Hz is logged only as a cross-check.
- **Work counts**: pose calls and ST-GCN passes per video-second (exact).
- **ST-GCN GMACs/s** = passes/s × GMACs per pass (measured once with torch's `FlopCounterMode`).
  MediaPipe is a TFLite graph, so its cost is reported in time and CPU-ms, not FLOPs.
- **Memory**: RSS after model load, and true peak RSS (`getrusage` high-water mark). Each
  (strategy, repeat) is its own process so the peak belongs to one strategy only.

**Detection quality** — fall recall, false-alarm rate, clip accuracy, median delay of the first fall
flag vs `full30`, and falls `full30` caught that the strategy missed.

**Pi controls** — SoC temp, CPU clock, throttle flags (`vcgencmd get_throttled` before/after each
clip), governor, all recorded; any run with active/new throttling or under-voltage is discarded and
re-run. Strategy order is shuffled within every repeat, with a cool-down to `--cool-temp` before each
run. GC is off during the frame loop; nothing is printed or written inside it.

## Output files

| file | content |
|---|---|
| `results/summary.md` | the comparison tables: median [IQR] over repeats, % change vs `full30` |
| `results/summary.csv`, `per_repeat.csv` | same numbers, machine-readable |
| `results/benchmark_runs.csv` | one row per clip × strategy × repeat (all raw metrics) |
| `results/traces/*.npz` | per-frame timings, used for pooled percentiles |
| `results/env.json`, `model_cost.json` | device, OS, library versions, settings; ST-GCN GFLOPs |

## Differences from the original `Stgcn-DH307` scripts (deliberate)

- `fps_hold` / `fps_interp` now run MediaPipe **only on the ~5 fps samples**. The old code ran pose on
  all 30 frames and then discarded 25, so those modes showed no compute saving.
- Time is split per stage; the old `avg_inference_compute_ms` was whole-frame time.
- CPU is process CPU-time, not system-wide `cpu_percent`; peak RSS is the true high-water mark.
- Warm-up and the first 29 no-inference frames are excluded from steady-state statistics.
- Weights are checked on load; a mismatching checkpoint raises instead of silently benchmarking random weights.

## Troubleshooting

- `mediapipe has no legacy mp.solutions API` → you installed a newer mediapipe; re-run `pip install -r requirements.txt`.
- `ImportError: libGL.so.1` → `sudo apt-get install -y libgl1 libglib2.0-0`.
- `[WARN] CPU governor is 'ondemand'` → `bash scripts/governor.sh performance`.
- Runs marked `INVALID(throttled)` → check `vcgencmd get_throttled`, power supply, cooling; raise `--cool-temp` margin.
- `no video files found` → clips must be `.mp4/.avi/.mov/.mkv` directly inside `videos/`.
