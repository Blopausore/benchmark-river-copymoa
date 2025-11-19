from __future__ import annotations

import os
import json
import itertools
import logging
import multiprocessing
import pandas as pd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# WORKER FUNCTION — Executed inside each subprocess
# ---------------------------------------------------------
def run_dataset(model_str: str, no_dataset: int, no_track: int):
    """
    This function is executed inside each worker.
    All heavy objects (TRACKS, MODELS, datasets) are imported *inside*
    so they are NOT pickled and do not carry I/O handles between processes.
    """

    # Import heavy objects INSIDE the worker → NO pickling
    from config import MODELS, TRACKS, N_CHECKPOINTS
    import copy
    from river import metrics

    # Output file for this worker
    os.makedirs("tmp_results", exist_ok=True)
    track = TRACKS[no_track]
    dataset = track.datasets[no_dataset]
    model = MODELS[track.name][model_str].clone()

    out_path = f"tmp_results/{track.name}_{dataset.__class__.__name__}_{model_str}.jsonl"

    # Deepcopy inside the worker (safe, no pickling afterwards)
    track_local = copy.deepcopy(track)

    cumulative_time = 0.0

    with open(out_path, "w") as f:
        for state in track_local.run(model, dataset, n_checkpoints=N_CHECKPOINTS):
            cumulative_time += state["Time"].total_seconds()

            res = {
                "step": state["Step"],
                "track": track.name,
                "model": model_str,
                "dataset": dataset.__class__.__name__,
                "Memory in Mb": state["Memory"] / 1024**2,
                "Time in s": cumulative_time,
            }

            # Convert metrics to floats
            for k, v in state.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = float(v.get())

            f.write(json.dumps(res) + "\n")

    # Worker returns only a simple STRING → always pickle-safe
    return out_path


# ---------------------------------------------------------
# RUN WHOLE TRACK WITH MULTIPROCESSING
# ---------------------------------------------------------
def run_track(models: list[str], no_track: int, n_workers: int = 10):
    """
    Launch benchmark in parallel.
    Each worker writes results in tmp_results/*.jsonl.
    Then main process merges everything.
    """

    print(f"\nRunning experiments with {n_workers} workers...\n")

    # Build cartesian product of (model, dataset, track)
    # Note: TRACKS is imported ONLY in main process here → safe
    from config import TRACKS

    track = TRACKS[no_track]
    runs = list(itertools.product(models, range(len(track.datasets)), [no_track]))

    # Launch workers
    with multiprocessing.Pool(processes=n_workers) as pool:
        file_paths = pool.starmap(run_dataset, runs)

    # Merge results
    combined_rows = []
    for path in file_paths:
        with open(path) as f:
            for line in f:
                combined_rows.append(json.loads(line))

    csv_name = track.name.replace(" ", "_").lower()
    pd.DataFrame(combined_rows).to_csv(f"{csv_name}.csv", index=False)

    print(f"\nSaved results → {csv_name}.csv\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Import heavy configuration only in main
    from config import MODELS, TRACKS

    # Choose track
    track_no = 1  # Multiclass classification
    track = TRACKS[track_no]

    # Define the models to benchmark
    selected_models = list(MODELS[track.name].keys())

    # Save details.json
    details = {track.name: {"Dataset": {}, "Model": {}}}

    for dataset in track.datasets:
        details[track.name]["Dataset"][dataset.__class__.__name__] = repr(dataset)
    for name, model in MODELS[track.name].items():
        details[track.name]["Model"][name] = repr(model)

    with open("details.json", "w") as f:
        json.dump(details, f, indent=2)

    # Run benchmark
    run_track(models=selected_models, no_track=track_no, n_workers=50)
