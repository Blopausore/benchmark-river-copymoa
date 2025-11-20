from __future__ import annotations

import os
import json
import pandas as pd
import itertools
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def run_capy_model(model_name, model, dataset, track_name, n_checkpoints, out_dir):
    """
    Exécute un modèle CapyMoa packagé avec CapyMoa2RiverClassifier/Regressor
    sur un dataset River, en enregistrant les résultats dans un fichier .jsonl.
    """

    # Output file for this run
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{track_name}_{dataset.__class__.__name__}_{model_name}.jsonl"
    )

    from river import metrics

    # Prépare la boucle de benchmarking
    step = 0
    cumulative_time = 0.0

    # On utilise le track River directement
    # → pas de multiprocessing ici, tout en séquentiel, sûr à 100%
    try:
        for state in track_name.run(model, dataset, n_checkpoints=n_checkpoints):
            cumulative_time += state["Time"].total_seconds()

            # Safe dict
            res = {
                "step": state["Step"],
                "track": track_name.name,
                "model": model_name,
                "dataset": dataset.__class__.__name__,
                "Memory in Mb": state["Memory"] / 1024**2,
                "Time in s": cumulative_time,
                "using_fallback": getattr(model, "using_fallback", False),
            }

            # Extraire uniquement les métriques
            for k, v in state.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = float(v.get())

            # Write line
            with open(out_path, "a") as f:
                f.write(json.dumps(res) + "\n")
    except Exception as exc:
        logger.error("Echec sur %s/%s: %s", dataset.__class__.__name__, model_name, exc)
            # Write line
        with open(out_path, "a") as f:
                f.write(json.dumps({
                "track": track_name.name,
                "model": model_name,
                "dataset": dataset.__class__.__name__,
                "error": str(exc),
            }) + "\n")

    return out_path


def run_track_capy(track_name, track, models, n_checkpoints, out_dir):
    """
    Exécute tous les modèles CapyMoa pour une track donnée.
    """

    print(f"\n=== Running track: {track_name} (CapyMoa) ===\n")

    file_paths = []

    # Produit cartésien (models × datasets)
    jobs = itertools.product(models.items(), track.datasets)

    for (model_name, model), dataset in jobs:
        print(f"\n⚡ Running {model_name} on {dataset.__class__.__name__}")

        # Cloner le modèle CapyMoa wrapper, sinon l'état reste entre datasets
        model_instance = model.clone() if hasattr(model, "clone") else model.__class__()
        
        
        path = run_capy_model(
            model_name=model_name,
            model=model_instance,
            dataset=dataset,
            track_name=track,
            n_checkpoints=n_checkpoints,
            out_dir=out_dir,
        )
        file_paths.append(path)

    return file_paths


if __name__ == "__main__":
    import config

    N_CHECKPOINTS = config.N_CHECKPOINTS
    TRACKS = config.TRACKS
    MODELS_CAPY = config.MODELS_CAPY

    OUT_DIR = "tmp_results_capymoa"

    # On parcourt les tracks River comme d’habitude
    TRACK_NAMES = {
        "Binary classification": TRACKS[0],
        "Multiclass classification": TRACKS[1],
        "Regression": TRACKS[2],
    }

    # ----------------------------
    # 1. Lancer tous les modèles
    # ----------------------------
    all_results = {}

    for track_name, track in TRACK_NAMES.items():
        print(f"\n===== TRACK: {track_name} =====")

        models = MODELS_CAPY.get(track_name, {})
        file_paths = run_track_capy(
            track_name, track, models, N_CHECKPOINTS, OUT_DIR
        )
        all_results[track_name] = file_paths

    # ----------------------------
    # 2. Fusionner les JSONL → CSV
    # ----------------------------
    for track_name, file_list in all_results.items():
        combined = []
        for path in file_list:
            with open(path) as f:
                for line in f:
                    combined.append(json.loads(line))

        df = pd.DataFrame(combined)
        csv_name = track_name.replace(" ", "_").lower() + "_capymoa.csv"
        df.to_csv(csv_name, index=False)

        print(f"✔ Saved CSV: {csv_name}")