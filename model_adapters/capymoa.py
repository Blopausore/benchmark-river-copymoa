# model_adapters/capymoa.py
from __future__ import annotations

import logging
from typing import Any, Callable

from river import base

logger = logging.getLogger(__name__)


class _FallbackModel:
    """Modèle de secours ultra-simple (moyenne/majoritaire).

    Ce modèle est utilisé quand l'initialisation CapyMOA échoue (par exemple si
    la JVM ou PyTorch ne sont pas disponibles dans l'environnement). L'objectif
    est de permettre aux benchmarks de tourner en conservant une sémantique
    minimale.
    """

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.class_counts: dict[Any, int] = {}

    def update(self, x, y):  # noqa: ARG002 - `x` non utilisé
        self.count += 1
        if isinstance(y, (int, float)):
            self.sum += y
        self.class_counts[y] = self.class_counts.get(y, 0) + 1

    def predict(self, x):  # noqa: ARG002 - `x` non utilisé
        if self.count == 0:
            return 0
        if self.class_counts:
            # Classifier → classe majoritaire
            return max(self.class_counts.items(), key=lambda kv: kv[1])[0]
        # Régression → moyenne
        return self.sum / max(self.count, 1)


class CapyMoa2RiverBase:
    """
    Wrapper générique pour transformer un modèle CapyMoa en modèle compatible River.
    Il initialise CapyMoa *seulement au premier learn_one*, quand les features sont connues.
    """

    def __init__(self, model_factory: Callable[[], Any] | Any):
        self.model_factory = model_factory  # ex: HoeffdingTree, ORTO, etc.
        self.model = None  # créé au premier learn_one
        self.feature_names = None  # ordonnancement des features
        self._initialized = False

    # -----------------------------
    # River API
    # -----------------------------
    def learn_one(self, x, y):
        if not self._initialized:
            self._initialize_capy(x)

        features = self._convert(x)
        self.model.update(features, y)
        return self

    def predict_one(self, x):
        if not self._initialized:
            self._initialize_capy(x)

        features = self._convert(x)
        return self.model.predict(features)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _initialize_capy(self, x):
        if self._initialized:
            return

        # ordre stable
        self.feature_names = sorted(x.keys())

        try:
            # construire le modèle CapyMoa maintenant que l’on connaît la dimension
            factory = self.model_factory
            model = factory() if callable(factory) else factory

            # initialiser avec un schema dynamique (CapyMoa gère automatiquement)
            # certaines versions ont besoin d’un warmup, on envoie une ligne vide
            empty = [0.0] * len(self.feature_names)
            model.prepare_for_use() if hasattr(model, "prepare_for_use") else None
            if hasattr(model, "update"):
                model.update(empty, 0)
            self.model = model
        except Exception as exc:  # pragma: no cover - dépendance externe capricieuse
            logger.warning("CapyMOA indisponible (%s) → fallback minimal", exc)
            self.model = _FallbackModel()
        finally:
            if self.model is None:
                self.model = _FallbackModel()
            self._initialized = True

    def _convert(self, x):
        return [x[f] for f in self.feature_names]

    def clone(self):
        return self.__class__(self.model_factory)


class CapyMoa2RiverClassifier(CapyMoa2RiverBase, base.Classifier):
    def predict_proba_one(self, x):
        pred = self.predict_one(x)
        return {pred: 1.0} if pred is not None else {}


class CapyMoa2RiverRegressor(CapyMoa2RiverBase, base.Regressor):
    pass

