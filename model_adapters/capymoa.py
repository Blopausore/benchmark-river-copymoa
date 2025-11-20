from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

from river import base

logger = logging.getLogger(__name__)


from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

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
        self.schema = None
        self.feature_names = None  # ordonnancement des features
        self._initialized = False
        self._using_fallback = False

    # -----------------------------
    # River API
    # -----------------------------
    def learn_one(self, x, y):
        if not self._initialized:
            self._initialize_capy(x, y)

        if self._using_fallback:
            self.model.update(x, y)
            return self

        instance = self._build_instance(x, y)
        if instance is None:
            # schéma non compatible → fallback minimal
            self.model.update(x, y)
            return self

        trainer = getattr(self.model, "train", None)
        if callable(trainer):
            trainer(instance)
        elif hasattr(self.model, "update"):
            self.model.update(instance.x, y)
        return self

    def predict_one(self, x):
        if not self._initialized:
            # Sans cible, on ne peut pas initialiser correctement le schéma CapyMOA
            return None

        if self._using_fallback:
            return self.model.predict(x)

        instance = self._build_instance(x, y=None)
        if instance is None:
            return None

        predictor = getattr(self.model, "predict_proba", None)
        if callable(predictor):
            return predictor(instance)

        predict = getattr(self.model, "predict", None)
        if callable(predict):
            return predict(instance)
        return None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _initialize_capy(self, x, y):
        if self._initialized:
            return

        # ordre stable
        self.feature_names = sorted(x.keys())

        try:
            # construire le modèle CapyMoa maintenant que l’on connaît la dimension
            factory = self.model_factory
            schema = self._build_schema(first_label=y)

            model = self._create_model(factory, schema)
            self.schema = schema
            self.model = model
        except Exception as exc:  # pragma: no cover - dépendance externe capricieuse
            logger.warning("CapyMOA indisponible (%s) → fallback minimal", exc)
            self.model = _FallbackModel()
            self._using_fallback = True
        finally:
            if self.model is None:
                self.model = _FallbackModel()
                self._using_fallback = True
            self._initialized = True

    def _convert(self, x):
        return np.array([x[f] for f in self.feature_names], dtype=float)

    def _default_labels(self) -> Sequence[str]:
        base_numeric = [str(i) for i in range(100)]
        return ["False", "True", "No", "Yes", "-1"] + base_numeric

    def _build_schema(self, first_label: Any | None):
        from capymoa.stream import Schema

        feature_names: list[str] = list(self.feature_names)
        if isinstance(self, CapyMoa2RiverClassifier):
            labels = set(self._default_labels())
            if first_label is not None:
                labels.add(str(first_label))
            return Schema.from_custom(
                feature_names=feature_names,
                values_for_class_label=sorted(labels),
                dataset_name="RiverStream",
                target_attribute_name="target",
                target_type="categorical",
            )

        return Schema.from_custom(
            feature_names=feature_names,
            dataset_name="RiverStream",
            target_attribute_name="target",
            target_type="numeric",
        )

    def _create_model(self, factory: Callable[[], Any] | Any, schema):
        if callable(factory):
            try:
                model = factory(schema=schema)
            except TypeError:
                model = factory()
        else:
            model = factory
        if callable(getattr(model, "prepare_for_use", None)):
            model.prepare_for_use()
        return model

    def _build_instance(self, x: dict, y: Any | None):
        from capymoa import instance as capy_instance

        if self.schema is None:
            return None

        features = self._convert(x)

        try:
            if isinstance(self, CapyMoa2RiverClassifier):
                if y is None:
                    return capy_instance.Instance.from_array(self.schema, features)

                label = str(y)
                try:
                    y_index = self.schema.get_index_for_label(label)
                except Exception:
                    logger.warning("Label inconnu pour le schéma CapyMOA: %s", label)
                    raise
                return capy_instance.LabeledInstance.from_array(
                    self.schema, features, y_index
                )

            # Regression
            if y is None:
                return capy_instance.Instance.from_array(self.schema, features)
            return capy_instance.RegressionInstance.from_array(
                self.schema, features, float(y)
            )
        except Exception as exc:
            logger.warning("Conversion vers Instance CapyMOA impossible: %s", exc)
            self._using_fallback = True
            self.model = _FallbackModel()
            return None

    def clone(self):
        return self.__class__(self.model_factory)

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback


class CapyMoa2RiverClassifier(CapyMoa2RiverBase, base.Classifier):
    def predict_one(self, x):
        proba = self.predict_proba_one(x)
        if not proba:
            return None
        return max(proba, key=proba.get)

    def predict_proba_one(self, x):
        raw = super().predict_one(x)
        if raw is None:
            return {}

        if isinstance(raw, dict):
            return raw

        if isinstance(raw, (list, tuple, np.ndarray)) or hasattr(raw, "__len__"):
            labels = self.schema.get_label_values() if self.schema else None
            if labels is None:
                return {}
            probs = list(raw)
            return {label: float(prob) for label, prob in zip(labels, probs)}

        return {raw: 1.0}


class CapyMoa2RiverRegressor(CapyMoa2RiverBase, base.Regressor):
    pass