# model_adapters/capymoa.py
from __future__ import annotations

class CapyMoa2RiverBase:
    """
    Wrapper générique pour transformer un modèle CapyMoa en modèle compatible River.
    Il initialise CapyMoa *seulement au premier learn_one*, quand les features sont connues.
    """

    def __init__(self, model_cls):
        self.model_cls = model_cls   # ex: HoeffdingTree, ORTO, etc.
        self.model = None            # créé au premier learn_one
        self.feature_names = None    # ordonnancement des features
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
        # ordre stable
        self.feature_names = sorted(x.keys())

        # construire le modèle CapyMoa maintenant que l’on connaît la dimension
        self.model = self.model_cls()

        # initialiser avec un schema dynamique (CapyMoa gère automatiquement)
        # certaines versions ont besoin d’un warmup, on envoie une ligne vide
        empty = [0.0] * len(self.feature_names)
        self.model.prepare_for_use() if hasattr(self.model, "prepare_for_use") else None
        if hasattr(self.model, "update"):
            self.model.update(empty, 0)

        self._initialized = True

    def _convert(self, x):
        return [x[f] for f in self.feature_names]


class CapyMoa2RiverClassifier(CapyMoa2RiverBase):
    pass


class CapyMoa2RiverRegressor(CapyMoa2RiverBase):
    pass
