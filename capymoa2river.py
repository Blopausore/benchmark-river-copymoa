from river.base import Classifier
from capymoa.utils import make_moa_header
import copy

class CapyMoaRiverClassifier(Classifier):
    def __init__(self, capy_model):
        self.model = capy_model
        self._initialized = False

    def _init_schema(self, x, y):
        # 1. définir les types des features (tout en numeric)
        feature_names = list(x.keys())
        feature_types = ["numeric"] * len(feature_names)

        # 2. extraire le label comme nominal
        class_labels = [str(y)]  # au début une seule classe

        # 3. créer header MOA
        header = make_moa_header(
            feature_names=feature_names,
            feature_types=feature_types,
            target_name="class",
            class_labels=class_labels,
        )

        # 4. initialiser le learner MOA
        self.model.moa_learner.setModelContext(header)
        self.model.prepareForUse()
        self._initialized = True

    def learn_one(self, x, y):
        if not self._initialized:
            self._init_schema(x, y)

        self.model.partial_fit([x], [y])
        return self

    def predict_one(self, x):
        if not self._initialized:
            return None

        y = self.model.predict([x])
        return y[0] if len(y) else None

    def predict_proba_one(self, x):
        if not self._initialized or not hasattr(self.model, "predict_proba"):
            return {}

        p = self.model.predict_proba([x])
        return p[0] if len(p) else {}

    def clone(self):
        return CapyMoaRiverClassifier(copy.deepcopy(self.model))


from river.base import Regressor
from capymoa.utils import make_moa_header
import copy

class CapyMoaRiverRegressor(Regressor):
    def __init__(self, capy_model):
        self.model = capy_model
        self._initialized = False

    def _init_schema(self, x):
        feature_names = list(x.keys())
        feature_types = ["numeric"] * len(feature_names)

        # header MOA pour la régression → pas de class_labels
        header = make_moa_header(
            feature_names=feature_names,
            feature_types=feature_types,
            target_name="target",
            class_labels=None
        )

        self.model.moa_learner.setModelContext(header)
        self.model.prepareForUse()
        self._initialized = True

    def learn_one(self, x, y):
        if not self._initialized:
            self._init_schema(x)

        self.model.partial_fit([x], [y])
        return self

    def predict_one(self, x):
        if not self._initialized:
            return None
        y = self.model.predict([x])
        return float(y[0]) if len(y) else None

    def clone(self):
        return CapyMoaRiverRegressor(copy.deepcopy(self.model))
