from capymoa.classifier import HoeffdingTree
from river import datasets

from capymoa2river import CapyMoaRiverClassifier

model = CapyMoaRiverClassifier(HoeffdingTree())
dataset = datasets.Elec2()

for x, y in dataset.take(5):
    print("Before learn:", model.predict_one(x))
    model.learn_one(x, y)
    print("After learn:", model.predict_one(x))
