from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.training_data import Message, TrainingData
from rasa.utils.tensorflow.constants import (
    INTENT_CLASSIFICATION,
    NUM_TRANSFORMER_LAYERS,
    BILOU_FLAG,
)


def run():
    classifier = DIETClassifier(
        {NUM_TRANSFORMER_LAYERS: 0, INTENT_CLASSIFICATION: False, BILOU_FLAG: False}
    )

    message = Message(
        text="Rasa is a company.",
        data={"entities": {"value": "Rasa", "entity": "company", "start": 0, "end": 4}},
    )

    training_data = TrainingData([message])

    classifier.train(training_data)


if __name__ == "__main__":
    run()
