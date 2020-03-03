import asyncio

from rasa.nlu import train
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter
from rasa.utils.tensorflow.constants import (
    INTENT_CLASSIFICATION,
    NUM_TRANSFORMER_LAYERS,
    BILOU_FLAG,
)


async def run():
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "LexicalSyntacticFeaturizer"},
                {
                    "name": "DIETClassifier",
                    NUM_TRANSFORMER_LAYERS: 0,
                    INTENT_CLASSIFICATION: False,
                    BILOU_FLAG: False,
                },
            ],
            "language": "en",
        }
    )

    (trained, _, persisted_path) = await train(
        _config,
        path=".",
        data="examples/restaurantbot/data/nlu.md",
        component_builder=ComponentBuilder(),
    )

    loaded = Interpreter.load(persisted_path, ComponentBuilder())
    print(loaded.parse("I like Italian."))


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(run())
