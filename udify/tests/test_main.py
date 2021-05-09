from allennlp.data import instance
from typing import Dict, List
from allennlp.common.util import import_module_and_submodules
import torch
import pytest
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive
from pathlib import Path
from udify.models.udify_model import OUTPUTS  # type: ignore


@pytest.fixture(scope="session")
def model():
    import_module_and_submodules("udify")
    archive = load_archive(
        str((Path(__file__).parent / ".." / "data" / "archive").absolute())
    )
    config = archive.config
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    def proc(tokens_list: List[List[str]]) -> Dict:
        archive.model.eval()
        with torch.no_grad():
            instances = [
                dataset_reader.text_to_instance(tokens) for tokens in tokens_list
            ]
            outputs = archive.model.forward_on_instances(instances)
        return outputs

    return proc


def test_call(model):
    output = model([["Who", "are", "you", "?"]])[0]
    for k in [
        OUTPUTS.predicted_dependencies,
        OUTPUTS.upos,
        OUTPUTS.lemmas,
        OUTPUTS.predicted_heads,
    ]:
        assert k in output

