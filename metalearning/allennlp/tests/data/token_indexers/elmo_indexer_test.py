import numpy as np
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.batch import Batch
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.fields import ListField, TextField


class TestELMoTokenCharactersIndexer(AllenNlpTestCase):
    def test_bos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token("<S>")], Vocabulary())
        expected_indices = [
            259,
            257,
            260,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
        ]
        assert indices == {"elmo_tokens": [expected_indices]}

    def test_eos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token("</S>")], Vocabulary())
        expected_indices = [
            259,
            258,
            260,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
        ]
        assert indices == {"elmo_tokens": [expected_indices]}

    def test_unicode_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token(chr(256) + "t")], Vocabulary())
        expected_indices = [
            259,
            197,
            129,
            117,
            260,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
            261,
        ]
        assert indices == {"elmo_tokens": [expected_indices]}

    def test_elmo_as_array_produces_token_sequence(self):
        indexer = ELMoTokenCharactersIndexer()
        tokens = [Token("Second"), Token(".")]
        indices = indexer.tokens_to_indices(tokens, Vocabulary())
        padded_tokens = indexer.as_padded_tensor_dict(indices, padding_lengths={"elmo_tokens": 3})
        expected_padded_tokens = [
            [
                259,
                84,
                102,
                100,
                112,
                111,
                101,
                260,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
            ],
            [
                259,
                47,
                260,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]

        assert padded_tokens["elmo_tokens"].tolist() == expected_padded_tokens

    def test_elmo_indexer_with_additional_tokens(self):
        indexer = ELMoTokenCharactersIndexer(tokens_to_add={"<first>": 1})
        tokens = [Token("<first>")]
        indices = indexer.tokens_to_indices(tokens, Vocabulary())
        expected_indices = [
            [
                259,
                2,
                260,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
                261,
            ]
        ]
        assert indices["elmo_tokens"] == expected_indices

    def test_elmo_empty_token_list(self):
        # Basic test
        indexer = ELMoTokenCharactersIndexer()
        assert {"elmo_tokens": []} == indexer.get_empty_token_list()
        # Real world test
        indexer = {"elmo": indexer}
        tokens_1 = TextField([Token("Apple")], indexer)
        targets_1 = ListField([TextField([Token("Apple")], indexer)])
        tokens_2 = TextField([Token("Screen"), Token("device")], indexer)
        targets_2 = ListField(
            [TextField([Token("Screen")], indexer), TextField([Token("Device")], indexer)]
        )
        instance_1 = Instance({"tokens": tokens_1, "targets": targets_1})
        instance_2 = Instance({"tokens": tokens_2, "targets": targets_2})
        a_batch = Batch([instance_1, instance_2])
        a_batch.index_instances(Vocabulary())
        batch_tensor = a_batch.as_tensor_dict()
        elmo_target_token_indices = batch_tensor["targets"]["elmo"]["elmo_tokens"]
        # The TextField that is empty should have been created using the
        # `get_empty_token_list` and then padded with zeros.
        empty_target = elmo_target_token_indices[0][1].numpy()
        np.testing.assert_array_equal(np.zeros((1, 50)), empty_target)
        non_empty_targets = [
            elmo_target_token_indices[0][0],
            elmo_target_token_indices[1][0],
            elmo_target_token_indices[1][1],
        ]
        for non_empty_target in non_empty_targets:
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(np.zeros((1, 50)), non_empty_target)
