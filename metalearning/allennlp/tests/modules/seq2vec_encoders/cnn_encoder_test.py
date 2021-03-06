import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.common.testing import AllenNlpTestCase


class TestCnnEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = CnnEncoder(embedding_dim=5, num_filters=4, ngram_filter_sizes=(3, 5))
        assert encoder.get_output_dim() == 8
        assert encoder.get_input_dim() == 5
        encoder = CnnEncoder(
            embedding_dim=5, num_filters=4, ngram_filter_sizes=(3, 5), output_dim=7
        )
        assert encoder.get_output_dim() == 7
        assert encoder.get_input_dim() == 5

    def test_can_construct_from_params(self):
        params = Params({"embedding_dim": 5, "num_filters": 4, "ngram_filter_sizes": [3, 5]})
        encoder = CnnEncoder.from_params(params)
        assert encoder.get_output_dim() == 8
        params = Params(
            {"embedding_dim": 5, "num_filters": 4, "ngram_filter_sizes": [3, 5], "output_dim": 7}
        )
        encoder = CnnEncoder.from_params(params)
        assert encoder.get_output_dim() == 7

    def test_forward_does_correct_computation(self):
        encoder = CnnEncoder(embedding_dim=2, num_filters=1, ngram_filter_sizes=(1, 2))
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.0}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(encoder)
        input_tensor = torch.FloatTensor([[[0.7, 0.8], [0.1, 1.5]]])
        encoder_output = encoder(input_tensor, None)
        assert_almost_equal(
            encoder_output.data.numpy(), numpy.asarray([[1.6 + 1.0, 3.1 + 1.0]]), decimal=6
        )

    def test_forward_runs_with_larger_input(self):
        encoder = CnnEncoder(
            embedding_dim=7, num_filters=13, ngram_filter_sizes=(1, 2, 3, 4, 5), output_dim=30
        )
        tensor = torch.rand(4, 8, 7)
        assert encoder(tensor, None).size() == (4, 30)

    def test_forward_respects_masking(self):
        # seed 1 fails on the old cnn encoder code
        torch.manual_seed(1)
        encoder = CnnEncoder(embedding_dim=7, num_filters=13, ngram_filter_sizes=(1, 2, 3, 4, 5))
        init = Initializer.from_params(Params({"type": "normal", "mean": 0.0, "std": 10}))
        initializer = InitializerApplicator([(".*", init)])
        initializer(encoder)
        tokens = torch.ones(4, 8, 7)
        padded_tokens = torch.nn.functional.pad(tokens.transpose(1, 2), (0, 2), value=5).transpose(
            1, 2
        )
        mask = (
            torch.where(
                padded_tokens == 5, torch.zeros_like(padded_tokens), torch.ones_like(padded_tokens)
            )
            .bool()
            .any(dim=2)
        )
        regular_output = encoder.forward(tokens=tokens, mask=None)
        masked_output = encoder.forward(tokens=padded_tokens, mask=mask)
        assert_almost_equal(regular_output.data.numpy(), masked_output.data.numpy(), decimal=5)
