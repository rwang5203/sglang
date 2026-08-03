"""Unit tests for conditioning ingestion: pushed prior tokens."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    _validate_prior_ids,
)

_CODEBOOK_SIZE = 16


class TestPriorIdValidation:
    def test_in_range_ids_pass(self):
        _validate_prior_ids([0, 1, _CODEBOOK_SIZE - 1], codebook_size=_CODEBOOK_SIZE)
        _validate_prior_ids(
            torch.tensor([0, _CODEBOOK_SIZE - 1]), codebook_size=_CODEBOOK_SIZE
        )

    def test_out_of_range_ids_raise_before_the_gather(self):
        with pytest.raises(ValueError, match="prior token ids"):
            _validate_prior_ids([0, _CODEBOOK_SIZE], codebook_size=_CODEBOOK_SIZE)
        with pytest.raises(ValueError, match="prior token ids"):
            _validate_prior_ids([-1], codebook_size=_CODEBOOK_SIZE)

    def test_out_of_range_tensor_ids_raise_before_the_gather(self):
        with pytest.raises(ValueError, match="prior token ids"):
            _validate_prior_ids(
                torch.tensor([0, _CODEBOOK_SIZE]), codebook_size=_CODEBOOK_SIZE
            )
        with pytest.raises(ValueError, match="prior token ids"):
            _validate_prior_ids(torch.tensor([-1]), codebook_size=_CODEBOOK_SIZE)

    def test_empty_tensor_is_skipped(self):
        _validate_prior_ids(
            torch.empty(0, dtype=torch.long), codebook_size=_CODEBOOK_SIZE
        )
