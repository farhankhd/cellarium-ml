# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import BertForMaskedLM

from .base_module import BaseModule, PredictMixin


class llomics(BaseModule, PredictMixin):
    """
    llomics model.

    **References:**

    1. `Transfer learning enables predictions in network biology (Theodoris et al.)
       <https://www.nature.com/articles/s41586-023-06139-9>`_.

    Args:
        feature_schema:
            The list of the variable names in the input data.
        model:
            The bert model.
        mlm_probability:
            Ratio of tokens to mask for masked language modeling loss.
        transform:
            If not ``None`` is used to transform the input data.
        validate_input:
            If ``True`` the input data is validated.
    """

    def __init__(
        self,
        feature_schema: Sequence,
        model: BertForMaskedLM,
        mlm_probability: float,
        transform: torch.nn.Module | None = None,
        validate_input: bool = True,
    ):
        super().__init__()
        self.feature_schema = feature_schema
        self.model = model
        self.mlm_probability = mlm_probability
        self.transform = transform
        self.validate_input = validate_input
        self.feature_ids: torch.Tensor
        # ids for the features, 0 is for padding, 1 is for mask
        self.register_buffer("feature_ids", torch.arange(2, len(feature_schema) + 2))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        feature_list = tensor_dict["var_names"]
        return (x,), {"feature_list": feature_list}

    def tokenize(self, x_ng: torch.Tensor, feature_list: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_ids.expand(x_ng.shape)
        # sort by median-scaled gene values
        sorted_indices = torch.argsort(x_ng, dim=1, descending=True)
        sorted_indices = sorted_indices[:, : self.model.config.max_position_embeddings]
        ndx = torch.arange(x_ng.shape[0], device=x_ng.device)
        input_ids = tokens[ndx[:, None], sorted_indices]
        # mask out genes with zero expression
        sorted_x_ng = x_ng[ndx[:, None], sorted_indices]
        attention_mask = sorted_x_ng != 0
        # pad genes with zero expression
        input_ids[~attention_mask] = 0
        return input_ids, attention_mask


    def bin_gene_expressions(x_ng: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
        """
        Bins gene expressions into discrete categories.

        Args:
            x_ng: A tensor containing gene expressions. Shape: [batch_size, num_genes]
            num_bins: Number of bins to categorize the expressions into.

        Returns:
            A tensor with binned gene expression values. Shape: [batch_size, num_genes]
        """
        # Flatten the tensor to perform quantization
        print(x_ng)
        print(x_ng.shape)
        flat_x_ng = x_ng.flatten()


        # Determine bin edges
        bin_edges = torch.linspace(flat_x_ng.min(), flat_x_ng.max(), num_bins + 1)

        # Convert continuous values to bin indices
        binned_x_ng = torch.bucketize(flat_x_ng, bin_edges)

        # Reshape to the original shape
        binned_x_ng = binned_x_ng.reshape(x_ng.shape)
        print(binned_x_ng)
        print(binned_x_ng.shape)

        return binned_x_ng


    def forward(self, x_ng: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert "feature_list" in kwargs, "feature_list must be provided."
        feature_list: Sequence = kwargs.pop("feature_list")

        if self.validate_input:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(feature_list, self.feature_schema), "feature_list must match the feature_schema."

        # Bin the gene expressions to use as token types
        token_type_ids = self.bin_gene_expressions(x_ng)


        if self.transform is not None:
            x_ng = self.transform(x_ng)

        input_ids, attention_mask = self.tokenize(x_ng, feature_list)

        labels = input_ids.clone()
        labels_probs = torch.full(labels.shape, self.mlm_probability, device=x_ng.device)
        labels_probs[~attention_mask] = 0
        masked_indices = torch.bernoulli(labels_probs).bool()
        labels[~masked_indices] = -100  # we only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=x_ng.device)).bool() & masked_indices
        input_ids[indices_replaced] = 1  # tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=x_ng.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(x_ng.shape[1], labels.shape, dtype=torch.long, device=x_ng.device)
        input_ids[indices_random] = random_words[indices_random]

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # Add the token_type_ids here
            labels=labels,
        )
        return output.loss


    def predict(self, x_ng: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor | None]:
        assert "feature_list" in kwargs, "feature_list must be provided."
        feature_list: Sequence = kwargs.pop("feature_list")
        output_hidden_states: bool = kwargs.pop("output_hidden_states", True)
        output_attentions: bool = kwargs.pop("output_attentions", True)

        if self.validate_input:
            assert x_ng.shape[1] == len(feature_list), "The number of x_ng columns must match the feature_list length."
            assert np.array_equal(feature_list, self.feature_schema), "feature_list must match the feature_schema."

        # Bin the gene expressions to use as token types
        token_type_ids = self.bin_gene_expressions(x_ng)

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        input_ids, attention_mask = self.tokenize(x_ng, feature_list)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # Add the token_type_ids here
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return output
