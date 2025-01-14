# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch


class TestDataset(torch.utils.data.Dataset):
    """A simple dataset for testing purposes."""

    def __init__(self, data: np.ndarray, var_names: np.ndarray | None = None) -> None:
        self.data = data
        self.var_names = var_names

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = {"X": self.data[idx, None]}
        if self.var_names is not None:
            data["var_names"] = self.var_names
        return data


try:
    import crick
except ImportError:
    crick = None
requires_crick = pytest.mark.skipif(crick is None, reason="crick is not available")
