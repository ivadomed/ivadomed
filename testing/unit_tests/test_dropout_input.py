import pytest
from ivadomed.loader import loader as imed_loader
import torch


@pytest.mark.parametrize('seg_pair', [
    {"input": torch.rand((3, 5, 5))},
    {"input": torch.rand((5, 5, 5, 5))},
    {"input": (torch.rand((5, 5, 5, 2)) * torch.tensor([1, 0], dtype=torch.float)).transpose(0, -1)},
    {"input": (torch.rand((7, 7, 4)) * torch.tensor([1, 0, 1, 1], dtype=torch.float)).transpose(0, -1)}
])
def test_dropout_input(seg_pair):
    seg_pair = imed_loader.dropout_input(seg_pair)
    n_unique_values = [len(torch.unique(input_data)) == 1 for input_data in seg_pair['input']]
    # Verify that only one channel is set to zero
    assert sum(n_unique_values) == 1
    assert torch.unique(seg_pair['input'][n_unique_values.index(True)])[0] == 0
