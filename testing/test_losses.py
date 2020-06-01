#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.losses

import torch
import pytest
from math import isclose

from ivadomed.losses import FocalLoss, FocalDiceLoss, GeneralizedDiceLoss, MultiClassDiceLoss

@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1.0,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1/3,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     -1,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]),
     -(1 + 1/2) / 2,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]]),
     -(1 / 3),
     MultiClassDiceLoss(classes_of_interest=[1]))
])
def test_multiclassdiceloss(params):
    """Test MultiClassDiceLoss.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-3)

# TODO: add multilabel test
@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     -0.8,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     -1.,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     0.0,
     GeneralizedDiceLoss(epsilon=1e-5))
])
def test_generalizeddiceloss(params):
    """Test GeneralizedDiceLoss.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-5)
