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
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-3)
