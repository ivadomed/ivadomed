from ivadomed.loader.tools.subject_aggregation import *
import pytest

@pytest.mark.parametrize('params', [
    1
])
def test_subject_aggregation(params):
    sa = SubjectAggregation()
    sa.instantiate("sub-unf01_T2w_labels-disc-manual.nii.gz")
