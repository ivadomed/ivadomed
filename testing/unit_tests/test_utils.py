import string
from ivadomed.utils import get_timestamp
from loguru import logger


def test_timestamp():
    """
    Test the timestamp function.
    """
    output = get_timestamp()
    logger.debug(output)
    assert output.count("-") == 2
    assert output.count(".") == 1
    assert output.count("T") == 1
    for I in string.ascii_uppercase:
        if I == "T":
            assert output.count(I) == 1
        else:
            assert output.count(I) == 0
    for i in string.ascii_lowercase:
        assert output.count(i) == 0
