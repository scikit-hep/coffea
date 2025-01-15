import os

import pytest


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))
