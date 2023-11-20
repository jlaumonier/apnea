# content of conftest.py
import pytest
import os

@pytest.fixture(scope="function")
def base_directory(relative_path, request):
    directory = os.path.dirname(request.node.fspath)
    yield os.path.join(directory, relative_path)
