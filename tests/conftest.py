import pytest

from issue_filter.config import Settings
from issue_filter.nli import NLIProbabilities


@pytest.fixture
def history():
    return [
        {"id": "compile_issue", "key_words": ["编译"], "description": "编译不通过"},
        {"id": "no_init_issue", "key_words": ["NOT_INIT"], "description": "NOT_INIT 未定义风险"},
    ]


@pytest.fixture
def settings():
    return Settings(preload_model=False, inference_lock_timeout=0)


class FakeBackend:
    def __init__(self, values=None, error=None):
        self.values = values
        self.error = error
        self.calls = []
        self.ready = False

    def load(self):
        self.ready = True

    def predict(self, pairs):
        self.calls.append(pairs)
        if self.error:
            raise self.error
        self.load()
        return self.values if self.values is not None else [NLIProbabilities(0.02, 0.96, 0.02) for _ in pairs]


@pytest.fixture
def backend():
    return FakeBackend()
