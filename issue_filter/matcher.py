from dataclasses import dataclass
import logging
from threading import Lock

from .config import Settings
from .errors import ServiceError
from .history import HistoricalIssue
from .nli import NLIBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredIssue:
    id: str
    score: float
    contradiction: float


class IssueMatcher:
    def __init__(self, settings: Settings, backend: NLIBackend):
        self.settings = settings
        self.backend = backend
        self._lock = Lock()
        self.ready = False

    def _run(self, operation):
        if not self._lock.acquire(timeout=self.settings.inference_lock_timeout):
            raise ServiceError("model_busy", "模型正在处理其他请求，请稍后重试", 503)
        try:
            result = operation()
            self.ready = True
            return result
        except ServiceError:
            raise
        except Exception as error:
            self.ready = False
            # Avoid logging input text or exception strings that may include it.
            logger.error("Local NLI operation failed (%s); inspect configuration and resources", type(error).__name__)
            raise ServiceError("model_unavailable", "模型加载或推理失败；请检查依赖、权重、设备和内存", 503) from None
        finally:
            self._lock.release()

    def warmup(self):
        self._run(lambda: self.backend.predict([("编译失败", "编译失败")]))

    def score(self, description: str, candidates: list[HistoricalIssue]) -> list[ScoredIssue]:
        if not candidates:
            return []
        pairs = [(description, issue.description) for issue in candidates]
        if self.settings.match_mode == "bidirectional":
            pairs += [(issue.description, description) for issue in candidates]

        def infer():
            probabilities = self.backend.predict(pairs)
            if len(probabilities) != len(pairs):
                raise RuntimeError("Model returned the wrong number of scores")
            scored = []
            for index, issue in enumerate(candidates):
                forward = probabilities[index]
                backward = probabilities[index + len(candidates)] if len(pairs) > len(candidates) else forward
                scored.append(ScoredIssue(
                    issue.id, min(forward.entailment, backward.entailment),
                    max(forward.contradiction, backward.contradiction),
                ))
            return sorted(scored, key=lambda item: (-item.score, item.id))

        return self._run(infer)

    def select(self, scored: list[ScoredIssue]) -> str | None:
        eligible = [item for item in scored if item.score >= self.settings.threshold
                    and item.contradiction <= self.settings.max_contradiction]
        return min(eligible, key=lambda item: (-item.score, item.id)).id if eligible else None

    def match(self, description: str, candidates: list[HistoricalIssue]) -> str | None:
        return self.select(self.score(description, candidates))
