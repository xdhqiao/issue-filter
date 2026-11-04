from dataclasses import dataclass
import json
from typing import Any

from .config import Settings
from .errors import ServiceError


def validate_description(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceError("invalid_description", "description 必须为非空字符串")
    if len(value) > limit:
        raise ServiceError("invalid_description", f"description 最多 {limit} 个字符")
    return value


@dataclass(frozen=True)
class HistoricalIssue:
    id: str
    key_words: tuple[str, ...]
    description: str

    @classmethod
    def parse(cls, value: Any, max_chars: int):
        if not isinstance(value, dict) or set(value) != {"id", "key_words", "description"}:
            raise ServiceError("invalid_issue", "历史条目必须包含且仅包含 id、key_words、description")
        issue_id = value["id"]
        if not isinstance(issue_id, str) or not issue_id.strip() or len(issue_id) > 128:
            raise ServiceError("invalid_issue", "id 必须为 1–128 字符的非空字符串")
        keywords = value["key_words"]
        if not isinstance(keywords, list) or not keywords or len(keywords) > 100:
            raise ServiceError("invalid_issue", "key_words 必须为 1–100 项的数组")
        if any(not isinstance(k, str) or not k.strip() or len(k) > 256 for k in keywords):
            raise ServiceError("invalid_issue", "关键词必须为 1–256 字符的非空字符串")
        return cls(issue_id, tuple(keywords), validate_description(value["description"], max_chars))

    def to_dict(self) -> dict:
        return {"id": self.id, "key_words": list(self.key_words), "description": self.description}


class HistoryStore:
    def __init__(self, values: Any, settings: Settings):
        if not isinstance(values, list):
            raise ServiceError("invalid_history", "历史问题库必须为 JSON 数组")
        if len(values) > settings.max_candidates:
            raise ServiceError("invalid_history", "历史条目数超过 MAX_CANDIDATES；请调整配置")
        self.settings = settings
        self.issues = tuple(HistoricalIssue.parse(v, settings.max_description_chars) for v in values)
        self.by_id = {v.id: v for v in self.issues}
        if len(self.by_id) != len(self.issues):
            raise ServiceError("invalid_history", "历史问题 id 不能重复")

    @classmethod
    def load(cls, settings: Settings):
        # utf-8-sig also accepts JSON saved with a BOM by Windows editors.
        with settings.history_path.open(encoding="utf-8-sig") as file:
            return cls(json.load(file), settings)

    def keyword_matches(self, issue: HistoricalIssue, description: str) -> bool:
        if self.settings.keyword_case_sensitive:
            hits = (k in description for k in issue.key_words)
        else:
            text = description.casefold()
            hits = (k.casefold() in text for k in issue.key_words)
        return (any if self.settings.keyword_mode == "any" else all)(hits)

    def candidates(self, description: str) -> list[HistoricalIssue]:
        return [issue for issue in self.issues if self.keyword_matches(issue, description)]

    def validate_candidates(self, values: Any, description: str) -> list[HistoricalIssue]:
        if not isinstance(values, list) or len(values) > self.settings.max_candidates:
            raise ServiceError("invalid_candidates", "potential_historical_issue_list 必须为长度未超限的数组")
        result, seen = [], set()
        for value in values:
            issue = HistoricalIssue.parse(value, self.settings.max_description_chars)
            if issue.id in seen:
                raise ServiceError("invalid_candidates", "候选 id 不能重复")
            seen.add(issue.id)
            if self.by_id.get(issue.id) != issue:
                raise ServiceError("stale_candidates", "候选条目不在当前历史库中或内容已变化，请重新召回")
            if not self.keyword_matches(issue, description):
                raise ServiceError("invalid_candidates", "候选条目不满足当前关键词规则")
            result.append(issue)
        return result
