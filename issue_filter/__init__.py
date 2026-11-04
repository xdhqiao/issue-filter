import hmac

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import Settings
from .errors import ServiceError
from .history import HistoryStore, validate_description
from .matcher import IssueMatcher
from .nli import TransformersNLI


def create_app(settings: Settings | None = None, *, backend=None, history=None) -> Flask:
    settings = settings or Settings.from_env()
    store = HistoryStore.load(settings) if history is None else HistoryStore(history, settings)
    matcher = IssueMatcher(settings, TransformersNLI(settings) if backend is None else backend)
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.extensions["issue_matcher"] = matcher
    app.extensions["history_store"] = store

    @app.before_request
    def authenticate():
        if settings.api_key and request.path.startswith("/api/"):
            actual = request.headers.get("Authorization", "").encode("utf-8")
            expected = f"Bearer {settings.api_key}".encode("utf-8")
            if not hmac.compare_digest(actual, expected):
                raise ServiceError("unauthorized", "需要有效的 Bearer API key", 401)

    def read_body(required: set[str]):
        if not request.is_json:
            raise ServiceError("unsupported_media_type", "请使用 Content-Type: application/json", 415)
        body = request.get_json()
        if not isinstance(body, dict) or set(body) != required:
            raise ServiceError("invalid_request", f"JSON 对象必须包含且仅包含：{', '.join(sorted(required))}")
        description = validate_description(body["description"], settings.max_description_chars)
        return body, description

    @app.post("/api/v1/potential-issues")
    def potential_issues():
        _, description = read_body({"description"})
        return jsonify([issue.to_dict() for issue in store.candidates(description)])

    @app.post("/api/v1/match")
    def match_issue():
        body, description = read_body({"description", "potential_historical_issue_list"})
        candidates = store.validate_candidates(body["potential_historical_issue_list"], description)
        return jsonify(matched_issue_id=matcher.match(description, candidates))

    @app.get("/healthz")
    def health():
        return jsonify(status="ok", model_loaded=matcher.backend.ready, historical_issue_count=len(store.issues))

    @app.get("/readyz")
    def ready():
        available = matcher.ready and matcher.backend.ready
        return jsonify(status="ready" if available else "model_not_ready"), 200 if available else 503

    @app.errorhandler(ServiceError)
    def service_error(error):
        response = jsonify(error={"code": error.code, "message": str(error)})
        response.status_code = error.status
        if error.status == 503:
            response.headers["Retry-After"] = "1"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        response = error.get_response()
        response.data = app.json.dumps({"error": {"code": error.name.lower().replace(" ", "_"),
                                                   "message": error.description}})
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("Unhandled API exception (%s)", type(error).__name__)
        return jsonify(error={"code": "internal_error", "message": "服务内部错误"}), 500

    if settings.preload_model:
        matcher.warmup()
    return app
