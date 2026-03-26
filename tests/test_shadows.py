import io
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/braingeneers/iot/shadows.py"
SPEC = importlib.util.spec_from_file_location("test_shadows_module", MODULE_PATH)
SHADOWS_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SHADOWS_MODULE)
DatabaseInteractor = SHADOWS_MODULE.DatabaseInteractor


TEST_CREDENTIALS = """[strapi]
endpoint = http://example.test/api
api_key = test-api-key
"""


def _make_db(**kwargs):
    kwargs.setdefault("jwt_service_token", {"access_token": "test-token"})
    return DatabaseInteractor(credentials=io.StringIO(TEST_CREDENTIALS), **kwargs)


def _make_thing(db):
    thing = db._DatabaseInteractor__Thing(  # noqa: SLF001
        db.endpoint,
        db.token,
        db.jwt_service_token,
        db._request,
        db._auth_headers,
        db.timeout,
    )
    thing.id = 123
    thing.attributes = {"name": "mock-device", "shadow": {"state": "IDLE"}}
    return thing


def test_api_object_to_json_excludes_internal_callables():
    db = _make_db()
    thing = _make_thing(db)

    payload = thing.to_json()

    assert payload == {
        "id": 123,
        "attributes": {"name": "mock-device", "shadow": {"state": "IDLE"}},
        "jwt_service_token": {"access_token": "test-token"},
        "timeout": 30,
    }
    assert "_request" not in payload
    assert "_auth_headers" not in payload
    json.dumps(payload)


def test_api_object_str_uses_serializable_view():
    db = _make_db()
    thing = _make_thing(db)

    rendered = str(thing)

    assert "mock-device" in rendered
    assert "_request" not in rendered
    assert "_auth_headers" not in rendered


def test_database_interactor_env_settings_disable_auth_and_trust_env(monkeypatch):
    monkeypatch.setenv("BRAINGENEERS_HTTP_USE_AUTH", "false")
    monkeypatch.setenv("BRAINGENEERS_HTTP_TRUST_ENV", "false")

    db = _make_db()

    assert db.use_auth is False
    assert db.session.trust_env is False
    assert db._auth_headers() == {}


def test_database_interactor_explicit_settings_override_environment(monkeypatch):
    monkeypatch.setenv("BRAINGENEERS_HTTP_USE_AUTH", "false")
    monkeypatch.setenv("BRAINGENEERS_HTTP_TRUST_ENV", "false")

    db = _make_db(use_auth=True, trust_env=True)

    assert db.use_auth is True
    assert db.session.trust_env is True
    assert db._auth_headers() == {"Authorization": "Bearer test-token"}
