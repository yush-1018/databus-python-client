"""Tests for HTTP retry session helper and CLI options."""

import requests
from click.testing import CliRunner
from requests.adapters import HTTPAdapter

from databusclient.api.utils import get_http_session, fetch_databus_jsonld
from databusclient.cli import app


def test_get_http_session_defaults():
    session = get_http_session()
    assert isinstance(session, requests.Session)

    http_adapter = session.adapters.get("http://")
    https_adapter = session.adapters.get("https://")

    assert isinstance(http_adapter, HTTPAdapter)
    assert isinstance(https_adapter, HTTPAdapter)
    assert http_adapter.max_retries.total == 3


def test_get_http_session_custom_retries():
    session = get_http_session(retries=5, backoff_factor=1.0)
    http_adapter = session.adapters.get("https://")

    assert http_adapter.max_retries.total == 5
    assert http_adapter.max_retries.backoff_factor == 1.0


def test_get_http_session_with_jitter():
    session = get_http_session(retries=4, backoff_factor=0.5, backoff_jitter=0.2)
    http_adapter = session.adapters.get("https://")
    assert http_adapter.max_retries.total == 4
    if hasattr(http_adapter.max_retries, "backoff_jitter"):
        assert http_adapter.max_retries.backoff_jitter == 0.2


def test_fetch_databus_jsonld_with_custom_session():
    class MockResponse:
        status_code = 200
        text = '{"@context": "https://databus.dbpedia.org/context.jsonld"}'

        def raise_for_status(self):
            pass

    class MockSession(requests.Session):
        def get(self, url, headers=None, timeout=None):
            assert timeout == 15
            assert headers["Accept"] == "application/ld+json"
            assert headers["X-API-KEY"] == "test_key"
            return MockResponse()

    session = MockSession()
    result = fetch_databus_jsonld(
        "https://databus.dbpedia.org/test",
        databus_key="test_key",
        session=session,
        timeout=15,
    )
    assert result == '{"@context": "https://databus.dbpedia.org/context.jsonld"}'


def test_cli_delete_force_and_retry_options(monkeypatch):
    captured = {}

    def fake_api_delete(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("databusclient.cli.api_delete", fake_api_delete)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "delete",
            "https://databus.dbpedia.org/user/group/artifact/version",
            "--databus-key",
            "test-key",
            "--force",
            "--retries",
            "5",
            "--request-timeout",
            "45",
        ],
    )

    assert result.exit_code == 0
    assert captured["force"] is True
    assert captured["timeout"] == 45
    assert captured["session"].adapters["https://"].max_retries.total == 5


def test_cli_delete_max_retries_alias(monkeypatch):
    captured = {}

    def fake_api_delete(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("databusclient.cli.api_delete", fake_api_delete)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "delete",
            "https://databus.dbpedia.org/user/group/artifact/version",
            "--databus-key",
            "test-key",
            "--force",
            "--max-retries",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert captured["session"].adapters["https://"].max_retries.total == 7


def test_retry_local_server_503_twice_then_200():
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class FlakyHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_GET(self):
            FlakyHandler.attempts += 1
            if FlakyHandler.attempts <= 2:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Service Unavailable")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Success!")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FlakyHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

    try:
        session = get_http_session(retries=3, backoff_factor=0.01)
        resp = session.get(f"http://127.0.0.1:{port}/")
        assert resp.status_code == 200
        assert resp.text == "Success!"
        assert FlakyHandler.attempts == 3
    finally:
        server.shutdown()


def test_retry_local_server_persistent_503_limit():
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Persistent503Handler(BaseHTTPRequestHandler):
        attempts = 0

        def do_GET(self):
            Persistent503Handler.attempts += 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"Service Unavailable")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Persistent503Handler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

    try:
        # retries=2: 1 initial attempt + 2 retries = 3 attempts total
        session = get_http_session(retries=2, backoff_factor=0.01)
        resp = session.get(f"http://127.0.0.1:{port}/")
        assert resp.status_code == 503
        assert Persistent503Handler.attempts == 3
    finally:
        server.shutdown()


def test_file_stat_fetch_honors_retry_settings_when_checksum_omitted():
    import hashlib
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from databusclient.api.deploy import create_dataset

    class FlakyFileHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_GET(self):
            FlakyFileHandler.attempts += 1
            if FlakyFileHandler.attempts == 1:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Temporary Error")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"data content for sha")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FlakyFileHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

    try:
        session = get_http_session(retries=2, backoff_factor=0.01)
        # Distribution string without checksum or size:
        dist_url = f"http://127.0.0.1:{port}/test.ttl"
        dataset = create_dataset(
            version_id="https://databus.dbpedia.org/user/group/artifact/2026-01-01",
            artifact_version_title="Test Title",
            artifact_version_abstract="Test Abstract",
            artifact_version_description="Test Description",
            license_url="https://example.org/license",
            distributions=[dist_url],
            session=session,
            timeout=10,
        )
        assert FlakyFileHandler.attempts == 2
        dist = dataset["@graph"][-1]["distribution"][0]
        expected_sha = hashlib.sha256(b"data content for sha").hexdigest()
        assert dist["sha256sum"] == expected_sha
        assert dist["byteSize"] == len(b"data content for sha")
    finally:
        server.shutdown()
