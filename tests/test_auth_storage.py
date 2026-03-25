from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from braingeneers.mcp.service_profiles import (
    DEFAULT_MCP_SERVICE_NAME,
    get_default_service_profile,
)
from braingeneers.mcp.token_manager import UserTokenManager
from braingeneers.utils import auth_storage


def _utc_string(delta: timedelta) -> str:
    return (datetime.utcnow() + delta).strftime("%Y-%m-%d %H:%M:%S UTC")


class AuthStorageTests(unittest.TestCase):
    def test_default_auth_dir_respects_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {auth_storage.AUTH_DIR_ENV_VAR: tmpdir}):
                self.assertEqual(auth_storage.default_auth_dir(), Path(tmpdir))

    def test_resolve_service_account_token_path_uses_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_path = Path(tmpdir) / "service-account-token.json"
            legacy_path = Path(tmpdir) / "legacy-config.json"
            legacy_path.write_text('{"access_token":"legacy"}', encoding="utf-8")

            with patch(
                "braingeneers.utils.auth_storage.service_account_token_path",
                return_value=primary_path,
            ), patch(
                "braingeneers.utils.auth_storage.legacy_service_account_token_path",
                return_value=legacy_path,
            ):
                self.assertEqual(
                    auth_storage.resolve_service_account_token_path(),
                    legacy_path,
                )

    def test_save_token_creates_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "token.json"
            auth_storage.save_token(target, {"access_token": "abc"})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"access_token": "abc"},
            )

class MCPServiceProfileTests(unittest.TestCase):
    def test_default_integrated_system_profile_exists(self) -> None:
        profile = get_default_service_profile()
        self.assertEqual(profile.name, DEFAULT_MCP_SERVICE_NAME)
        self.assertTrue(profile.remote_mcp_url.endswith("/mcp"))


class BridgeTokenManagerTests(unittest.TestCase):
    def test_get_access_token_refreshes_expiring_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "user-token.json"
            token_path.write_text(
                json.dumps(
                    {
                        "access_token": "old-access",
                        "expires_at": _utc_string(timedelta(minutes=1)),
                        "refresh_token": "refresh-token",
                        "token_url": "https://oauth2.example/realms/braingeneers/protocol/openid-connect/token",
                        "client_id": "braingeneerspy-bridge",
                        "issuer": "https://oauth2.example/realms/braingeneers",
                        "jwks_url": "https://oauth2.example/realms/braingeneers/protocol/openid-connect/certs",
                        "requested_scope": "openid offline_access mcp:tools",
                        "audience": "https://integrated-system-mcp.braingeneers.gi.ucsc.edu/",
                    }
                ),
                encoding="utf-8",
            )
            refreshed_token = {
                "access_token": "new-access",
                "refresh_token": "new-refresh-token",
                "expires_at": _utc_string(timedelta(hours=1)),
                "expires_in": 3600,
                "refresh_expires_in": 7200,
                "scope": "openid offline_access mcp:tools",
            }
            fake_response = MagicMock()
            fake_response.status_code = 200
            fake_response.json.return_value = refreshed_token

            manager = UserTokenManager(
                token_path=token_path,
                service_name="integrated-system-mcp",
            )

            with patch(
                "braingeneers.mcp.token_manager.requests.post",
                return_value=fake_response,
            ):
                access_token = manager.get_access_token()

            self.assertEqual(access_token, "new-access")
            saved = json.loads(token_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["access_token"], "new-access")
            self.assertEqual(saved["client_id"], "braingeneerspy-bridge")
            self.assertEqual(
                saved["token_url"],
                "https://oauth2.example/realms/braingeneers/protocol/openid-connect/token",
            )


if __name__ == "__main__":
    unittest.main()
