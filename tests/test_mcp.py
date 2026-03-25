from __future__ import annotations

import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt

from braingeneers.mcp.auth import (
    Auth0TokenVerifier,
    MCPAuthRuntimeConfig,
    OIDCTokenVerifier,
    extract_claim_values,
)
from braingeneers.mcp.iam import (
    AuthorizationDecision,
    AuthorizationPolicy,
    IAMCommandSpec,
    IAMPolicyAdapter,
    PolicyLoader,
    PolicyValidationError,
)
from braingeneers.mcp.models import Principal


TEST_COMMAND_SPEC = IAMCommandSpec(
    command_registry={
        "PING": {"command_group": "read"},
        "START": {"command_group": "control"},
    },
    command_groups_requiring_explicit_allow=("control",),
    classify_command=lambda command: None
    if command is None
    else TEST_COMMAND_SPEC.command_registry[command]["command_group"],
    required_access_level=lambda command: "bind" if command == "START" else "read",
)


@dataclass(frozen=True)
class AssetGrant:
    name: str
    access: tuple[str, ...]
    command_rules: tuple[Any, ...] = ()

    def matches(self, asset_name: str) -> bool:
        return asset_name == self.name

    def audit_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "access": list(self.access),
            "command_rule_count": len(self.command_rules),
        }


@dataclass(frozen=True)
class ScopeGrant:
    scope: str
    selector: Any
    access: tuple[str, ...]
    command_rules: tuple[Any, ...] = ()
    assets: tuple[AssetGrant, ...] = ()

    def matches_principal(self, principal: Principal, principal_groups: set[str]) -> bool:
        return self.selector.matches(principal, principal_groups)

    def matching_assets(self, asset_name: str) -> tuple[AssetGrant, ...]:
        return tuple(asset for asset in self.assets if asset.matches(asset_name))

    def audit_summary(self, *, assets: tuple[AssetGrant, ...] = ()) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "access": list(self.access),
            "matched_assets": [asset.audit_summary() for asset in assets],
            "command_rule_count": len(self.command_rules),
        }


class TestPolicyAdapter(IAMPolicyAdapter):
    def parse_grants(
        self,
        loader: PolicyLoader,
        data: Any,
        *,
        context: str,
    ) -> tuple[ScopeGrant, ...]:
        if not isinstance(data, list):
            raise PolicyValidationError(f"{context} must be a list.")
        grants: list[ScopeGrant] = []
        for index, raw_grant in enumerate(data):
            grant_context = f"{context}[{index}]"
            grant_data = loader.ensure_mapping(raw_grant, context=grant_context)
            grants.append(
                ScopeGrant(
                    scope=loader.require_string(
                        grant_data.get("scope"),
                        context=f"{grant_context}.scope",
                    ),
                    selector=loader.parse_principal_selector(
                        grant_data.get("principals"),
                        context=f"{grant_context}.principals",
                    ),
                    access=loader.validate_access_levels(
                        loader.string_list(
                            grant_data.get("access"),
                            context=f"{grant_context}.access",
                        ),
                        context=f"{grant_context}.access",
                    ),
                    command_rules=loader.parse_command_rules(
                        grant_data.get("command_rules"),
                        context=f"{grant_context}.command_rules",
                    ),
                    assets=self._parse_assets(
                        loader,
                        grant_data.get("assets"),
                        context=f"{grant_context}.assets",
                    ),
                )
            )
        return tuple(grants)

    def _parse_assets(
        self,
        loader: PolicyLoader,
        data: Any,
        *,
        context: str,
    ) -> tuple[AssetGrant, ...]:
        if data is None:
            return ()
        if not isinstance(data, list):
            raise PolicyValidationError(f"{context} must be a list.")
        assets: list[AssetGrant] = []
        for index, raw_asset in enumerate(data):
            asset_context = f"{context}[{index}]"
            asset_data = loader.ensure_mapping(raw_asset, context=asset_context)
            assets.append(
                AssetGrant(
                    name=loader.require_string(
                        asset_data.get("name"),
                        context=f"{asset_context}.name",
                    ),
                    access=loader.validate_access_levels(
                        loader.string_list(
                            asset_data.get("access"),
                            context=f"{asset_context}.access",
                        ),
                        context=f"{asset_context}.access",
                    ),
                    command_rules=loader.parse_command_rules(
                        asset_data.get("command_rules"),
                        context=f"{asset_context}.command_rules",
                    ),
                )
            )
        return tuple(assets)

    def authorize(
        self,
        policy: AuthorizationPolicy,
        principal: Principal,
        principal_groups: set[str],
        *,
        action: str,
        resource: dict[str, Any],
    ) -> AuthorizationDecision:
        scope = str(resource.get("scope") or "").strip()
        asset_name = str(resource.get("asset_name") or "").strip() or None
        command = str(resource.get("command") or "").strip().upper() or None
        if not scope:
            raise ValueError("test authorization requires a scope.")

        matching_grants = [
            grant
            for grant in policy.grants
            if grant.scope == scope and grant.matches_principal(principal, principal_groups)
        ]
        if not matching_grants:
            return self._decision(
                allowed=False,
                action=action,
                reason="no matching scope grant",
                scope=scope,
                asset_name=asset_name,
                command=command,
                policy=policy,
                matched_group_names=tuple(sorted(principal_groups)),
                matched_grant_count=0,
                matching_grants=(),
                grant_used=None,
            )

        command_group = policy.command_spec.classify_command(command) if command else None
        required_level = policy.command_spec.required_access_level(command) if command else "read"
        effective_grant_count = 0
        explicit_deny = False
        best_reason = "missing required access grant"
        matching_grant_summaries: list[dict[str, Any]] = []

        for grant in matching_grants:
            asset_grants = ()
            if asset_name is not None:
                asset_grants = grant.matching_assets(asset_name)
                if grant.assets and not asset_grants:
                    continue

            effective_grant_count += 1
            grant_summary = grant.audit_summary(assets=asset_grants)
            matching_grant_summaries.append(grant_summary)
            effective_access = set(grant.access)
            for asset in asset_grants:
                effective_access.update(asset.access)

            if command is None:
                if required_level in effective_access:
                    return self._decision(
                        allowed=True,
                        action=action,
                        reason="resource_access_granted",
                        scope=scope,
                        asset_name=asset_name,
                        command=None,
                        policy=policy,
                        matched_group_names=tuple(sorted(principal_groups)),
                        matched_grant_count=effective_grant_count,
                        matching_grants=tuple(matching_grant_summaries),
                        grant_used=grant_summary,
                    )
                best_reason = f"missing '{required_level}' access"
                continue

            command_rules = list(grant.command_rules)
            for asset in asset_grants:
                command_rules.extend(asset.command_rules)

            matched_denies = [
                rule
                for rule in command_rules
                if rule.effect == "deny" and rule.matches(command, command_group)
            ]
            if matched_denies:
                explicit_deny = True
                best_reason = "command explicitly denied by IAM policy"
                continue

            matched_allows = [
                rule
                for rule in command_rules
                if rule.effect == "allow" and rule.matches(command, command_group)
            ]
            has_allowlist = any(rule.has_allowlist_entry for rule in command_rules)

            if (
                command_group in policy.command_spec.command_groups_requiring_explicit_allow
                and not matched_allows
            ):
                best_reason = "control commands require an explicit allow rule"
                continue

            if has_allowlist and not matched_allows:
                best_reason = "command is not present in the IAM allowlist"
                continue

            if required_level not in effective_access:
                best_reason = f"missing '{required_level}' access"
                continue

            return self._decision(
                allowed=True,
                action=action,
                reason="resource_access_granted",
                scope=scope,
                asset_name=asset_name,
                command=command,
                policy=policy,
                matched_group_names=tuple(sorted(principal_groups)),
                matched_grant_count=effective_grant_count,
                matching_grants=tuple(matching_grant_summaries),
                grant_used=grant_summary,
            )

        if explicit_deny:
            best_reason = "command explicitly denied by IAM policy"
        elif effective_grant_count == 0 and asset_name is not None:
            best_reason = "asset is outside the scope grant allowlist"

        return self._decision(
            allowed=False,
            action=action,
            reason=best_reason,
            scope=scope,
            asset_name=asset_name,
            command=command,
            policy=policy,
            matched_group_names=tuple(sorted(principal_groups)),
            matched_grant_count=effective_grant_count,
            matching_grants=tuple(matching_grant_summaries),
            grant_used=None,
        )

    def discover_resources(
        self,
        policy: AuthorizationPolicy,
        principal: Principal,
        principal_groups: set[str],
    ) -> tuple[dict[str, Any], ...]:
        discoveries: dict[str, dict[str, Any]] = {}
        for grant in policy.grants:
            if not grant.matches_principal(principal, principal_groups):
                continue
            discovery = discoveries.setdefault(
                grant.scope,
                {
                    "scope": grant.scope,
                    "access": set(),
                    "asset_patterns": set(),
                    "assets": {},
                    "matched_group_names": set(),
                    "matched_grant_count": 0,
                    "command_rule_count": 0,
                },
            )
            discovery["access"].update(grant.access)
            discovery["matched_group_names"].update(
                group_name for group_name in grant.selector.groups if group_name in principal_groups
            )
            discovery["matched_grant_count"] += 1
            discovery["command_rule_count"] += len(grant.command_rules)
            for asset in grant.assets:
                discovery["asset_patterns"].add(asset.name)
                asset_entry = discovery["assets"].setdefault(
                    asset.name,
                    {
                        "name": asset.name,
                        "access": set(),
                        "command_rule_count": 0,
                    },
                )
                asset_entry["access"].update(asset.access)
                asset_entry["command_rule_count"] += len(asset.command_rules)

        results: list[dict[str, Any]] = []
        for scope in sorted(discoveries):
            discovery = discoveries[scope]
            assets = []
            for asset_name in sorted(discovery["assets"]):
                asset = discovery["assets"][asset_name]
                assets.append(
                    {
                        "name": asset_name,
                        "access": sorted(asset["access"]),
                        "command_rule_count": asset["command_rule_count"],
                    }
                )
            results.append(
                {
                    "scope": scope,
                    "access": sorted(discovery["access"]),
                    "asset_patterns": sorted(discovery["asset_patterns"]),
                    "assets": assets,
                    "matched_group_names": sorted(discovery["matched_group_names"]),
                    "matched_grant_count": discovery["matched_grant_count"],
                    "command_rule_count": discovery["command_rule_count"],
                }
            )
        return tuple(results)

    def _decision(
        self,
        *,
        allowed: bool,
        action: str,
        reason: str,
        scope: str,
        asset_name: str | None,
        command: str | None,
        policy: AuthorizationPolicy,
        matched_group_names: tuple[str, ...],
        matched_grant_count: int,
        matching_grants: tuple[dict[str, Any], ...],
        grant_used: dict[str, Any] | None,
    ) -> AuthorizationDecision:
        return AuthorizationDecision(
            allowed=allowed,
            action=action,
            reason=reason,
            experiment_uuid=None,
            device_name=None,
            command=command,
            command_group=policy.command_spec.classify_command(command) if command else None,
            required_access_level=policy.command_spec.required_access_level(command)
            if command
            else "read",
            policy_path=str(policy.source_path),
            group_paths=tuple(str(path) for path in policy.group_paths),
            resource_scope={"scope": scope, "asset_name": asset_name, "command": command},
            matched_group_names=matched_group_names,
            matched_grant_count=matched_grant_count,
            grant_used=grant_used,
            matching_grants=matching_grants,
        )

    def summary(self, policy: AuthorizationPolicy) -> dict[str, Any]:
        return {
            "resource_hierarchy": ["scope", "asset", "command"],
            "resource_model": "test-scope-asset-command",
        }


TEST_POLICY_ADAPTER = TestPolicyAdapter()


class MCPAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifier_extracts_roles(self) -> None:
        secret = "test-secret-key-that-is-at-least-32-bytes"
        verifier = Auth0TokenVerifier(
            issuer_url="https://issuer.example/",
            jwks_url="https://issuer.example/.well-known/jwks.json",
            audience="https://resource.example/",
            role_claim_paths=("https://braingeneers.gi.ucsc.edu/roles", "roles"),
            algorithms=("HS256",),
            key_resolver=lambda token: secret,
        )
        token = jwt.encode(
            {
                "sub": "user-123",
                "azp": "client-123",
                "iss": "https://issuer.example/",
                "aud": "https://resource.example/",
                "exp": 4102444800,
                "https://braingeneers.gi.ucsc.edu/roles": ["mcp"],
            },
            secret,
            algorithm="HS256",
        )

        access_token = await verifier.verify_token(token)
        self.assertIsNotNone(access_token)
        self.assertEqual(getattr(access_token, "roles", []), ["mcp"])

    async def test_provider_neutral_verifier_supports_identity_only_tokens(self) -> None:
        secret = "test-secret-key-that-is-at-least-32-bytes"
        verifier = OIDCTokenVerifier(
            issuer_url="https://issuer.example/",
            jwks_url="https://issuer.example/.well-known/jwks.json",
            audience="https://resource.example/",
            role_claim_paths=(),
            algorithms=("HS256",),
            key_resolver=lambda token: secret,
        )
        token = jwt.encode(
            {
                "sub": "oauth2|CILogon|http://cilogon.org/serverA/users/139196",
                "iss": "https://issuer.example/",
                "aud": "https://resource.example/",
                "exp": 4102444800,
                "scope": "openid",
            },
            secret,
            algorithm="HS256",
        )

        access_token = await verifier.verify_token(token)
        self.assertIsNotNone(access_token)
        self.assertEqual(getattr(access_token, "roles", []), [])

    def test_runtime_auth_config_audit_fields(self) -> None:
        auth = MCPAuthRuntimeConfig(
            issuer_url="https://issuer.example/",
            jwks_url="https://issuer.example/.well-known/jwks.json",
            audience="https://resource.example/",
            resource_server_url="https://resource.example",
            role_claim_paths=("roles", "nested.groups"),
        )
        self.assertEqual(auth.audit_fields()["resource_server_url"], "https://resource.example")
        self.assertEqual(auth.audit_fields()["algorithms"], ["RS256"])

    def test_extract_claim_values_supports_nested_paths(self) -> None:
        claims = {"nested": {"groups": "sandbox-hardware mcp"}}
        self.assertEqual(
            extract_claim_values(claims, "nested.groups"),
            ["sandbox-hardware", "mcp"],
        )


class MCPIAMTests(unittest.TestCase):
    def _write_policy_bundle(self, temp_path: Path, *, groups_yaml: str, policy_yaml: str) -> Path:
        groups_path = temp_path / "groups.yaml"
        policy_path = temp_path / "test.policy.yaml"
        groups_path.write_text(groups_yaml.strip() + "\n", encoding="utf-8")
        policy_path.write_text(policy_yaml.strip() + "\n", encoding="utf-8")
        return policy_path

    def test_loader_parses_policy_for_identity_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      readers:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - readers
                        access:
                          - read
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()
            decision = policy.authorize(
                Principal(client_id="client-123", subject="user-123", roles=()),
                action="list_assets",
                resource={"scope": "sandbox"},
                fallback_required_roles=(),
            )
            self.assertTrue(decision.allowed)
            self.assertEqual(
                policy.summary()["eligibility"]["mode"],
                "identity-plus-iam",
            )

    def test_runtime_fallback_roles_are_enforced_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      readers:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - readers
                        access:
                          - read
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()
            decision = policy.authorize_service(
                Principal(client_id="client-123", subject="user-123", roles=("user",)),
                action="list_assets",
                fallback_required_roles=("mcp",),
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(
                policy.summary(fallback_required_roles=("mcp",))["eligibility"]["source"],
                "runtime_default",
            )

    def test_policy_can_discover_principal_visible_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      readers:
                        principals:
                          subjects:
                            - user-123
                      operators:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - readers
                        access:
                          - read
                        assets:
                          - name: alpha
                            access:
                              - read
                      - scope: sandbox
                        principals:
                          groups:
                            - operators
                        access:
                          - operate
                        assets:
                          - name: alpha
                            access:
                              - operate
                            command_rules:
                              - effect: allow
                                commands:
                                  - PING
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()

            discoveries = policy.discover_resources(
                Principal(client_id="client-123", subject="user-123", roles=())
            )

            self.assertEqual(len(discoveries), 1)
            self.assertEqual(discoveries[0]["scope"], "sandbox")
            self.assertEqual(discoveries[0]["access"], ["operate", "read"])
            self.assertEqual(discoveries[0]["asset_patterns"], ["alpha"])
            self.assertEqual(discoveries[0]["assets"][0]["access"], ["operate", "read"])
            self.assertEqual(discoveries[0]["matched_group_names"], ["operators", "readers"])
            self.assertEqual(discoveries[0]["matched_grant_count"], 2)
            self.assertEqual(discoveries[0]["assets"][0]["command_rule_count"], 1)

    def test_policy_declared_roles_override_runtime_identity_only_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      readers:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    eligibility:
                      required_roles_any:
                        - mcp
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - readers
                        access:
                          - read
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()
            decision = policy.authorize(
                Principal(client_id="client-123", subject="user-123", roles=("user",)),
                action="list_assets",
                resource={"scope": "sandbox"},
                fallback_required_roles=(),
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(policy.summary()["eligibility"]["source"], "policy")

    def test_loader_rejects_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      binders:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - binders
                        access:
                          - bind
                        command_rules:
                          - effect: allow
                            commands:
                              - UNKNOWN
                    """
                ),
            )
            with self.assertRaises(PolicyValidationError):
                PolicyLoader(
                    policy_path,
                    command_spec=TEST_COMMAND_SPEC,
                    grant_adapter=TEST_POLICY_ADAPTER,
                    cache_seconds=0,
                ).get_policy()

    def test_control_commands_require_explicit_allow_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      binders:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - binders
                        access:
                          - bind
                        assets:
                          - name: alpha
                            access:
                              - bind
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()
            decision = policy.authorize(
                Principal(client_id="client-123", subject="user-123", roles=()),
                action="run_command",
                resource={"scope": "sandbox", "asset_name": "alpha", "command": "START"},
                fallback_required_roles=(),
            )
            self.assertFalse(decision.allowed)
            self.assertIn("explicit allow", decision.reason)

    def test_deny_by_default_when_scope_grant_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            policy_path = self._write_policy_bundle(
                temp_path,
                groups_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-iam-groups
                    groups:
                      readers:
                        principals:
                          subjects:
                            - user-123
                    """
                ),
                policy_yaml=textwrap.dedent(
                    """
                    schema_version: 1
                    kind: braingeneers-mcp-policy
                    service: test-mcp
                    group_files:
                      - groups.yaml
                    grants:
                      - scope: sandbox
                        principals:
                          groups:
                            - readers
                        access:
                          - read
                    """
                ),
            )
            policy = PolicyLoader(
                policy_path,
                command_spec=TEST_COMMAND_SPEC,
                grant_adapter=TEST_POLICY_ADAPTER,
                cache_seconds=0,
            ).get_policy()
            decision = policy.authorize(
                Principal(client_id="client-123", subject="user-123", roles=()),
                action="list_assets",
                resource={"scope": "other-scope"},
                fallback_required_roles=(),
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "no matching scope grant")


if __name__ == "__main__":
    unittest.main()
