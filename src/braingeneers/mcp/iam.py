from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

from .models import Principal


ACCESS_LEVELS = {"read", "operate", "bind"}
GROUPS_FILE_KIND = "braingeneers-iam-groups"
POLICY_FILE_KIND = "braingeneers-mcp-policy"
POLICY_SCHEMA_VERSION = 1


class PolicyValidationError(ValueError):
    pass


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyValidationError(f"{context} must be a mapping.")
    return value


def _optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PolicyValidationError(f"{context} must be a non-empty string when provided.")
    return value.strip()


def _require_string(value: Any, *, context: str) -> str:
    result = _optional_string(value, context=context)
    if result is None:
        raise PolicyValidationError(f"{context} is required.")
    return result


def _string_list(value: Any, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PolicyValidationError(f"{context} must be a list.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_require_string(item, context=f"{context}[{index}]"))
    return tuple(normalized)


def _validate_access_levels(levels: tuple[str, ...], *, context: str) -> tuple[str, ...]:
    invalid = sorted(set(levels) - ACCESS_LEVELS)
    if invalid:
        raise PolicyValidationError(
            f"{context} contains unsupported access levels: {', '.join(invalid)}."
        )
    return tuple(sorted(set(levels)))


@dataclass(frozen=True)
class IAMCommandSpec:
    command_registry: dict[str, dict[str, Any]]
    command_groups_requiring_explicit_allow: tuple[str, ...]
    classify_command: Any
    required_access_level: Any

    @property
    def known_command_groups(self) -> set[str]:
        return {spec["command_group"] for spec in self.command_registry.values()}


@dataclass(frozen=True)
class PrincipalSelector:
    subjects: tuple[str, ...] = ()
    client_ids: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    def matches(self, principal: Principal, principal_groups: set[str]) -> bool:
        return any(
            (
                principal.subject in self.subjects,
                principal.client_id in self.client_ids,
                any(group in principal_groups for group in self.groups),
            )
        )

    @property
    def is_empty(self) -> bool:
        return not (self.subjects or self.client_ids or self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjects": list(self.subjects),
            "client_ids": list(self.client_ids),
            "groups": list(self.groups),
        }


@dataclass(frozen=True)
class GroupDefinition:
    name: str
    description: str | None = None
    subjects: tuple[str, ...] = ()
    client_ids: tuple[str, ...] = ()

    def matches(self, principal: Principal) -> bool:
        return principal.subject in self.subjects or principal.client_id in self.client_ids


@dataclass(frozen=True)
class CommandRule:
    effect: str
    commands: tuple[str, ...] = ()
    command_groups: tuple[str, ...] = ()
    description: str | None = None

    def matches(self, command: str, command_group: str | None) -> bool:
        if command_group is None:
            return command in self.commands
        return command in self.commands or command_group in self.command_groups

    @property
    def has_allowlist_entry(self) -> bool:
        return self.effect == "allow"


class IAMPolicyAdapter:
    def parse_grants(
        self,
        loader: PolicyLoader,
        data: Any,
        *,
        context: str,
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def authorize(
        self,
        policy: AuthorizationPolicy,
        principal: Principal,
        principal_groups: set[str],
        *,
        action: str,
        resource: dict[str, Any],
    ) -> AuthorizationDecision:
        raise NotImplementedError

    def summary(self, policy: AuthorizationPolicy) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    action: str
    reason: str
    experiment_uuid: str | None
    device_name: str | None
    command: str | None
    command_group: str | None
    required_access_level: str | None
    policy_path: str
    group_paths: tuple[str, ...]
    resource_scope: dict[str, Any] = field(default_factory=dict)
    matched_group_names: tuple[str, ...] = ()
    matched_grant_count: int = 0
    grant_used: dict[str, Any] | None = None
    matching_grants: tuple[dict[str, Any], ...] = ()
    policy_layer: str = "iam"

    def error_message(self) -> str:
        if self.allowed:
            return ""
        if self.policy_layer == "service_eligibility":
            return self.reason
        if self.command and self.device_name and self.experiment_uuid:
            return (
                f"Not authorized to run {self.command} on {self.device_name} "
                f"for UUID {self.experiment_uuid}: {self.reason}"
            )
        if self.device_name and self.experiment_uuid:
            return (
                f"Not authorized to read {self.device_name} for UUID "
                f"{self.experiment_uuid}: {self.reason}"
            )
        if self.experiment_uuid:
            return f"Not authorized for UUID {self.experiment_uuid}: {self.reason}"
        if self.resource_scope:
            scope = ", ".join(
                f"{key}={value}"
                for key, value in self.resource_scope.items()
                if value is not None and value != ""
            )
            if scope:
                return f"Not authorized for {scope}: {self.reason}"
        return self.reason

    def audit_details(self) -> dict[str, Any]:
        return {
            "policy_layer": self.policy_layer,
            "policy_path": self.policy_path,
            "group_paths": list(self.group_paths),
            "experiment_uuid": self.experiment_uuid,
            "device_name": self.device_name,
            "command": self.command,
            "command_group": self.command_group,
            "required_access_level": self.required_access_level,
            "resource_scope": self.resource_scope,
            "reason": self.reason,
            "matched_group_names": list(self.matched_group_names),
            "matched_grant_count": self.matched_grant_count,
            "grant_used": self.grant_used,
            "matching_grants": list(self.matching_grants),
        }


@dataclass(frozen=True)
class AuthorizationPolicy:
    service: str
    description: str | None
    source_path: Path
    group_paths: tuple[Path, ...]
    eligibility_required_roles_any: tuple[str, ...]
    groups: dict[str, GroupDefinition]
    grants: tuple[Any, ...]
    command_spec: IAMCommandSpec
    grant_adapter: IAMPolicyAdapter

    def resolve_principal_groups(self, principal: Principal) -> set[str]:
        resolved = set()
        for group_name, group in self.groups.items():
            if group.matches(principal):
                resolved.add(group_name)
        return resolved

    def effective_required_roles(
        self,
        *,
        fallback_required_roles: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self.eligibility_required_roles_any:
            return self.eligibility_required_roles_any
        return tuple(sorted(set(fallback_required_roles)))

    def eligibility_summary(
        self,
        *,
        fallback_required_roles: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        fallback_roles = tuple(sorted(set(fallback_required_roles)))
        effective_roles = self.effective_required_roles(
            fallback_required_roles=fallback_required_roles
        )
        if self.eligibility_required_roles_any:
            source = "policy"
        elif fallback_roles:
            source = "runtime_default"
        else:
            source = "identity_only"
        return {
            "mode": "role-plus-iam" if effective_roles else "identity-plus-iam",
            "source": source,
            "policy_required_roles_any": list(self.eligibility_required_roles_any),
            "fallback_required_roles": list(fallback_roles),
            "effective_required_roles_any": list(effective_roles),
        }

    def authorize_service(
        self,
        principal: Principal,
        *,
        fallback_required_roles: tuple[str, ...],
        action: str,
    ) -> AuthorizationDecision:
        required_roles = self.effective_required_roles(
            fallback_required_roles=fallback_required_roles
        )
        if required_roles and not set(principal.roles).intersection(required_roles):
            return AuthorizationDecision(
                allowed=False,
                action=action,
                reason=(
                    "MCP access requires one of these roles: "
                    + ", ".join(required_roles)
                    + "."
                ),
                experiment_uuid=None,
                device_name=None,
                command=None,
                command_group=None,
                required_access_level=None,
                policy_path=str(self.source_path),
                group_paths=tuple(str(path) for path in self.group_paths),
                policy_layer="service_eligibility",
            )

        return AuthorizationDecision(
            allowed=True,
            action=action,
            reason="service_eligible",
            experiment_uuid=None,
            device_name=None,
            command=None,
            command_group=None,
            required_access_level=None,
            policy_path=str(self.source_path),
            group_paths=tuple(str(path) for path in self.group_paths),
            policy_layer="service_eligibility",
        )

    def authorize(
        self,
        principal: Principal,
        *,
        action: str,
        resource: dict[str, Any],
        fallback_required_roles: tuple[str, ...],
    ) -> AuthorizationDecision:
        service_decision = self.authorize_service(
            principal,
            fallback_required_roles=fallback_required_roles,
            action=action,
        )
        if not service_decision.allowed:
            return service_decision

        principal_groups = self.resolve_principal_groups(principal)
        return self.grant_adapter.authorize(
            self,
            principal,
            principal_groups,
            action=action,
            resource=resource,
        )

    def authorize_resource(
        self,
        principal: Principal,
        *,
        experiment_uuid: str,
        action: str,
        device_name: str | None = None,
        command: str | None = None,
        fallback_required_roles: tuple[str, ...],
    ) -> AuthorizationDecision:
        return self.authorize(
            principal,
            action=action,
            resource={
                "experiment_uuid": experiment_uuid,
                "device_name": device_name,
                "command": command,
            },
            fallback_required_roles=fallback_required_roles,
        )

    def summary(
        self,
        *,
        fallback_required_roles: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return {
            "service": self.service,
            "description": self.description,
            "policy_path": str(self.source_path),
            "group_paths": [str(path) for path in self.group_paths],
            "eligibility_required_roles_any": list(self.eligibility_required_roles_any),
            "eligibility": self.eligibility_summary(
                fallback_required_roles=fallback_required_roles
            ),
            "group_names": sorted(self.groups),
            "grant_count": len(self.grants),
            **self.grant_adapter.summary(self),
        }


class PolicyLoader:
    def __init__(
        self,
        policy_path: str | Path,
        *,
        command_spec: IAMCommandSpec,
        grant_adapter: IAMPolicyAdapter,
        cache_seconds: int = 5,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.command_spec = command_spec
        self.grant_adapter = grant_adapter
        self.cache_seconds = cache_seconds
        self._cached_policy: AuthorizationPolicy | None = None
        self._cached_at = 0.0

    def get_policy(self) -> AuthorizationPolicy:
        if self._cached_policy is not None and monotonic() - self._cached_at < self.cache_seconds:
            return self._cached_policy
        policy = self._load_policy()
        self._cached_policy = policy
        self._cached_at = monotonic()
        return policy

    def ensure_mapping(self, value: Any, *, context: str) -> dict[str, Any]:
        return _ensure_mapping(value, context=context)

    def optional_string(self, value: Any, *, context: str) -> str | None:
        return _optional_string(value, context=context)

    def require_string(self, value: Any, *, context: str) -> str:
        return _require_string(value, context=context)

    def string_list(self, value: Any, *, context: str) -> tuple[str, ...]:
        return _string_list(value, context=context)

    def validate_access_levels(
        self,
        levels: tuple[str, ...],
        *,
        context: str,
    ) -> tuple[str, ...]:
        return _validate_access_levels(levels, context=context)

    def parse_principal_selector(self, data: Any, *, context: str) -> PrincipalSelector:
        selector_data = _ensure_mapping(data, context=context)
        selector = PrincipalSelector(
            subjects=_string_list(selector_data.get("subjects"), context=f"{context}.subjects"),
            client_ids=_string_list(selector_data.get("client_ids"), context=f"{context}.client_ids"),
            groups=_string_list(selector_data.get("groups"), context=f"{context}.groups"),
        )
        if selector.is_empty:
            raise PolicyValidationError(
                f"{context} must include at least one subject, client_id, or group."
            )
        return selector

    def parse_command_rules(
        self,
        rules_data: Any,
        *,
        context: str,
    ) -> tuple[CommandRule, ...]:
        if rules_data is None:
            return ()
        if not isinstance(rules_data, list):
            raise PolicyValidationError(f"{context} must be a list.")
        rules: list[CommandRule] = []
        for index, raw_rule in enumerate(rules_data):
            rule_context = f"{context}[{index}]"
            rule_data = _ensure_mapping(raw_rule, context=rule_context)
            effect = _require_string(
                rule_data.get("effect"),
                context=f"{rule_context}.effect",
            ).lower()
            if effect not in {"allow", "deny"}:
                raise PolicyValidationError(f"{rule_context}.effect must be 'allow' or 'deny'.")

            commands = tuple(
                command.upper()
                for command in _string_list(
                    rule_data.get("commands"),
                    context=f"{rule_context}.commands",
                )
            )
            unknown_commands = sorted(
                command for command in commands if command not in self.command_spec.command_registry
            )
            if unknown_commands:
                raise PolicyValidationError(
                    f"{rule_context}.commands contains unsupported commands: "
                    f"{', '.join(unknown_commands)}."
                )

            command_groups = _string_list(
                rule_data.get("command_groups"),
                context=f"{rule_context}.command_groups",
            )
            unknown_command_groups = sorted(
                set(command_groups) - self.command_spec.known_command_groups
            )
            if unknown_command_groups:
                raise PolicyValidationError(
                    f"{rule_context}.command_groups contains unsupported groups: "
                    + ", ".join(unknown_command_groups)
                    + "."
                )

            if not commands and not command_groups:
                raise PolicyValidationError(
                    f"{rule_context} must match at least one command or command group."
                )

            rules.append(
                CommandRule(
                    effect=effect,
                    commands=commands,
                    command_groups=command_groups,
                    description=_optional_string(
                        rule_data.get("description"),
                        context=f"{rule_context}.description",
                    ),
                )
            )
        return tuple(rules)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            raise PolicyValidationError(f"Policy file '{path}' was not found.") from exc
        except yaml.YAMLError as exc:
            raise PolicyValidationError(f"Failed to parse YAML from '{path}': {exc}") from exc
        return _ensure_mapping(data, context=str(path))

    def _parse_groups(
        self,
        policy_dir: Path,
        group_files: tuple[str, ...],
    ) -> tuple[dict[str, GroupDefinition], tuple[Path, ...]]:
        parsed_groups: dict[str, GroupDefinition] = {}
        resolved_paths: list[Path] = []
        for group_file in group_files:
            group_path = (policy_dir / group_file).resolve()
            group_document = self._load_yaml(group_path)
            schema_version = group_document.get("schema_version", POLICY_SCHEMA_VERSION)
            if schema_version != POLICY_SCHEMA_VERSION:
                raise PolicyValidationError(
                    f"{group_path} uses unsupported schema_version '{schema_version}'."
                )
            kind = group_document.get("kind")
            if kind != GROUPS_FILE_KIND:
                raise PolicyValidationError(
                    f"{group_path} must declare kind '{GROUPS_FILE_KIND}'."
                )

            groups_section = _ensure_mapping(
                group_document.get("groups", {}),
                context=f"{group_path}.groups",
            )
            for group_name, raw_group in groups_section.items():
                if group_name in parsed_groups:
                    raise PolicyValidationError(f"Duplicate group definition '{group_name}'.")
                group_data = _ensure_mapping(
                    raw_group,
                    context=f"{group_path}.groups.{group_name}",
                )
                principal_data = _ensure_mapping(
                    group_data.get("principals", {}),
                    context=f"{group_path}.groups.{group_name}.principals",
                )
                group = GroupDefinition(
                    name=group_name,
                    description=_optional_string(
                        group_data.get("description"),
                        context=f"{group_path}.groups.{group_name}.description",
                    ),
                    subjects=_string_list(
                        principal_data.get("subjects"),
                        context=f"{group_path}.groups.{group_name}.principals.subjects",
                    ),
                    client_ids=_string_list(
                        principal_data.get("client_ids"),
                        context=f"{group_path}.groups.{group_name}.principals.client_ids",
                    ),
                )
                if not group.subjects and not group.client_ids:
                    raise PolicyValidationError(
                        f"{group_path}.groups.{group_name} must define at least one "
                        "subject or client_id."
                    )
                parsed_groups[group_name] = group
            resolved_paths.append(group_path)
        return parsed_groups, tuple(resolved_paths)

    def _load_policy(self) -> AuthorizationPolicy:
        policy_path = self.policy_path.resolve()
        document = self._load_yaml(policy_path)

        schema_version = document.get("schema_version", POLICY_SCHEMA_VERSION)
        if schema_version != POLICY_SCHEMA_VERSION:
            raise PolicyValidationError(
                f"{policy_path} uses unsupported schema_version '{schema_version}'."
            )

        kind = document.get("kind")
        if kind != POLICY_FILE_KIND:
            raise PolicyValidationError(f"{policy_path} must declare kind '{POLICY_FILE_KIND}'.")

        policy_dir = policy_path.parent
        groups, group_paths = self._parse_groups(
            policy_dir,
            _string_list(document.get("group_files"), context=f"{policy_path}.group_files"),
        )
        eligibility = _ensure_mapping(
            document.get("eligibility", {}),
            context=f"{policy_path}.eligibility",
        )
        grants = self.grant_adapter.parse_grants(
            self,
            document.get("grants", []),
            context=f"{policy_path}.grants",
        )

        return AuthorizationPolicy(
            service=_require_string(document.get("service"), context=f"{policy_path}.service"),
            description=_optional_string(
                document.get("description"),
                context=f"{policy_path}.description",
            ),
            source_path=policy_path,
            group_paths=group_paths,
            eligibility_required_roles_any=_string_list(
                eligibility.get("required_roles_any"),
                context=f"{policy_path}.eligibility.required_roles_any",
            ),
            groups=groups,
            grants=grants,
            command_spec=self.command_spec,
            grant_adapter=self.grant_adapter,
        )
