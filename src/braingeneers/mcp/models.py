from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Principal:
    client_id: str
    subject: str
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    claims: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def requester_name(self) -> str:
        return self.subject or self.client_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "subject": self.subject,
            "roles": list(self.roles),
            "scopes": list(self.scopes),
        }
