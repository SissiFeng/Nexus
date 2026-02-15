"""API key authentication and role-based access control."""

from __future__ import annotations

import hashlib
import secrets
from time import time

from optimization_copilot.platform.models import ApiKey, Role, ROLE_HIERARCHY
from optimization_copilot.platform.workspace import Workspace


class AuthError(Exception):
    """Authentication or authorization error."""


class AuthManager:
    """API key management with role hierarchy."""

    KEY_PREFIX = "ocp_"

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    # ── Key Management ────────────────────────────────────────

    def create_key(self, name: str, role: Role) -> str:
        """Create a new API key. Returns the raw key (only time it's visible)."""
        raw_key = self._generate_key()
        key_hash = self._hash_key(raw_key)

        api_key = ApiKey(
            key_hash=key_hash,
            name=name,
            role=role,
            created_at=time(),
        )

        keys = self._workspace.load_keys()
        keys.append(api_key)
        self._workspace.save_keys(keys)

        return raw_key

    def revoke_key(self, key_hash: str) -> None:
        """Revoke an API key by its hash."""
        keys = self._workspace.load_keys()
        for key in keys:
            if key.key_hash == key_hash:
                key.active = False
                break
        else:
            raise AuthError(f"Key not found: {key_hash}")
        self._workspace.save_keys(keys)

    def list_keys(self) -> list[ApiKey]:
        """List all API keys (active and revoked)."""
        return self._workspace.load_keys()

    # ── Authentication ────────────────────────────────────────

    def authenticate(self, raw_key: str) -> ApiKey | None:
        """Authenticate a raw API key. Returns ApiKey if valid, None otherwise."""
        key_hash = self._hash_key(raw_key)
        keys = self._workspace.load_keys()

        for key in keys:
            if key.key_hash == key_hash and key.active:
                key.last_used = time()
                self._workspace.save_keys(keys)
                return key

        return None

    # ── Authorization ─────────────────────────────────────────

    @staticmethod
    def authorize(api_key: ApiKey, required_role: Role) -> bool:
        """Check if an API key has sufficient privileges."""
        return ROLE_HIERARCHY[api_key.role] >= ROLE_HIERARCHY[required_role]

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """SHA-256 hash of a raw API key."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def _generate_key(cls) -> str:
        """Generate a new raw API key."""
        return f"{cls.KEY_PREFIX}{secrets.token_urlsafe(32)}"
