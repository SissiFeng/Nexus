"""Tests for API key authentication and role-based access control."""

from pathlib import Path

import pytest

from optimization_copilot.platform.auth import AuthError, AuthManager
from optimization_copilot.platform.models import ApiKey, Role, ROLE_HIERARCHY
from optimization_copilot.platform.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "auth_ws")
    workspace.init()
    return workspace


@pytest.fixture
def auth(ws: Workspace) -> AuthManager:
    return AuthManager(ws)


# ── Key Generation ─────────────────────────────────────────────────


class TestKeyGeneration:
    def test_create_key_returns_prefixed_string(self, auth: AuthManager):
        raw_key = auth.create_key("my-key", Role.OPERATOR)
        assert raw_key.startswith("ocp_")

    def test_create_key_persists_to_workspace(self, auth: AuthManager):
        auth.create_key("persisted-key", Role.ADMIN)
        keys = auth.list_keys()
        assert len(keys) == 1
        assert keys[0].name == "persisted-key"
        assert keys[0].role is Role.ADMIN

    def test_generate_key_produces_unique_keys(self, auth: AuthManager):
        keys = {auth._generate_key() for _ in range(50)}
        assert len(keys) == 50

    def test_multiple_keys_can_be_created(self, auth: AuthManager):
        auth.create_key("key-1", Role.VIEWER)
        auth.create_key("key-2", Role.OPERATOR)
        auth.create_key("key-3", Role.ADMIN)
        assert len(auth.list_keys()) == 3


# ── Authentication ─────────────────────────────────────────────────


class TestAuthentication:
    def test_authenticate_valid_key(self, auth: AuthManager):
        raw_key = auth.create_key("valid", Role.OPERATOR)
        result = auth.authenticate(raw_key)
        assert result is not None
        assert isinstance(result, ApiKey)
        assert result.name == "valid"

    def test_authenticate_invalid_key_returns_none(self, auth: AuthManager):
        auth.create_key("real", Role.VIEWER)
        result = auth.authenticate("ocp_bogus_key_that_does_not_exist")
        assert result is None

    def test_authenticate_revoked_key_returns_none(self, auth: AuthManager):
        raw_key = auth.create_key("revoked", Role.OPERATOR)
        key_hash = auth._hash_key(raw_key)
        auth.revoke_key(key_hash)
        result = auth.authenticate(raw_key)
        assert result is None

    def test_authenticate_updates_last_used(self, auth: AuthManager):
        raw_key = auth.create_key("tracked", Role.VIEWER)
        # Before authentication, last_used should be None
        keys_before = auth.list_keys()
        assert keys_before[0].last_used is None

        auth.authenticate(raw_key)
        keys_after = auth.list_keys()
        assert keys_after[0].last_used is not None


# ── Revocation ─────────────────────────────────────────────────────


class TestRevocation:
    def test_revoke_key_marks_inactive(self, auth: AuthManager):
        raw_key = auth.create_key("to-revoke", Role.ADMIN)
        key_hash = auth._hash_key(raw_key)
        auth.revoke_key(key_hash)
        keys = auth.list_keys()
        assert keys[0].active is False

    def test_revoke_unknown_key_raises(self, auth: AuthManager):
        with pytest.raises(AuthError, match="Key not found"):
            auth.revoke_key("nonexistent_hash")


# ── Authorization ──────────────────────────────────────────────────


class TestAuthorization:
    def _make_key(self, role: Role) -> ApiKey:
        return ApiKey(key_hash="h", name="k", role=role, created_at=1.0)

    def test_admin_can_access_admin(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.ADMIN), Role.ADMIN) is True

    def test_admin_can_access_operator(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.ADMIN), Role.OPERATOR) is True

    def test_admin_can_access_viewer(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.ADMIN), Role.VIEWER) is True

    def test_operator_can_access_operator(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.OPERATOR), Role.OPERATOR) is True

    def test_operator_can_access_viewer(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.OPERATOR), Role.VIEWER) is True

    def test_operator_cannot_access_admin(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.OPERATOR), Role.ADMIN) is False

    def test_viewer_can_access_viewer(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.VIEWER), Role.VIEWER) is True

    def test_viewer_cannot_access_operator(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.VIEWER), Role.OPERATOR) is False

    def test_viewer_cannot_access_admin(self, auth: AuthManager):
        assert auth.authorize(self._make_key(Role.VIEWER), Role.ADMIN) is False


# ── Hash Determinism ───────────────────────────────────────────────


class TestHashing:
    def test_hash_key_is_deterministic(self):
        h1 = AuthManager._hash_key("ocp_testkey123")
        h2 = AuthManager._hash_key("ocp_testkey123")
        assert h1 == h2

    def test_different_keys_produce_different_hashes(self):
        h1 = AuthManager._hash_key("ocp_key_a")
        h2 = AuthManager._hash_key("ocp_key_b")
        assert h1 != h2


# ── Persistence Across Instances ───────────────────────────────────


class TestPersistence:
    def test_keys_persist_across_auth_manager_instances(self, ws: Workspace):
        auth1 = AuthManager(ws)
        raw_key = auth1.create_key("persist-test", Role.ADMIN)

        auth2 = AuthManager(ws)
        result = auth2.authenticate(raw_key)
        assert result is not None
        assert result.name == "persist-test"

    def test_list_keys_from_fresh_instance(self, ws: Workspace):
        auth1 = AuthManager(ws)
        auth1.create_key("k1", Role.VIEWER)
        auth1.create_key("k2", Role.OPERATOR)

        auth2 = AuthManager(ws)
        assert len(auth2.list_keys()) == 2
