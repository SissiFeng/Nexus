"""JSON-file workspace persistence with atomic writes.

Directory layout:
    workspace_dir/
    ├── manifest.json
    ├── auth/keys.json
    ├── campaigns/{id}/record.json
    ├── campaigns/{id}/spec.json
    ├── campaigns/{id}/checkpoint.json
    ├── campaigns/{id}/result.json
    ├── campaigns/{id}/store.json
    ├── campaigns/{id}/audit.json
    ├── meta_learning/advisor.json
    └── rag/index.json
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from time import time
from typing import Any

from optimization_copilot.platform.models import (
    ApiKey,
    CampaignRecord,
    WorkspaceManifest,
)


class WorkspaceError(Exception):
    """Workspace operation error."""


class CampaignNotFoundError(WorkspaceError):
    """Campaign does not exist."""


class Workspace:
    """JSON-file workspace with atomic writes and per-campaign locking."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).resolve()
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    # ── Manifest ──────────────────────────────────────────────

    def init(self) -> WorkspaceManifest:
        """Create workspace directory structure and manifest."""
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "auth").mkdir(exist_ok=True)
        (self._root / "campaigns").mkdir(exist_ok=True)
        (self._root / "meta_learning").mkdir(exist_ok=True)
        (self._root / "rag").mkdir(exist_ok=True)

        manifest_path = self._root / "manifest.json"
        if manifest_path.exists():
            return self.manifest()

        manifest = WorkspaceManifest(
            workspace_id=str(uuid.uuid4()),
            created_at=time(),
        )
        self._atomic_write(manifest_path, manifest.to_dict())
        return manifest

    def manifest(self) -> WorkspaceManifest:
        """Read workspace manifest."""
        data = self._read_json(self._root / "manifest.json")
        if data is None:
            raise WorkspaceError("Workspace not initialized. Call init() first.")
        return WorkspaceManifest.from_dict(data)

    def _save_manifest(self, manifest: WorkspaceManifest) -> None:
        self._atomic_write(self._root / "manifest.json", manifest.to_dict())

    # ── Campaign CRUD ─────────────────────────────────────────

    def save_campaign(self, record: CampaignRecord) -> None:
        """Save campaign record (creates campaign dir if needed)."""
        campaign_dir = self._campaign_dir(record.campaign_id)
        campaign_dir.mkdir(parents=True, exist_ok=True)

        lock = self._get_lock(record.campaign_id)
        with lock:
            self._atomic_write(campaign_dir / "record.json", record.to_dict())

        # Update manifest
        with self._global_lock:
            m = self.manifest()
            m.campaigns[record.campaign_id] = record.name
            self._save_manifest(m)

    def load_campaign(self, campaign_id: str) -> CampaignRecord:
        """Load campaign record by ID."""
        path = self._campaign_dir(campaign_id) / "record.json"
        data = self._read_json(path)
        if data is None:
            raise CampaignNotFoundError(f"Campaign not found: {campaign_id}")
        return CampaignRecord.from_dict(data)

    def list_campaigns(self) -> list[CampaignRecord]:
        """List all campaign records."""
        campaigns_dir = self._root / "campaigns"
        if not campaigns_dir.exists():
            return []

        records = []
        for entry in sorted(campaigns_dir.iterdir()):
            if entry.is_dir():
                record_path = entry / "record.json"
                data = self._read_json(record_path)
                if data is not None:
                    records.append(CampaignRecord.from_dict(data))
        return records

    def delete_campaign(self, campaign_id: str) -> None:
        """Delete campaign directory entirely."""
        campaign_dir = self._campaign_dir(campaign_id)
        if campaign_dir.exists():
            shutil.rmtree(campaign_dir)

        with self._global_lock:
            m = self.manifest()
            m.campaigns.pop(campaign_id, None)
            self._save_manifest(m)

    def campaign_exists(self, campaign_id: str) -> bool:
        """Check if campaign exists."""
        return (self._campaign_dir(campaign_id) / "record.json").exists()

    # ── Campaign Artifacts ────────────────────────────────────

    def save_spec(self, campaign_id: str, spec_dict: dict[str, Any]) -> None:
        lock = self._get_lock(campaign_id)
        with lock:
            self._atomic_write(
                self._campaign_dir(campaign_id) / "spec.json", spec_dict
            )

    def load_spec(self, campaign_id: str) -> dict[str, Any]:
        data = self._read_json(self._campaign_dir(campaign_id) / "spec.json")
        if data is None:
            raise CampaignNotFoundError(f"Spec not found for campaign: {campaign_id}")
        return data

    def save_checkpoint(self, campaign_id: str, state_dict: dict[str, Any]) -> None:
        lock = self._get_lock(campaign_id)
        with lock:
            self._atomic_write(
                self._campaign_dir(campaign_id) / "checkpoint.json", state_dict
            )

    def load_checkpoint(self, campaign_id: str) -> dict[str, Any] | None:
        return self._read_json(self._campaign_dir(campaign_id) / "checkpoint.json")

    def save_result(self, campaign_id: str, result_dict: dict[str, Any]) -> None:
        lock = self._get_lock(campaign_id)
        with lock:
            self._atomic_write(
                self._campaign_dir(campaign_id) / "result.json", result_dict
            )

    def load_result(self, campaign_id: str) -> dict[str, Any] | None:
        return self._read_json(self._campaign_dir(campaign_id) / "result.json")

    def save_store(self, campaign_id: str, store_dict: dict[str, Any]) -> None:
        lock = self._get_lock(campaign_id)
        with lock:
            self._atomic_write(
                self._campaign_dir(campaign_id) / "store.json", store_dict
            )

    def load_store(self, campaign_id: str) -> dict[str, Any] | None:
        return self._read_json(self._campaign_dir(campaign_id) / "store.json")

    def save_audit(self, campaign_id: str, audit_dict: dict[str, Any]) -> None:
        lock = self._get_lock(campaign_id)
        with lock:
            self._atomic_write(
                self._campaign_dir(campaign_id) / "audit.json", audit_dict
            )

    def load_audit(self, campaign_id: str) -> dict[str, Any] | None:
        return self._read_json(self._campaign_dir(campaign_id) / "audit.json")

    # ── Meta-learning ─────────────────────────────────────────

    def save_advisor(self, advisor_dict: dict[str, Any]) -> None:
        with self._global_lock:
            self._atomic_write(
                self._root / "meta_learning" / "advisor.json", advisor_dict
            )

    def load_advisor(self) -> dict[str, Any] | None:
        return self._read_json(self._root / "meta_learning" / "advisor.json")

    # ── Auth ──────────────────────────────────────────────────

    def save_keys(self, keys: list[ApiKey]) -> None:
        with self._global_lock:
            self._atomic_write(
                self._root / "auth" / "keys.json",
                {"keys": [k.to_dict() for k in keys]},
            )

    def load_keys(self) -> list[ApiKey]:
        data = self._read_json(self._root / "auth" / "keys.json")
        if data is None:
            return []
        return [ApiKey.from_dict(k) for k in data.get("keys", [])]

    # ── RAG Index ─────────────────────────────────────────────

    def save_rag_index(self, index_data: dict[str, Any]) -> None:
        with self._global_lock:
            self._atomic_write(self._root / "rag" / "index.json", index_data)

    def load_rag_index(self) -> dict[str, Any] | None:
        return self._read_json(self._root / "rag" / "index.json")

    # ── Internal helpers ──────────────────────────────────────

    def _campaign_dir(self, campaign_id: str) -> Path:
        return self._root / "campaigns" / campaign_id

    def _get_lock(self, campaign_id: str) -> threading.Lock:
        with self._global_lock:
            if campaign_id not in self._locks:
                self._locks[campaign_id] = threading.Lock()
            return self._locks[campaign_id]

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON atomically: write to .tmp then os.replace()."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read JSON file, returning None if not found."""
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
