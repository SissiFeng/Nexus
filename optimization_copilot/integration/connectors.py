"""Abstract connector interface and concrete adapters for lab systems.

Provides a uniform interface for reading observations from and writing
suggestions to various data backends (CSV files, JSON files, in-memory
stores). Custom connectors can be created by subclassing ``LabConnector``.
"""

from __future__ import annotations

import csv
import io
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import Observation


# ── Dataclasses ────────────────────────────────────────


@dataclass
class ConnectorStatus:
    """Status information for a lab connector.

    Parameters
    ----------
    connected:
        Whether the connector is currently connected / available.
    last_sync:
        Unix timestamp of the last successful read or write, or ``None``
        if no sync has occurred.
    record_count:
        Number of records currently available through the connector.
    """

    connected: bool
    last_sync: float | None
    record_count: int


# ── Abstract Base ──────────────────────────────────────


class LabConnector(ABC):
    """Abstract base class for lab system connectors.

    All connectors must implement three operations:
    - Reading observations from the data source
    - Writing parameter suggestions back to the data sink
    - Reporting current connector status
    """

    @abstractmethod
    def read_observations(self) -> list[Observation]:
        """Read observations from the connected data source.

        Returns
        -------
        list[Observation]
            All available observations.
        """
        ...

    @abstractmethod
    def write_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        """Write parameter suggestions to the connected data sink.

        Parameters
        ----------
        suggestions:
            A list of dictionaries, each mapping parameter names to
            suggested values.
        """
        ...

    @abstractmethod
    def status(self) -> ConnectorStatus:
        """Return the current status of this connector.

        Returns
        -------
        ConnectorStatus
            Connection state, last sync time, and record count.
        """
        ...


# ── CSV Connector ──────────────────────────────────────


class CSVConnector(LabConnector):
    """Connector that reads observations from and writes suggestions to CSV files.

    Parameters
    ----------
    read_path:
        Path to the CSV file containing observation data. Parameter and
        KPI columns are auto-detected: columns whose values are all
        numeric are treated as either parameters or KPIs based on naming
        conventions.
    write_path:
        Path to write suggestion CSV rows to.
    """

    def __init__(self, read_path: str, write_path: str) -> None:
        self._read_path = read_path
        self._write_path = write_path
        self._last_sync: float | None = None
        self._record_count: int = 0

    def read_observations(self) -> list[Observation]:
        """Read observations from the CSV file.

        Auto-detects column roles: an ``iteration`` column is used for
        iteration indices (falling back to row index), and remaining
        numeric columns are split into parameters and KPI values based
        on whether a column name starts with ``kpi_`` or ``objective_``.
        All other numeric columns are treated as parameters.

        Returns
        -------
        list[Observation]
            Parsed observations from the CSV file.
        """
        try:
            with open(self._read_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []

                observations: list[Observation] = []
                kpi_prefixes = ("kpi_", "objective_", "obj_")

                for row_idx, row in enumerate(reader):
                    # Determine iteration.
                    if "iteration" in row:
                        try:
                            iteration = int(row["iteration"])
                        except (ValueError, TypeError):
                            iteration = row_idx
                    else:
                        iteration = row_idx

                    parameters: dict[str, Any] = {}
                    kpi_values: dict[str, float] = {}
                    metadata: dict[str, Any] = {}

                    for col, raw in row.items():
                        if col == "iteration":
                            continue

                        # Try numeric conversion.
                        try:
                            val = float(raw)
                            if col.lower().startswith(kpi_prefixes):
                                kpi_values[col] = val
                            else:
                                parameters[col] = val
                        except (ValueError, TypeError):
                            metadata[col] = raw

                    observations.append(
                        Observation(
                            iteration=iteration,
                            parameters=parameters,
                            kpi_values=kpi_values,
                            metadata=metadata,
                        )
                    )

                self._record_count = len(observations)
                self._last_sync = time.time()
                return observations

        except FileNotFoundError:
            return []

    def write_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        """Write suggestions as CSV rows.

        If the file does not exist, a header row is written first.
        Subsequent calls append to the file.

        Parameters
        ----------
        suggestions:
            Parameter suggestion dictionaries to write.
        """
        if not suggestions:
            return

        # Collect all keys across suggestions for the header.
        all_keys: list[str] = []
        seen: set[str] = set()
        for s in suggestions:
            for k in s:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # Check if file exists and has content.
        write_header = True
        try:
            with open(self._write_path, "r") as f:
                if f.readline().strip():
                    write_header = False
        except FileNotFoundError:
            pass

        with open(self._write_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for s in suggestions:
                writer.writerow(s)

        self._last_sync = time.time()

    def status(self) -> ConnectorStatus:
        """Return the current status of the CSV connector.

        Returns
        -------
        ConnectorStatus
            Reports connected as ``True`` if the read file exists.
        """
        connected = False
        try:
            with open(self._read_path, "r"):
                connected = True
        except FileNotFoundError:
            pass

        return ConnectorStatus(
            connected=connected,
            last_sync=self._last_sync,
            record_count=self._record_count,
        )


# ── JSON Connector ─────────────────────────────────────


class JSONConnector(LabConnector):
    """Connector that reads observations from and writes suggestions to JSON files.

    Parameters
    ----------
    read_path:
        Path to a JSON file containing an array of observation dictionaries.
        Each dictionary should have at least ``parameters`` and ``kpi_values``
        keys.
    write_path:
        Path to write suggestion JSON array to.
    """

    def __init__(self, read_path: str, write_path: str) -> None:
        self._read_path = read_path
        self._write_path = write_path
        self._last_sync: float | None = None
        self._record_count: int = 0

    def read_observations(self) -> list[Observation]:
        """Read observations from the JSON file.

        Expects a JSON array of objects, each with at least ``parameters``
        and ``kpi_values`` keys. Optional keys: ``iteration``, ``qc_passed``,
        ``is_failure``, ``failure_reason``, ``timestamp``, ``metadata``.

        Returns
        -------
        list[Observation]
            Parsed observations from the JSON file.
        """
        try:
            with open(self._read_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        observations: list[Observation] = []
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue

            obs = Observation(
                iteration=entry.get("iteration", idx),
                parameters=entry.get("parameters", {}),
                kpi_values=entry.get("kpi_values", {}),
                qc_passed=entry.get("qc_passed", True),
                is_failure=entry.get("is_failure", False),
                failure_reason=entry.get("failure_reason"),
                timestamp=entry.get("timestamp", 0.0),
                metadata=entry.get("metadata", {}),
            )
            observations.append(obs)

        self._record_count = len(observations)
        self._last_sync = time.time()
        return observations

    def write_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        """Write suggestions as a JSON array.

        If the file already contains a JSON array, new suggestions are
        appended. Otherwise, a new array is created.

        Parameters
        ----------
        suggestions:
            Parameter suggestion dictionaries to write.
        """
        if not suggestions:
            return

        existing: list[dict[str, Any]] = []
        try:
            with open(self._write_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing = data
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        existing.extend(suggestions)

        with open(self._write_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        self._last_sync = time.time()

    def status(self) -> ConnectorStatus:
        """Return the current status of the JSON connector.

        Returns
        -------
        ConnectorStatus
            Reports connected as ``True`` if the read file exists.
        """
        connected = False
        try:
            with open(self._read_path, "r"):
                connected = True
        except FileNotFoundError:
            pass

        return ConnectorStatus(
            connected=connected,
            last_sync=self._last_sync,
            record_count=self._record_count,
        )


# ── In-Memory Connector ───────────────────────────────


class InMemoryConnector(LabConnector):
    """In-memory connector for testing and prototyping.

    Stores observations and suggestions in plain lists. No file I/O
    is performed.
    """

    def __init__(self) -> None:
        self._observations: list[Observation] = []
        self._suggestions: list[dict[str, Any]] = []
        self._last_sync: float | None = None

    def add_observation(self, obs: Observation) -> None:
        """Add an observation to the in-memory store.

        Parameters
        ----------
        obs:
            The observation to add.
        """
        self._observations.append(obs)
        self._last_sync = time.time()

    def read_observations(self) -> list[Observation]:
        """Return all stored observations.

        Returns
        -------
        list[Observation]
            A copy of the internal observation list.
        """
        self._last_sync = time.time()
        return list(self._observations)

    def write_suggestions(self, suggestions: list[dict[str, Any]]) -> None:
        """Store suggestions in memory.

        Parameters
        ----------
        suggestions:
            Parameter suggestion dictionaries to store.
        """
        self._suggestions.extend(suggestions)
        self._last_sync = time.time()

    def status(self) -> ConnectorStatus:
        """Return the current status of the in-memory connector.

        Returns
        -------
        ConnectorStatus
            Always reports connected as ``True``.
        """
        return ConnectorStatus(
            connected=True,
            last_sync=self._last_sync,
            record_count=len(self._observations),
        )
