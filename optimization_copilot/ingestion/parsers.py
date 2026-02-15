"""Data parsers for CSV and JSON formats using Python stdlib only."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


class CSVParser:
    """Parse CSV files into list[dict[str, str]] using stdlib csv."""

    @staticmethod
    def parse_file(path: str) -> list[dict[str, str]]:
        """Parse a CSV file.  Returns list of row dicts."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def parse_string(csv_string: str) -> list[dict[str, str]]:
        """Parse a CSV string.  Returns list of row dicts."""
        reader = csv.DictReader(io.StringIO(csv_string))
        return list(reader)


class JSONParser:
    """Parse JSON files into list[dict[str, Any]].

    Supports two formats:
    - List of dicts: [{"col1": val1, ...}, ...]
    - Dict of lists: {"col1": [val1, val2, ...], "col2": [...]}
    """

    @staticmethod
    def parse_file(path: str) -> list[dict[str, Any]]:
        """Parse a JSON file.  Returns list of row dicts."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return JSONParser._normalize(data)

    @staticmethod
    def parse_string(json_string: str) -> list[dict[str, Any]]:
        """Parse a JSON string.  Returns list of row dicts."""
        data = json.loads(json_string)
        return JSONParser._normalize(data)

    @staticmethod
    def _normalize(data: Any) -> list[dict[str, Any]]:
        """Normalize both list-of-dicts and dict-of-lists to list-of-dicts."""
        if isinstance(data, list):
            # List of dicts.
            return [dict(row) if isinstance(row, dict) else {} for row in data]

        if isinstance(data, dict):
            # Dict of lists — transpose.
            keys = list(data.keys())
            if not keys:
                return []
            # All values should be lists.
            lists = {}
            max_len = 0
            for k in keys:
                v = data[k]
                if isinstance(v, list):
                    lists[k] = v
                    max_len = max(max_len, len(v))
                else:
                    # Single value — wrap in list.
                    lists[k] = [v]
                    max_len = max(max_len, 1)

            rows: list[dict[str, Any]] = []
            for i in range(max_len):
                row: dict[str, Any] = {}
                for k in keys:
                    vals = lists[k]
                    row[k] = vals[i] if i < len(vals) else None
                rows.append(row)
            return rows

        raise ValueError(
            f"JSON data must be a list of dicts or a dict of lists, "
            f"got {type(data).__name__}"
        )
