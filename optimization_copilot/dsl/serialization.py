"""JSON round-trip and simple YAML-subset formatter for OptimizationSpec.

Pure stdlib only -- no PyYAML dependency. The YAML subset handles the
structure that OptimizationSpec.to_dict() produces: nested dicts, lists
of dicts, and scalar values (str, int, float, bool, None/null).
"""

from __future__ import annotations

import json
import os
from typing import Any

from optimization_copilot.dsl.spec import OptimizationSpec


# ── JSON serialization ────────────────────────────────


def to_json(spec: OptimizationSpec, indent: int = 2) -> str:
    """Serialize an OptimizationSpec to a JSON string."""
    return json.dumps(spec.to_dict(), indent=indent)


def from_json(json_str: str) -> OptimizationSpec:
    """Deserialize an OptimizationSpec from a JSON string."""
    data = json.loads(json_str)
    return OptimizationSpec.from_dict(data)


# ── YAML-subset serialization ─────────────────────────

# Characters that force quoting in YAML scalar values.
_YAML_SPECIAL_CHARS = set(":{}\n\t[]&*!|>'\"%@`,?#")

# Reserved YAML scalars that need quoting to stay as strings.
_YAML_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~"}


def _yaml_format_scalar(value: Any) -> str:
    """Format a Python scalar as a YAML scalar string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    # str
    s = str(value)
    if (
        not s
        or s.lower() in _YAML_RESERVED
        or any(c in _YAML_SPECIAL_CHARS for c in s)
        or s.startswith(" ")
        or s.endswith(" ")
        or s.startswith("- ")
    ):
        # Use double-quoted style, escaping embedded double-quotes and
        # backslashes.
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _yaml_dump(data: Any, indent: int = 0) -> str:
    """Recursively format *data* as a minimal YAML string.

    Parameters
    ----------
    data:
        A dict, list, or scalar value.
    indent:
        Current indentation level (number of spaces).
    """
    prefix = " " * indent

    if isinstance(data, dict):
        if not data:
            return f"{prefix}{{}}\n"
        lines: list[str] = []
        for key, value in data.items():
            key_str = _yaml_format_scalar(key)
            if isinstance(value, dict):
                if not value:
                    lines.append(f"{prefix}{key_str}: {{}}\n")
                else:
                    lines.append(f"{prefix}{key_str}:\n")
                    lines.append(_yaml_dump(value, indent + 2))
            elif isinstance(value, list):
                if not value:
                    lines.append(f"{prefix}{key_str}: []\n")
                else:
                    lines.append(f"{prefix}{key_str}:\n")
                    lines.append(_yaml_dump(value, indent + 2))
            else:
                lines.append(
                    f"{prefix}{key_str}: {_yaml_format_scalar(value)}\n"
                )
        return "".join(lines)

    if isinstance(data, list):
        if not data:
            return f"{prefix}[]\n"
        lines = []
        for item in data:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}\n")
                else:
                    # First key-value pair goes on the same line as "- ".
                    first = True
                    for k, v in item.items():
                        k_str = _yaml_format_scalar(k)
                        if first:
                            if isinstance(v, (dict, list)) and v:
                                lines.append(f"{prefix}- {k_str}:\n")
                                lines.append(
                                    _yaml_dump(v, indent + 4)
                                )
                            else:
                                scalar = (
                                    "[]" if isinstance(v, list) and not v
                                    else "{}" if isinstance(v, dict) and not v
                                    else _yaml_format_scalar(v)
                                )
                                lines.append(
                                    f"{prefix}- {k_str}: {scalar}\n"
                                )
                            first = False
                        else:
                            # Subsequent keys are indented to align with the
                            # first key (indent + 2 spaces for the "- ").
                            sub_prefix = " " * (indent + 2)
                            if isinstance(v, (dict, list)) and v:
                                lines.append(f"{sub_prefix}{k_str}:\n")
                                lines.append(
                                    _yaml_dump(v, indent + 4)
                                )
                            else:
                                scalar = (
                                    "[]" if isinstance(v, list) and not v
                                    else "{}" if isinstance(v, dict) and not v
                                    else _yaml_format_scalar(v)
                                )
                                lines.append(
                                    f"{sub_prefix}{k_str}: {scalar}\n"
                                )
            elif isinstance(item, list):
                # Nested list -- unusual for our spec but handled.
                lines.append(f"{prefix}-\n")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(
                    f"{prefix}- {_yaml_format_scalar(item)}\n"
                )
        return "".join(lines)

    # Bare scalar at the top level (shouldn't normally happen for a spec).
    return f"{prefix}{_yaml_format_scalar(data)}\n"


def to_yaml_string(spec: OptimizationSpec) -> str:
    """Serialize an OptimizationSpec to a simple YAML-subset string.

    No PyYAML dependency is used.  The output handles nested dicts, lists,
    and scalar values with indentation-based nesting.
    """
    return _yaml_dump(spec.to_dict())


# ── YAML-subset parser ────────────────────────────────


def _parse_yaml_scalar(raw: str) -> Any:
    """Parse a raw YAML scalar string into a Python value."""
    s = raw.strip()

    # Empty string
    if not s:
        return ""

    # Quoted string
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        inner = s[1:-1]
        # Unescape common sequences for double-quoted strings.
        if s.startswith('"'):
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner

    # null
    if s in ("null", "~"):
        return None

    # bool
    if s in ("true", "yes", "on"):
        return True
    if s in ("false", "no", "off"):
        return False

    # int
    try:
        return int(s)
    except ValueError:
        pass

    # float
    try:
        return float(s)
    except ValueError:
        pass

    return s


def _indent_of(line: str) -> int:
    """Return the number of leading spaces of *line*."""
    return len(line) - len(line.lstrip(" "))


def from_yaml_string(yaml_str: str) -> OptimizationSpec:
    """Parse a simple YAML-subset string into an OptimizationSpec.

    Handles indented key-value pairs, list items with ``- `` prefix, and
    nested objects.  Does **not** support anchors, aliases, or multi-line
    scalars.
    """
    data = _parse_yaml_block(yaml_str)
    return OptimizationSpec.from_dict(data)


def _parse_yaml_block(text: str) -> Any:
    """Parse a YAML-subset block of text into nested Python structures."""
    lines = text.splitlines()
    result, _ = _parse_yaml_lines(lines, 0, 0)
    return result


def _parse_yaml_lines(
    lines: list[str], start: int, base_indent: int
) -> tuple[Any, int]:
    """Recursive line-by-line YAML-subset parser.

    Returns ``(parsed_value, next_line_index)``.
    """
    if start >= len(lines):
        return {}, start

    # Skip blank lines at the start.
    idx = start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return {}, idx

    first_line = lines[idx]
    stripped = first_line.strip()

    # Detect whether we are in a list or dict context.
    if stripped.startswith("- "):
        return _parse_yaml_list(lines, idx, base_indent)
    else:
        return _parse_yaml_dict(lines, idx, base_indent)


def _parse_yaml_dict(
    lines: list[str], start: int, base_indent: int
) -> tuple[dict, int]:
    """Parse a YAML mapping (dict) starting at *start*."""
    result: dict[str, Any] = {}
    idx = start

    while idx < len(lines):
        line = lines[idx]

        # Skip blank lines.
        if not line.strip():
            idx += 1
            continue

        current_indent = _indent_of(line)
        if current_indent < base_indent:
            break

        stripped = line.strip()

        # Skip lines that are list items -- they belong to a parent list.
        if stripped.startswith("- ") and current_indent == base_indent:
            break

        if ":" not in stripped:
            idx += 1
            continue

        # Split on the first colon.
        colon_pos = stripped.index(":")
        key_raw = stripped[:colon_pos].strip()
        value_raw = stripped[colon_pos + 1 :].strip()

        key = _parse_yaml_scalar(key_raw)

        if value_raw == "":
            # Value is a nested block (dict or list) on subsequent lines.
            child_indent = current_indent + 2
            # Find the actual child indent from the next non-blank line.
            peek = idx + 1
            while peek < len(lines) and not lines[peek].strip():
                peek += 1
            if peek < len(lines):
                child_indent = _indent_of(lines[peek])
                if child_indent <= current_indent:
                    # Empty nested value -- treat as empty string.
                    result[str(key)] = ""
                    idx += 1
                    continue
            else:
                result[str(key)] = ""
                idx += 1
                continue
            child_value, idx = _parse_yaml_lines(
                lines, peek, child_indent
            )
            result[str(key)] = child_value
        elif value_raw == "[]":
            result[str(key)] = []
            idx += 1
        elif value_raw == "{}":
            result[str(key)] = {}
            idx += 1
        else:
            result[str(key)] = _parse_yaml_scalar(value_raw)
            idx += 1

    return result, idx


def _parse_yaml_list(
    lines: list[str], start: int, base_indent: int
) -> tuple[list, int]:
    """Parse a YAML sequence (list) starting at *start*."""
    result: list[Any] = []
    idx = start

    while idx < len(lines):
        line = lines[idx]

        # Skip blank lines.
        if not line.strip():
            idx += 1
            continue

        current_indent = _indent_of(line)
        if current_indent < base_indent:
            break

        stripped = line.strip()

        if not stripped.startswith("- "):
            # Not a list item at this level; we've exited the list.
            if current_indent == base_indent:
                break
            # Could be continuation of a previous list-item dict.
            idx += 1
            continue

        if current_indent > base_indent:
            # Nested list item -- belongs to a child.
            break

        # Remove the "- " prefix to get the item content.
        item_content = stripped[2:].strip()

        if item_content == "":
            # Bare "- " -- nested block follows.
            child_indent = current_indent + 2
            peek = idx + 1
            while peek < len(lines) and not lines[peek].strip():
                peek += 1
            if peek < len(lines) and _indent_of(lines[peek]) > current_indent:
                child_indent = _indent_of(lines[peek])
                child_value, idx = _parse_yaml_lines(
                    lines, peek, child_indent
                )
                result.append(child_value)
            else:
                result.append(None)
                idx += 1
        elif item_content == "[]":
            result.append([])
            idx += 1
        elif item_content == "{}":
            result.append({})
            idx += 1
        elif ":" in item_content:
            # This is a dict item in the list.  Parse the first key-value
            # pair from the content after "- ", then look for continuation
            # lines at indent+2.
            colon_pos = item_content.index(":")
            first_key = item_content[:colon_pos].strip()
            first_val_raw = item_content[colon_pos + 1 :].strip()

            item_dict: dict[str, Any] = {}
            first_key_parsed = str(_parse_yaml_scalar(first_key))

            if first_val_raw == "":
                # Nested block under the first key.
                child_indent = current_indent + 4
                peek = idx + 1
                while peek < len(lines) and not lines[peek].strip():
                    peek += 1
                if (
                    peek < len(lines)
                    and _indent_of(lines[peek]) > current_indent + 1
                ):
                    child_indent = _indent_of(lines[peek])
                    child_value, idx = _parse_yaml_lines(
                        lines, peek, child_indent
                    )
                    item_dict[first_key_parsed] = child_value
                else:
                    item_dict[first_key_parsed] = ""
                    idx += 1
            elif first_val_raw == "[]":
                item_dict[first_key_parsed] = []
                idx += 1
            elif first_val_raw == "{}":
                item_dict[first_key_parsed] = {}
                idx += 1
            else:
                item_dict[first_key_parsed] = _parse_yaml_scalar(
                    first_val_raw
                )
                idx += 1

            # Now read continuation keys at indent + 2 (aligned with first
            # key after "- ").
            continuation_indent = current_indent + 2
            while idx < len(lines):
                cline = lines[idx]
                if not cline.strip():
                    idx += 1
                    continue
                c_indent = _indent_of(cline)
                if c_indent < continuation_indent:
                    break
                if c_indent == continuation_indent:
                    c_stripped = cline.strip()
                    if c_stripped.startswith("- "):
                        # New list item at a different level.
                        break
                    if ":" in c_stripped:
                        c_colon = c_stripped.index(":")
                        c_key = str(
                            _parse_yaml_scalar(c_stripped[:c_colon].strip())
                        )
                        c_val_raw = c_stripped[c_colon + 1 :].strip()
                        if c_val_raw == "":
                            # Nested block.
                            peek = idx + 1
                            while (
                                peek < len(lines)
                                and not lines[peek].strip()
                            ):
                                peek += 1
                            if (
                                peek < len(lines)
                                and _indent_of(lines[peek])
                                > continuation_indent
                            ):
                                child_indent = _indent_of(lines[peek])
                                child_value, idx = _parse_yaml_lines(
                                    lines, peek, child_indent
                                )
                                item_dict[c_key] = child_value
                            else:
                                item_dict[c_key] = ""
                                idx += 1
                        elif c_val_raw == "[]":
                            item_dict[c_key] = []
                            idx += 1
                        elif c_val_raw == "{}":
                            item_dict[c_key] = {}
                            idx += 1
                        else:
                            item_dict[c_key] = _parse_yaml_scalar(c_val_raw)
                            idx += 1
                    else:
                        idx += 1
                elif c_indent > continuation_indent:
                    # Belongs to a child of the last key; skip forward.
                    idx += 1
                else:
                    break

            result.append(item_dict)
        else:
            # Plain scalar list item.
            result.append(_parse_yaml_scalar(item_content))
            idx += 1

    return result, idx


# ── File I/O ──────────────────────────────────────────


def to_file(
    spec: OptimizationSpec, path: str, format: str = "json"
) -> None:
    """Serialize an OptimizationSpec to a file.

    Parameters
    ----------
    spec:
        The spec to write.
    path:
        Destination file path.
    format:
        ``"json"`` or ``"yaml"``.  Defaults to ``"json"``.
    """
    fmt = format.lower()
    if fmt == "json":
        content = to_json(spec)
    elif fmt in ("yaml", "yml"):
        content = to_yaml_string(spec)
    else:
        raise ValueError(f"Unsupported format: {format!r} (use 'json' or 'yaml')")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def from_file(path: str) -> OptimizationSpec:
    """Deserialize an OptimizationSpec from a file.

    Format is auto-detected from the file extension (``.json``, ``.yaml``,
    or ``.yml``).
    """
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    if ext == ".json":
        return from_json(content)
    elif ext in (".yaml", ".yml"):
        return from_yaml_string(content)
    else:
        raise ValueError(
            f"Cannot auto-detect format from extension {ext!r}. "
            "Use .json, .yaml, or .yml."
        )
