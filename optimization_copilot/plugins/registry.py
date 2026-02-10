"""Plugin registry for discovering and instantiating algorithm backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.plugins.base import AlgorithmPlugin


@dataclass
class BackendPolicy:
    """Allow/deny list that constrains which plugins the registry will serve.

    If *allowlist* is non-empty only those names are permitted.
    If *denylist* is non-empty those names are excluded (applied after allowlist).
    """

    allowlist: list[str] = field(default_factory=list)
    denylist: list[str] = field(default_factory=list)

    def is_allowed(self, name: str) -> bool:
        if self.allowlist and name not in self.allowlist:
            return False
        if name in self.denylist:
            return False
        return True


class PluginRegistry:
    """Central registry for :class:`AlgorithmPlugin` implementations.

    Usage::

        registry = PluginRegistry()
        registry.register(RandomSampler)
        plugin = registry.get("random_sampler")
    """

    def __init__(self, policy: BackendPolicy | None = None) -> None:
        self._plugins: dict[str, type[AlgorithmPlugin]] = {}
        self._policy = policy or BackendPolicy()

    # -- mutation --------------------------------------------------------

    def register(self, plugin_class: type[AlgorithmPlugin]) -> None:
        """Register a plugin class.

        The class is instantiated briefly to discover its ``name()`` and
        then stored for later on-demand creation via :meth:`get`.

        Raises
        ------
        TypeError
            If *plugin_class* is not a subclass of :class:`AlgorithmPlugin`.
        ValueError
            If a plugin with the same name is already registered.
        """
        if not (isinstance(plugin_class, type) and issubclass(plugin_class, AlgorithmPlugin)):
            raise TypeError(
                f"Expected a subclass of AlgorithmPlugin, got {plugin_class!r}"
            )
        instance = plugin_class()
        plugin_name = instance.name()
        if plugin_name in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' is already registered")
        self._plugins[plugin_name] = plugin_class

    # -- queries ---------------------------------------------------------

    def get(self, name: str) -> AlgorithmPlugin:
        """Instantiate and return the plugin registered under *name*.

        Raises
        ------
        KeyError
            If no plugin with that name exists.
        PermissionError
            If the backend policy denies the requested name.
        """
        if name not in self._plugins:
            raise KeyError(
                f"Unknown plugin '{name}'. "
                f"Available: {self.list_plugins()}"
            )
        if not self._policy.is_allowed(name):
            raise PermissionError(
                f"Plugin '{name}' is blocked by the current backend policy"
            )
        return self._plugins[name]()

    def list_plugins(self) -> list[str]:
        """Return the names of all registered plugins (respecting policy)."""
        return [
            name
            for name in sorted(self._plugins)
            if self._policy.is_allowed(name)
        ]

    def match_capabilities(
        self,
        requirements: dict[str, Any],
    ) -> list[str]:
        """Return plugin names whose capabilities satisfy *requirements*.

        Each key/value in *requirements* is compared against the dict
        returned by the plugin's ``capabilities()`` method.  A plugin
        matches only if **every** requirement key is present in its
        capabilities and the capability value equals (or exceeds, for
        numeric comparisons) the required value.
        """
        matches: list[str] = []
        for name in self.list_plugins():
            plugin = self._plugins[name]()
            caps = plugin.capabilities()
            if _caps_satisfy(caps, requirements):
                matches.append(name)
        return sorted(matches)


def _caps_satisfy(caps: dict[str, Any], requirements: dict[str, Any]) -> bool:
    """Return True if *caps* satisfies every entry in *requirements*."""
    for key, required_value in requirements.items():
        if key not in caps:
            return False
        actual = caps[key]
        # For booleans: the capability must be True when required is True.
        if isinstance(required_value, bool):
            if required_value and not actual:
                return False
        # For numeric values: actual must be >= required.
        elif isinstance(required_value, (int, float)):
            if actual is None:
                # None conventionally means "unlimited" — passes.
                continue
            if actual < required_value:
                return False
        # For everything else: equality check.
        else:
            if actual != required_value:
                return False
    return True
