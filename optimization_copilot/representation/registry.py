"""Registry for named RepresentationProvider instances.

Provides a central lookup for representation providers, enabling
the optimization engine to select providers by name. The default
NGramTanimoto provider is registered automatically.
"""

from __future__ import annotations

from optimization_copilot.representation.ngram_tanimoto import NGramTanimoto
from optimization_copilot.representation.provider import RepresentationProvider


class RepresentationRegistry:
    """Registry of named RepresentationProvider instances.

    Maintains a mapping from provider names to provider instances.
    The built-in NGramTanimoto provider is registered by default.

    Example::

        registry = RepresentationRegistry()
        assert "ngram_tanimoto" in registry
        provider = registry.get("ngram_tanimoto")
        registry.register(my_custom_provider)
    """

    def __init__(self) -> None:
        """Initialize registry with built-in providers."""
        self._providers: dict[str, RepresentationProvider] = {}
        # Register built-in default provider
        self.register(NGramTanimoto())

    def register(self, provider: RepresentationProvider) -> None:
        """Register a provider instance.

        If a provider with the same name already exists, it is replaced.

        Args:
            provider: RepresentationProvider instance to register.
        """
        self._providers[provider.name] = provider

    def get(self, name: str) -> RepresentationProvider | None:
        """Look up a provider by name.

        Args:
            name: Provider name to look up.

        Returns:
            The provider instance, or None if not found.
        """
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider name strings.
        """
        return list(self._providers.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a provider name is registered.

        Args:
            name: Provider name to check.

        Returns:
            True if the name is registered.
        """
        return name in self._providers

    def __len__(self) -> int:
        """Return the number of registered providers."""
        return len(self._providers)

    def __repr__(self) -> str:
        names = self.list_providers()
        return f"RepresentationRegistry(providers={names})"
