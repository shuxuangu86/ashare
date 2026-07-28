class FactorError(Exception):
    """Base error for factor infrastructure."""


class FactorSpecError(FactorError, ValueError):
    """A factor definition violates the metadata contract."""


class FactorRegistryError(FactorError, ValueError):
    """A registry operation would make metadata ambiguous."""


class FactorLifecycleError(FactorRegistryError):
    """A requested lifecycle transition is not permitted."""
