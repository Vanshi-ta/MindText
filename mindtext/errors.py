"""Exception hierarchy so the CLI can print clean messages instead of tracebacks."""


class MindTextError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(MindTextError, ValueError):
    """Invalid or inconsistent configuration."""


class DataError(MindTextError, ValueError):
    """Dataset missing, malformed, or failing validation."""


class LeakageError(MindTextError):
    """A train/validation/test separation guarantee was violated."""


class ModelError(MindTextError, ValueError):
    """Unknown model type, incompatible input, or unusable model object."""


class IntegrityError(MindTextError):
    """A serialized artifact does not match its recorded checksum."""


class MissingDependencyError(MindTextError, ImportError):
    """An optional dependency required by the requested feature is not installed."""
