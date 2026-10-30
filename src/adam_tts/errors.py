"""Error types. Each maps to one documented exit code (design §3.7)."""

from __future__ import annotations

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_UNSUPPORTED = 4
EXIT_BACKEND = 5
EXIT_ALIGN = 6
EXIT_PARTIAL = 7
EXIT_DEPENDENCY = 8


class AdamTTSError(Exception):
    exit_code = EXIT_INTERNAL
    type_name = "InternalError"

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        return {
            "code": self.exit_code,
            "type": self.type_name,
            "message": self.message,
            "hint": self.hint,
        }


class UsageError(AdamTTSError):
    exit_code = EXIT_USAGE
    type_name = "UsageError"


class ConfigError(AdamTTSError):
    exit_code = EXIT_CONFIG
    type_name = "ConfigError"


class UnsupportedParam(AdamTTSError):
    exit_code = EXIT_UNSUPPORTED
    type_name = "UnsupportedParam"


class BackendError(AdamTTSError):
    exit_code = EXIT_BACKEND
    type_name = "BackendError"


class AlignError(AdamTTSError):
    exit_code = EXIT_ALIGN
    type_name = "AlignError"


class PartialFailure(AdamTTSError):
    exit_code = EXIT_PARTIAL
    type_name = "PartialFailure"


class DependencyMissing(AdamTTSError):
    exit_code = EXIT_DEPENDENCY
    type_name = "DependencyMissing"


class NotImplementedBackend(DependencyMissing):
    """Backend is registered but not implemented in this version."""

    type_name = "NotImplementedBackend"
