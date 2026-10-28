"""Application errors that map cleanly to HTTP responses (no internals leaked)."""


class AppError(Exception):
    status_code = 500
    error = "internal_error"
    default_message = "Internal server error."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class InvalidInputError(AppError):
    status_code = 422
    error = "invalid_input"
    default_message = "Invalid input."


class InferenceTimeoutError(AppError):
    status_code = 504
    error = "inference_timeout"
    default_message = "Model inference timed out."


class OverloadedError(AppError):
    status_code = 503
    error = "overloaded"
    default_message = "Too many inference requests in flight; retry later."


class CacheUnavailableError(AppError):
    status_code = 503
    error = "cache_unavailable"
    default_message = "Prediction cache is unavailable and the failure policy is 'fail'."
