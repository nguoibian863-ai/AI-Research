class ResearchException(Exception):
    """Base exception for research engine."""
    pass


class StateTransitionError(ResearchException):
    """Raised when an invalid state transition is requested."""
    pass


class BudgetExceededError(ResearchException):
    """Raised when hard limits (steps, calls, tokens) are breached."""
    pass


class ToolExecutionError(ResearchException):
    """Raised when an external tool (search, fetch) fails."""
    pass


class VerificationError(ResearchException):
    """Raised when claim or evidence integrity check fails."""
    pass


class ModelInferenceError(ResearchException):
    """Raised when LLM call fails or schema validation fails."""
    pass
