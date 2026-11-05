"""Security exceptions. Catch SecurityError to catch them all."""


class SecurityError(Exception):
    """An action was blocked by a security rule."""


class PathSecurityError(SecurityError):
    """A file path is unsafe."""


class CommandNotAllowed(SecurityError):
    """A command is not on the allow-list."""