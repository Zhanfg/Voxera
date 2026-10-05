"""Voxera training-only tools.

This package lives outside src/ and is intentionally excluded from the runtime
wheel. Making it a package allows repository tests and training utilities to
share validated preprocessing code without adding runtime dependencies.
"""
