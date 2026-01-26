"""
FPL Assistant - Fantasy Premier League Optimization and AI Assistant.

A local application that uses mathematical optimization and LLM integration
to provide weekly squad recommendations, transfer suggestions, and strategic advice.
"""

__version__ = "0.1.0"
__author__ = "Jim Hatton"

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
