"""SENTINEL - shared POMDP belief-state engine for inventory management under uncertainty.

The notebook and the dashboard both import from this package, so there is a
single implementation of the mathematics. All parameters are synthetic.
"""
from .config import (ACTION_KEYS, ACTION_SHORT, ACTIONS, CHANNEL_KEYS, CHANNELS, COST_UNIT, STATES,
                     SentinelConfig, default_config)
from .pomdp_engine import SentinelEngine

__version__ = "1.0.0"
__all__ = ["SentinelEngine", "SentinelConfig", "default_config", "STATES", "ACTIONS", "ACTION_KEYS",
           "ACTION_SHORT", "CHANNELS", "CHANNEL_KEYS", "COST_UNIT", "__version__"]
