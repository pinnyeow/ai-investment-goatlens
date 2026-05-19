"""
GOATlens - Investor Agent Modules

Each agent represents a legendary investor's mental model and analysis approach.
"""

from .buffett import BuffettAgent
from .lynch import LynchAgent
from .graham import GrahamAgent
from .munger import MungerAgent
from .dalio import DalioAgent
from .burry import BurryAgent

__all__ = [
    "BuffettAgent",
    "LynchAgent",
    "GrahamAgent",
    "MungerAgent",
    "DalioAgent",
    "BurryAgent",
]
