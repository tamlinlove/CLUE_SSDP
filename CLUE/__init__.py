'''
Creates the CLUE_SSDP module
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Environments
from InfluenceDiagram import InfluenceDiagram
from RandomSSDP import RandomSSDP

# Agents
from Agent import Agent
from TruePolicyAgent import TruePolicyAgent
from BaselineAgent import BaselineAgent
from NaiveAdviceFollower import NaiveAdviceFollower
from ClueAgent import ClueAgent
from DecayedRelianceAgent import DecayedRelianceAgent

# Experts
from Expert import Expert
from UnreliableExpert import UnreliableExpert
from PartiallyReliableExpert import PartiallyReliableExpert
from Panel import Panel

# Helpers
from Utility import Utility
from StateTable import StateTable
import Experiment
import Plot

# Environment keys
envs = {
"RandomSSDP":RandomSSDP
}

def make(name,**kwargs):
    """
    Creates a single stage decision problem environment

    Usage:
        env = make(name,**kwargs)

    Input:
        name - a valid environment name
        **kwargs - passed to environment instantiation
    Output:
        env - an environment object corresponding to the given name
    """
    if name not in envs:
        error_message = name+" not recognised!"
    else:
        return envs[name](**kwargs)
    raise Exception(error_message)
