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

# Experts
from Expert import Expert
from UnreliableExpert import UnreliableExpert
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

def make_panels(panel_dict,env,mu=10,gamma=0.01):
    '''
    Make a list of panels of experts

    Input:
        panel_dict - dict mapping panel name to list of true reliabilities
        env - instance of InfluenceDiagram class representing environment
        mu - interval parameter (number of trials between advice givings)
            default: 10
        gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
            default: 0.01

    Output:
        panels - list of Panel objects

    Todo:
        allow for each expert in a panel to have different parameters
    '''
    oracle = TruePolicyAgent(env) # Oracle used to retrieve best advice for each expert
    panels = [] # List of panels
    for panel in panel_dict:
        panels.append(Panel(env,panel,oracle,panel_dict[panel],mu,gamma)) # Create Panel object
    return panels
