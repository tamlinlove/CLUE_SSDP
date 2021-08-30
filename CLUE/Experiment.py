'''
A bunch of methods for running and facilitating experiments
'''
import numpy as np
import os
import csv

from TruePolicyAgent import TruePolicyAgent
from BaselineAgent import BaselineAgent
from NaiveAdviceFollower import NaiveAdviceFollower
from ClueAgent import ClueAgent
from DecayedRelianceAgent import DecayedRelianceAgent

from RandomSSDP import RandomSSDP

from Panel import Panel
from PartiallyReliableExpert import PartiallyReliableExpert

# Dict mapping agent name to whether or not they take panel advice
takes_advice = {
    "Baseline Agent":False,
    "CLUE":True,
    "CLUE Regular Update":True,
    "Decayed Reliance":True,
    "NAF":True,
    "Naive CLUE":True,
    "True Policy Agent":False
}

# Dict mapping agent name to whether or not they store rho history
keeps_rho_history = {
    "Baseline Agent":False,
    "CLUE":True,
    "CLUE Regular Update":True,
    "Decayed Reliance":False,
    "NAF":False,
    "Naive CLUE":True,
    "True Policy Agent":False
}

def expert_param_test(env,agents,panel_dict,mus,gammas,trials,runs,display=False,display_interval=10):
    '''
    Run the expert parameter test

    Input:
        env - instance of InfluenceDiagram representing the environment
        agents - dict mapping agent name to Agent object
        panel_dict - dict mapping panel name to list of true reliabilities
        mus - list of mu values to be tested
        gammas - list of gamma values to be tested
        runs - number of runs
        trials - number of trials
        display - boolean. Whether or not the experiment will print out updates
        display_interval - if display = True, this determines the number of runs between printouts
    Output:
        regret - dict mapping agents, panels, mus and gammas to array of regret
            regret[agent][panel][mu][gamma] = array of size runs
    '''
    # Initialise dict
    regret = {}
    for agent in agents:
        if agents[agent].takes_advice():
            regret[agent] = {}
            for panel in panel_dict:
                regret[agent][panel] = {}
                for mu in mus:
                    regret[agent][panel][str(mu)] = {}
                    for gamma in gammas:
                        regret[agent][panel][str(mu)][str(gamma)] = []

        else:
            regret[agent] = []

    # Initialise oracle
    oracle = TruePolicyAgent(env)

    # Run experiments
    for agent in agents:
        if agents[agent].takes_advice(): # Panels matter
            for mu in mus:
                for gamma in gammas:
                    panels = make_panels(panel_dict,env,mu=mu,gamma=gamma)
                    for panel in panels:
                        if display:
                            print("==="+agent+" : "+panel.name+" : mu = "+str(mu)+",gamma = "+str(gamma)+"===")
                        for r in range(runs):
                            if display and r % display_interval == 0:
                                print("Run "+str(r))
                            this_regret = run_panel_regret(env,agents[agent],panel,trials,oracle)
                            regret[agent][panel.name][str(mu)][str(gamma)].append(this_regret)
        else: # Panels don't matter
            if display:
                print("==="+agent+"===")
            for r in range(runs):
                if display and r % display_interval == 0:
                    print("Run "+str(r))
                this_regret = run_standard_regret(env,agents[agent],trials,oracle)
                regret[agent].append(this_regret)
    return regret


def make_agents(agent_list,env,trials,**kwargs):
    '''
    Take in a list of agent names and return a dict of agents

    Input:
        agent_list - list of agents, each item must be one of the following
            "Baseline Agent" - baseline agent
            "CLUE" - CLUE agent
            "Decayed Reliance" - Agent with decayed reliance on advice
            "NAF" - NAF agent
            "Naive CLUE" - CLUE with no_bayes set to True
            "True Policy Agent" - true policy agent
        env - instance of InfluenceDiagram class representing environment
        trials - number of trials
        **kwargs - passed to each agent initialisation
    Output:
        agents - dict mapping agent name to agent object
    '''
    agents = {}
    for agent in agent_list:
        if agent == "Baseline Agent":
            agents[agent] = BaselineAgent(env,trials,**kwargs)
        elif agent == "CLUE":
            agents[agent] = ClueAgent(env,trials=trials,**kwargs)
        elif agent == "CLUE Regular Update":
            agents[agent] = ClueAgent(env,trials=trials,regular_update=True,**kwargs)
        elif agent == "Decayed Reliance":
            agents[agent] = DecayedRelianceAgent(env,trials,**kwargs)
        elif agent == "NAF":
            agents[agent] = NaiveAdviceFollower(env,trials=trials,**kwargs)
        elif agent == "Naive CLUE":
            agents[agent] = ClueAgent(env,trials=trials,no_bayes=True,**kwargs)
        elif agent == "True Policy Agent":
            agents[agent] = TruePolicyAgent(env,**kwargs)
    return agents

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

def make_panel_comparison_dicts(agents,panels,trials,runs):
    '''
    Initialise dictionaries for panel comparison experiment

    Input:
        agents - dict mapping agent name to Agent object
        panels - list of Panel objects
        trials - number of trials
        runs - number of runs

    Output:
        rewards - dict for storing rewards
            if agent takes advice, rewards[agent][panel] = zero array of size runs x trials
            otherwise, rewards[agent] = zero array of size runs x trials
        rhos - dict for storing rhos
            if agent takes advice and stores rhos, rhos[agent][panel][expert] = zero array of size runs x trials
            otherwise, rhos[agent] is not specified
    '''
    rewards = {}
    rhos = {}

    for agent in agents:
        if agents[agent].takes_advice(): # Panel has an effect
            rewards[agent] = {}
            if agents[agent].get_history() is not None: # Agent keeps track of rhos
                rhos[agent] = {}
            for panel in panels:
                rewards[agent][panel.name] = np.zeros((runs,trials))
                if agents[agent].get_history() is not None:
                    rhos[agent][panel.name] = {}
                    for expert in panel.experts:
                        rhos[agent][panel.name][expert] = np.zeros((runs,trials))
        else:
            rewards[agent] = np.zeros((runs,trials))
    return rewards,rhos

def panel_comparison(env,agents,panels,trials,runs,display=False,display_interval=10):
    '''
    Run an experiment comparing rewards obtained over trials by each agent-panel configuration

    Input:
        env - instance of InfluenceDiagram representing environment
        agents - dict mapping agent name to Agent object
        panels - list of Panel objects
        trials - number of trials for each run
        runs - number of runs over which rewards will be averaged
        display - boolean. Whether or not the experiment will print out updates
        display_interval - if display = True, this determines the number of runs between printouts

    Output:
        rewards - dict for storing rewards
            if agent takes advice, rewards[agent][panel] = array of size runs x trials
            otherwise, rewards[agent] = array of size runs x trials
        rhos - dict for storing rhos
            if agent takes advice and stores rhos, rhos[agent][panel][expert] = array of size runs x trials
            otherwise, rhos[agent] is not specified
    '''
    # Initialise reward and rho dicts
    rewards,rhos = make_panel_comparison_dicts(agents,panels,trials,runs)
    for agent in agents:
        if agents[agent].takes_advice(): # Panels matter
            for panel in panels:
                if display:
                    print("==="+agent+" : "+panel.name+"===")
                for r in range(runs):
                    if display and r % display_interval == 0:
                        print("Run "+str(r))
                    rewards[agent][panel.name][r,:] = run_panel(env,agents[agent],panel,trials)
                    history = agents[agent].get_history()
                    if history is not None:
                        for expert in panel.experts:
                            rhos[agent][panel.name][expert][r,:] = history["rho"][expert]
        else: # Panels don't matter
            if display:
                print("==="+agent+"===")
            for r in range(runs):
                if display and r % display_interval == 0:
                    print("Run "+str(r))
                rewards[agent][r,:] = run_standard(env,agents[agent],trials)
    return rewards,rhos

def panel_comparison_partially_reliable_experts(env,agent_list,panel_dict,trials,runs,display=False,display_interval=10):
    '''
    Run an experiment comparing rewards obtained over trials by each agent-panel configuration
    Each run uses a different configuration of hidden nodes for each expert

    Input:
        TODO
    Output:
        TODO
    '''
    # Initialise empty dicts for rewards and rho
    rewards = {}
    rhos = {}

    for agent in agent_list:
        if takes_advice[agent]: # Panel has an effect
            rewards[agent] = {}
            if keeps_rho_history[agent]: # Agent keeps track of rhos
                rhos[agent] = {}
            for panel in panel_dict:
                rewards[agent][panel] = np.zeros((runs,trials))
                if keeps_rho_history[agent]:
                    rhos[agent][panel] = {}
                    for expert in panel_dict[panel]:
                        rhos[agent][panel][str(expert)] = np.zeros((runs,trials))
        else:
            rewards[agent] = np.zeros((runs,trials))
    # Initialise agents
    agents = make_agents(agent_list,env,trials)
    # Run experiment
    for r in range(runs):
        if display and r % display_interval == 0:
            print("Run "+str(r))
        # Initialise panels
        panels = []
        for panel in panel_dict:
            experts = {}
            for num in panel_dict[panel]:
                hidden_nodes = np.random.choice(env.chance,num,replace=False)
                experts[str(num)] = PartiallyReliableExpert(env,hidden_nodes)
            panels.append(Panel(env,panel,None,None,experts=experts))
        # Run
        for agent in agents:
            if agents[agent].takes_advice(): # Panels have an effect
                for panel in panels:
                    rewards[agent][panel.name][r,:] = run_panel(env,agents[agent],panel,trials)
                    history = agents[agent].get_history()
                    if history is not None:
                        for expert in panel.experts:
                            rhos[agent][panel.name][expert][r,:] = history["rho"][expert]
            else: # Panels don't matter
                rewards[agent][r,:] = run_standard(env,agents[agent],trials)
    return rewards,rhos

def panel_comparison_random_envs(agent_list,panel_dict,trials,runs,display=False,display_interval=10,num_chance=10,num_decision=3):
    '''
    Run an experiment comparing rewards obtained over trials by each agent-panel configuration
    Each run on a different randomly generated environment

    Input:
        agent_list - list of agent names, used to generate agents (see make_agents in __init__.py)
        panel_dict - dict mapping panel names to list of reliabilities
        trials - number of trials for each run
        runs - number of runs over which rewards will be averaged
        display - boolean. Whether or not the experiment will print out updates
            default: False
        display_interval - if display = True, this determines the number of runs between printouts
            default: True
        num_chance - number of state variables
            default: 10
        num_decision - number of action variables
            default: 3
    '''
    # Initialise empty dicts for rewards and rho
    rewards = {}
    rhos = {}

    for agent in agent_list:
        if takes_advice[agent]: # Panel has an effect
            rewards[agent] = {}
            if keeps_rho_history[agent]: # Agent keeps track of rhos
                rhos[agent] = {}
            for panel in panel_dict:
                rewards[agent][panel] = np.zeros((runs,trials))
                if keeps_rho_history[agent]:
                    rhos[agent][panel] = {}
                    for expert in panel_dict[panel]:
                        rhos[agent][panel][str(expert)] = np.zeros((runs,trials))
        else:
            rewards[agent] = np.zeros((runs,trials))
    # Run experiment
    for r in range(runs):
        if display and r % display_interval == 0:
            print("Run "+str(r))
        # Initialise environment
        env = RandomSSDP(num_chance=num_chance,num_decision=num_decision,seed=r)
        # Initialise agents
        agents = make_agents(agent_list,env,trials)
        # Initialise panels
        panels = make_panels(panel_dict,env)
        # Run
        for agent in agents:
            if agents[agent].takes_advice(): # Panels have an effect
                for panel in panels:
                    rewards[agent][panel.name][r,:] = run_panel(env,agents[agent],panel,trials)
                    history = agents[agent].get_history()
                    if history is not None:
                        for expert in panel.experts:
                            rhos[agent][panel.name][expert][r,:] = history["rho"][expert]
            else: # Panels don't matter
                rewards[agent][r,:] = run_standard(env,agents[agent],trials)
    return rewards,rhos

def regret_test(env,agents,panels,trials,runs,display=False,display_interval=10):
    '''
    Run each agent and stores regret

    Input:
        env - instance of InfluenceDiagram class representing environment
        agents - dict mapping agent name to agent object
        panels - dict mapping panel name to panel object
        trials - number of trials
        runs - number of runs
        display - boolean. Whether or not the experiment will print out updates
        display_interval - if display = True, this determines the number of runs between printouts
    Output:
        regret - dict mapping agents and panels to array of regret
            regret[agent][panel] = array of size runs
    '''
    # Initialise dict
    regret = {}
    for agent in agents:
        if agents[agent].takes_advice():
            regret[agent] = {}
            for panel in panels:
                regret[agent][panel] = []
        else:
            regret[agent] = []

    # Initialise oracle
    oracle = TruePolicyAgent(env)

    # Run experiments
    for agent in agents:
        if agents[agent].takes_advice(): # Panels matter
            for panel in panels:
                if display:
                    print("==="+agent+" : "+panel.name+"===")
                for r in range(runs):
                    if display and r % display_interval == 0:
                        print("Run "+str(r))
                    this_regret = run_panel_regret(env,agents[agent],panel,trials,oracle)
                    regret[agent][panel].append(this_regret)
        else: # Panels don't matter
            if display:
                print("==="+agent+"===")
            for r in range(runs):
                if display and r % display_interval == 0:
                    print("Run "+str(r))
                this_regret = run_standard_regret(env,agents[agent],trials,oracle)
                regret[agent].append(this_regret)
    return regret


def run_panel(env,agent,panel,trials):
    '''
    Perform one SSDP learning session with a single agent and a panel of experts

    Input:
        env - instance of InfluenceDiagram representing environment
        agent - instance of Agent
        panel - instance of Panel
        trials - number of trials
    Output:
        rewards - list of rewards obtained this run. length = trials
    '''
    rewards = [] # Empty list of rewards
    agent.reset(panel) # Reset agent to forget any older training
    panel.reset() # Reset each expert in the panel
    for i in range(trials): # Loop through each trial
        state = env.reset() # Reset the environment to an empty state, then sample initial observations
        action = agent.act(state) # Compute the action to take
        reward = env.step(action) # Feed action to environment, get reward
        advice = panel.advise(state,action,reward) # Each expert may or may not advise the agent
        agent.learn(state,action,reward,advice) # Perform some learning
        rewards.append(reward) # Add reward to list
    return rewards

def run_panel_regret(env,agent,panel,trials,oracle):
    '''
    Perform one SSDP learning session with a single agent and a panel of experts
    Store regret

    Input:
        env - instance of InfluenceDiagram representing environment
        agent - instance of Agent
        panel - instance of Panel
        trials - number of trials
        oracle - instance of TruePolicyAgent
    Output:
        regret - regret accumulated this run
    '''
    regret = 0
    agent.reset(panel) # Reset agent to forget any older training
    panel.reset() # Reset each expert in the panel
    for i in range(trials): # Loop through each trial
        state = env.reset() # Reset the environment to an empty state, then sample initial observations
        action = agent.act(state) # Compute the action to take
        reward = env.step(action) # Feed action to environment, get reward
        advice = panel.advise(state,action,reward) # Each expert may or may not advise the agent
        agent.learn(state,action,reward,advice) # Perform some learning
        oracle_action = oracle.act(state) # Get optimal action
        oracle_reward = oracle.expected_utility(state,oracle_action) # Get max reward
        regret += oracle_reward - reward # Calculate regret, add to running total
    return regret

def run_standard(env,agent,trials):
    '''
    Perform one SSDP learning session with a single agent and no experts

    Input:
        env - instance of InfluenceDiagram representing environment
        agent - instance of Agent
        trials - number of trials
    Output:
        rewards - list of rewards obtained this run. length = trials
    '''
    rewards = [] # Empty list of rewards
    agent.reset() # Reset agent to forget any older training
    for i in range(trials): # Loop through each trial
        state = env.reset() # Reset the environment to an empty state, then sample initial observations
        action = agent.act(state) # Compute the action to take
        reward = env.step(action) # Feed action to environment, get reward
        agent.learn(state,action,reward) # Perform any learning
        rewards.append(reward) # Add reward to list
    return rewards

def run_standard_regret(env,agent,trials,oracle):
    '''
    Perform one SSDP learning session with a single agent and no experts

    Input:
        env - instance of InfluenceDiagram representing environment
        agent - instance of Agent
        trials - number of trials
        oracle - instance of TruePolicyAgent
    Output:
        regret - regret accumulated this run
    '''
    regret = 0
    agent.reset() # Reset agent to forget any older training
    for i in range(trials): # Loop through each trial
        state = env.reset() # Reset the environment to an empty state, then sample initial observations
        action = agent.act(state) # Compute the action to take
        reward = env.step(action) # Feed action to environment, get reward
        agent.learn(state,action,reward) # Perform any learning
        oracle_action = oracle.act(state) # Get optimal action
        oracle_reward = oracle.expected_utility(state,oracle_action) # Get max reward
        regret += oracle_reward - reward # Calculate regret, add to running total
    return regret

def save_beta_param_test_to_csv(regrets,env_name,agents,panels,trials,runs,baseline="Baseline Agent",directory="beta_param_test"):
    base_dir = "results/"+env_name+"/"+directory+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
    file_dir = base_dir+"beta_param_test/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    f = open(file_dir+"regrets.csv","w",newline="")
    writer = csv.writer(f)
    # Save regrets
    for agent in agents:
        if agents[agent].takes_advice():
            for panel in panels:
                if agents[agent].get_history() is not None:
                    row = [agents[agent].name,panel.name,str(agents[agent].initial_estimate[0]),str(agents[agent].initial_estimate[1])]
                else:
                    row = [agents[agent].name,panel.name,"",""]
                writer.writerow(row+regrets[agent][panel])
        else:
            row = [agent,"","",""]
            writer.writerow(row+regrets[agent])
    f.close()

def save_expert_param_test_to_csv(regrets,env_name,agents,panel_dict,trials,runs,baseline="Baseline Agent",directory="expert_param_test"):
    base_dir = "results/"+env_name+"/"+directory+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
    file_dir = base_dir+"expert_param_test/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    f = open(file_dir+"regrets.csv","w",newline="")
    writer = csv.writer(f)
    # Save regrets
    for agent in agents:
        if agents[agent].takes_advice():
            for panel in panel_dict:
                for mu in regrets[agent][panel]:
                    for gamma in regrets[agent][panel][mu]:
                        row = [agents[agent].name,panel,mu,gamma]
                        writer.writerow(row+regrets[agent][panel][mu][gamma])
        else:
            row = [agent,"","",""]
            writer.writerow(row+regrets[agent])
    f.close()

def save_panel_comparison_to_csv(rewards,rhos,env_name,agents,panels,trials,runs,directory="panel_comparison"):
    base_dir = "results/"+env_name+"/"+directory+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
    file_dir = base_dir+"rewards/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    for agent in agents:
        if agents[agent].takes_advice():
            for panel in panels:
                f = open(file_dir+agent+"_"+panel.name+".csv","w",newline="")
                writer = csv.writer(f)
                writer.writerow([agent,panel.name])
                for run in range(runs):
                    writer.writerow(rewards[agent][panel.name][run,:])
                f.close()
        else:
            f = open(file_dir+agent+".csv","w",newline="")
            writer = csv.writer(f)
            writer.writerow([agent,None])
            for run in range(runs):
                writer.writerow(rewards[agent][run,:])
            f.close()
    print("Saving rhos to csv...")
    file_dir = base_dir+"rhos/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    for agent in agents:
        if agents[agent].get_history() is not None:
            for panel in panels:
                for expert in panel.experts:
                    f = open(file_dir+agent+"_"+panel.name+"_"+expert+".csv","w",newline="")
                    writer = csv.writer(f)
                    writer.writerow([agent,panel.name,expert])
                    for run in range(runs):
                        writer.writerow(rhos[agent][panel.name][expert][run,:])
                    f.close()

def save_panel_comparison_random_envs_to_csv(rewards,rhos,num_chance,num_decision,agent_list,panel_dict,trials,runs,directory):
    env_name = "Random ("+str(num_chance)+","+str(num_decision)+")"
    base_dir = "results/"+env_name+"/"+directory+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
    print("Saving rewards to csv...")
    file_dir = base_dir+"rewards/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    for agent in agent_list:
        if takes_advice[agent]:
            for panel in panel_dict:
                f = open(file_dir+agent+"_"+panel+".csv","w",newline="")
                writer = csv.writer(f)
                writer.writerow([agent,panel])
                for run in range(runs):
                    writer.writerow(rewards[agent][panel][run,:])
                f.close()
        else:
            f = open(file_dir+agent+".csv","w",newline="")
            writer = csv.writer(f)
            writer.writerow([agent,None])
            for run in range(runs):
                writer.writerow(rewards[agent][run,:])
            f.close()
    print("Saving rhos to csv...")
    file_dir = base_dir+"rhos/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    for agent in agent_list:
        if keeps_rho_history[agent]:
            for panel in panel_dict:
                for expert in panel_dict[panel]:
                    f = open(file_dir+agent+"_"+panel+"_"+str(expert)+".csv","w",newline="")
                    writer = csv.writer(f)
                    writer.writerow([agent,panel,expert])
                    for run in range(runs):
                        writer.writerow(rhos[agent][panel][str(expert)][run,:])
                    f.close()
