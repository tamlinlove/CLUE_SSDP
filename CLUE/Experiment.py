'''
A bunch of methods for running and facilitating experiments
'''
import numpy as np
import os
import csv

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

def save_panel_comparison_to_csv(rewards,rhos,env,agents,panels,trials,runs,directory="panel_comparison"):
    base_dir = "results/"+env.name+"/"+directory+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
    print("Saving rewards to csv...")
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
