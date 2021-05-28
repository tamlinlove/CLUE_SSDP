import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import statsmodels.api as sm # for smoothing

# For Latex
plt.rcParams.update({
    "text.usetex": True
    })

def smooth(y_mean,y_std,trials,reward_range=[None,None]):
    '''
    Smooth a mean curve and add boundaries for shading

    Input:
        y_mean - mean curve, an array of numbers
        y_std - standard deviation, an array of numbers
        trials - number of trials
        reward_range - a list [min,max], where min is the minimum value and max the max value
            a value of None corresponds to no bound
            shaded areas will not go above and below these values
            default - [None,None]
    Output:
        y - the smoothed curve, an array of numbers
        low - the lower bound of the shaded region
        high - the upper bound of the shaded region
    '''
    x = np.arange(trials)
    if reward_range[0] is None:
        low = y_mean - y_std
    else:
        low = np.clip(y_mean-y_std,reward_range[0],None)
    if reward_range[1] is None:
        high = y_mean + y_std
    else:
        high = np.clip(y_mean+y_std,None,reward_range[1])
    lowess = sm.nonparametric.lowess
    y = lowess(y_mean, x, frac= 0.1, it=0, return_sorted=False)
    low = lowess(low, x, frac= 0.1, it=0, return_sorted=False)
    high = lowess(high, x, frac= 0.1, it=0, return_sorted=False)
    return y,low,high

def read_rewards(reward_path,trials,accepted_panels=None,reward_range=[None,None]):
    '''
    Read reward csvs into arrays

    Input:
        reward_path - path to directory with reward csvs
        trials - number of trials
        accepted_panels - list of panels to be read
            None if all panels are to be accepted
            default - None
        reward_range - a list [min,max], where min is the minimum value and max the max value
            a value of None corresponds to no bound
            shaded areas will not go above and below these values
            default - [None,None]
    Output:
        reward_means - dict mapping agent to smoothed reward curve
            reward_means[agent] = array of rewards
            reward_means[agent][panel] = array of rewards
        reward_low - dict mapping agent to lower bound of shaded area
            reward_means[agent] = array of rewards
            reward_means[agent][panel] = array of rewards
        reward_high - dict mapping agent to upper bound of shaded area
            reward_means[agent] = array of rewards
            reward_means[agent][panel] = array of rewards
        takes_advice - dict mapping agent to either True or False
            takes_advice[agent] = True if agent takes advice from panel, False otherwise
        agents - list of agent names
        panels - list of panel names
    '''
    reward_means = {}
    reward_low = {}
    reward_high = {}
    takes_advice = {}
    agents = []
    panels = []

    files = os.listdir(reward_path)
    # Read and smooth
    for filename in files:
        f = open(reward_path+filename,"r",newline="")
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        agent = header[0]
        panel = header[1]
        if accepted_panels is None or panel in accepted_panels or panel == "":
            reward_list = []
            for row in reader:
                reward_list.append(row)
            f.close()
            rewards = np.array(reward_list).astype(float)
            y_mean = np.mean(rewards,axis=0)
            y_std = np.std(rewards,axis=0)
            y,low,high = smooth(y_mean,y_std,trials,reward_range=reward_range)
            if panel=="": # doesn't take advice
                reward_means[agent] = np.array(y,dtype=float)
                reward_low[agent] = np.array(low,dtype=float)
                reward_high[agent] = np.array(high,dtype=float)
                takes_advice[agent] = False
                agents.append(agent)
            else:
                if panel not in panels:
                    panels.append(panel)
                if agent not in agents:
                    reward_means[agent] = {}
                    reward_low[agent] = {}
                    reward_high[agent] = {}
                    takes_advice[agent] = True
                    agents.append(agent)
                reward_means[agent][panel] = np.array(y,dtype=float)
                reward_low[agent][panel] = np.array(low,dtype=float)
                reward_high[agent][panel] = np.array(high,dtype=float)
        else:
            f.close()

    return reward_means,reward_low,reward_high,takes_advice,agents,panels

def plot_reward_comparison(base_path,trials,accepted_panels=None,panel_titles=None,reward_range=[None,None]):
    '''
    Plot a comparison of rewards

    Input:
        base_path - path to directory which contains rewards directory
        trials - number of trials
        accepted_panels - list of panels to include in plot
            if None, will include all
            default - None
        panel_titles - dict mapping panel name to title of plot
            if None, will just use panel name
            default - None
        reward_range - a list [min,max], where min is the minimum value and max the max value
            a value of None corresponds to no bound
            shaded areas will not go above and below these values
            default - [None,None]
    '''
    path = "results/"+base_path
    fig_path = "figures/"+base_path
    reward_path = path+"rewards/"

    reward_means,reward_low,reward_high,takes_advice,agents,panels = read_rewards(reward_path,trials,accepted_panels,reward_range=reward_range)

    if accepted_panels is None:
        accepted_panels = panels

    # Correct agent names
    agent_names = {}
    agent_labels = []
    for agent in agents:
        agent_names[agent] = agent.replace("_"," ")
        agent_labels.append(agent_names[agent])

    # Plot
    x = np.arange(trials)
    file_dir = fig_path+"agent_comparison/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    fig, ax = plt.subplots(ncols=3,figsize=(12,4.8))
    plot_list = []
    for i in range(len(panels)):
        panel = accepted_panels[i]
        for agent in agents:
            agent_name = agent_names[agent]
            if takes_advice[agent]:
                ax[i].fill_between(x, reward_low[agent][panel], reward_high[agent][panel], alpha=0.2)
                l, = ax[i].plot(x,reward_means[agent][panel],label=agent_name)
            else:
                ax[i].fill_between(x, reward_low[agent], reward_high[agent], alpha=0.2)
                l, = ax[i].plot(x,reward_means[agent],label=agent_name)
            plot_list.append(l)
            ax[i].set_xlabel("Trials")
            ax[i].set_ylabel("Average Reward")
            if panel_titles is not None:
                ax[i].set_title(panel_titles[panel])
            else:
                ax[i].set_title(panel)
    bottom = 0.5
    wspace = 0.45
    fig.subplots_adjust(bottom=bottom, wspace=wspace)
    plt.legend(handles = plot_list , labels=agent_labels,loc='upper center',
             bbox_to_anchor=(-wspace, -bottom/2),fancybox=False, shadow=False, ncol=len(agents))
    plt.savefig(file_dir+"reward_comparison.png",dpi=256)
    plt.close()