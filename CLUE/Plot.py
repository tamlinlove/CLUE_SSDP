import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import statsmodels.api as sm # for smoothing
from mpl_toolkits.axes_grid1 import AxesGrid

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

def read_rewards(reward_path,trials,accepted_panels=None,reward_range=[None,None],display=True):
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
        display - a boolean, if True will print regular updates on plotting progress
            default - True
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
            if display:
                print("Processing "+filename)
            reward_list = []
            for row in reader:
                reward_list.append(row)
            f.close()
            rewards = np.array(reward_list).astype(float)
            y_mean = np.mean(rewards,axis=0)
            y_std = np.std(rewards,axis=0)
            print(filename+" std:"+str(y_std[-1]))
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

def read_rhos(rho_path,trials,accepted_panels=None):

    rhos = {}
    rho_low = {}
    rho_high = {}
    agents = []
    panels = []
    rels = {}

    files = os.listdir(rho_path)
    # Read and smooth
    for filename in files:
        f = open(rho_path+filename,"r",newline="")
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        agent = header[0]
        panel = header[1]
        rel = header[2]
        if accepted_panels is None or panel in accepted_panels or panel == "":
            rho_list = []
            for row in reader:
                rho_list.append(row)
            f.close()
            rho = np.array(rho_list).astype(float)
            y_mean = np.mean(rho,axis=0)
            y_std = np.std(rho,axis=0)
            y,low,high = smooth(y_mean,y_std,trials)
            if panel not in panels:
                panels.append(panel)
                rels[panel] = []
            if agent not in agents:
                rhos[agent] = {}
                rho_low[agent] = {}
                rho_high[agent] = {}
                agents.append(agent)
            if rel not in rels[panel]:
                rels[panel].append(rel)
            if panel not in rhos[agent]:
                rhos[agent][panel] = {}
                rho_low[agent][panel] = {}
                rho_high[agent][panel] = {}
            rhos[agent][panel][str(rel)] = np.array(y,dtype=float)
            rho_low[agent][panel][str(rel)] = np.array(low,dtype=float)
            rho_high[agent][panel][str(rel)] = np.array(high,dtype=float)
        else:
            f.close()

    return rhos,rho_low,rho_high,agents,panels,rels

def plot_beta_heatmap(base_path,alphas,betas,panel_titles,baseline="Baseline Agent",rounding=1,vrange=None):
    # Read in regrets
    f = open("results/"+base_path+"regrets.csv","r",newline="")
    reader = csv.reader(f, delimiter=',')

    regrets = {}
    takes_advice = {}
    has_beta = {}
    baseline_regret = 0
    baseline_regret_matrix = np.zeros((len(alphas),len(betas)))
    for line in reader:
        agent = line[0]
        panel = line[1]
        a = line[2]
        b = line[3]
        mean_regret = np.mean(np.array(line[4:],dtype=float))
        if agent not in regrets: # First time seeing this agent
            if panel == "": # No panel
                takes_advice[agent] = False
                has_beta[agent] = False
                regrets[agent] = mean_regret
                if agent == baseline:
                    baseline_regret = mean_regret
                    baseline_regret_matrix = np.ones((len(alphas),len(betas)))*baseline_regret
            else: # Uses panel
                takes_advice[agent] = True
                regrets[agent] = {}
                if a == "": # Doesn't have beta distribution
                    has_beta[agent] = False
                    regrets[agent][panel] = mean_regret
                else: # Uses beta distribution
                    has_beta[agent] = True
                    regrets[agent][panel] = np.zeros((len(alphas),len(betas)))
                    regrets[agent][panel][alphas.index(float(a))][betas.index(float(b))] = mean_regret
        else: # Agent already in dict
            if a == "": # Doesn't have beta distribution
                regrets[agent][panel] = mean_regret
            else: # Uses beta distribution
                if panel in regrets[agent]: # Panel already seen before
                    regrets[agent][panel][alphas.index(float(a))][betas.index(float(b))] = mean_regret
                else: # Never encountered agent-panel pair
                    regrets[agent][panel] = np.zeros((len(alphas),len(betas)))
                    regrets[agent][panel][alphas.index(float(a))][betas.index(float(b))] = mean_regret
    f.close()

    # Calculate differences
    regret_vals = {}
    regret_diffs = {}
    for agent in regrets:
        if takes_advice[agent]:
            if has_beta[agent]:
                regret_vals[agent] = {}
                regret_diffs[agent] = {}
            for panel in regrets[agent]:
                if has_beta[agent]:
                    print(agent+" - "+panel+" : "+str(regrets[agent][panel]))
                    regret_vals[agent][panel] = regrets[agent][panel]
                    regret_diffs[agent][panel] = regrets[agent][panel] - baseline_regret_matrix
                else:
                    print(agent+" - "+panel+" : "+str(regrets[agent][panel]))
        else:
            print(agent+" : "+str(regrets[agent]))

    # Plot
    fig_path = "figures/"+base_path
    for agent in regret_vals:
        plot_heatmap(fig_path,agent+"_regrets",panel_titles,regret_vals[agent],"$\\alpha$",alphas,"$\\beta$",betas,rounding=rounding,vrange=vrange)
        plot_heatmap(fig_path,agent+"_regret_diffs",panel_titles,regret_diffs[agent],"$\\alpha$",alphas,"$\\beta$",betas,rounding=rounding,vrange=vrange)

def plot_expert_heatmap(base_path,mus,gammas,panel_titles,baseline="Baseline Agent",rounding=1,vrange=None):
    # Read in regrets
    f = open("results/"+base_path+"regrets.csv","r",newline="")
    reader = csv.reader(f, delimiter=',')

    # Make sure mus and gammas are floats
    mus = list(np.array(mus,dtype=float))
    gammas = list(np.array(gammas,dtype=float))

    regrets = {}
    takes_advice = {}
    baseline_regret = 0
    baseline_regret_matrix = np.zeros((len(mus),len(gammas)))
    for line in reader:
        agent = line[0]
        panel = line[1]
        mu = line[2]
        gamma = line[3]
        mean_regret = np.mean(np.array(line[4:],dtype=float))
        if agent not in regrets: # First time seeing this agent
            if panel == "": # No panel
                takes_advice[agent] = False
                regrets[agent] = mean_regret
                if agent == baseline:
                    baseline_regret = mean_regret
                    baseline_regret_matrix = np.ones((len(mus),len(gammas)))*baseline_regret
            else: # Uses panel
                takes_advice[agent] = True
                regrets[agent] = {}
                regrets[agent][panel] = np.zeros((len(mus),len(gammas)))
                regrets[agent][panel][mus.index(float(mu))][gammas.index(float(gamma))] = mean_regret
        else: # Agent already in dict
            if panel in regrets[agent]: # Panel already seen before
                regrets[agent][panel][mus.index(float(mu))][gammas.index(float(gamma))] = mean_regret
            else: # Never encountered agent-panel pair
                regrets[agent][panel] = np.zeros((len(mus),len(gammas)))
                regrets[agent][panel][mus.index(float(mu))][gammas.index(float(gamma))] = mean_regret
    f.close()

    # Calculate differences
    regret_vals = {}
    regret_diffs = {}
    for agent in regrets:
        if takes_advice[agent]:
            regret_vals[agent] = {}
            regret_diffs[agent] = {}
            for panel in regrets[agent]:
                regret_vals[agent][panel] = regrets[agent][panel]
                regret_diffs[agent][panel] = regrets[agent][panel] - baseline_regret_matrix
        else:
            print(agent+" : "+str(regrets[agent]))

    # Plot
    fig_path = "figures/"+base_path
    for agent in regret_vals:
        plot_heatmap(fig_path,agent+"_regrets",panel_titles,regret_vals[agent],"$\\mu$",mus,"$\\gamma$",gammas,rounding=rounding,vrange=vrange)
        plot_heatmap(fig_path,agent+"_regret_diffs",panel_titles,regret_diffs[agent],"$\\mu$",mus,"$\\gamma$",gammas,rounding=rounding,vrange=vrange)

def plot_heatmap(fig_path,filename,panel_titles,vals,x_label,x_vals,y_label,y_vals,rounding=1,rev=True,vrange=None):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    panels = list(panel_titles.keys())
    if vrange is None:
        mins = []
        maxs = []
        for panel in panels:
            mins.append(np.min(vals[panel]))
            maxs.append(np.max(vals[panel]))
        vmin = np.min(np.array(mins))
        vmax = np.max(np.array(maxs))
        print(filename+": ("+str(vmin)+","+str(vmax)+")")
    else:
        vmin = vrange[0]
        vmax = vrange[1]
    fig = plt.figure(figsize=(4*len(panels), 4))
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, len(panels)),
                axes_pad=0.05,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                )
    if rev:
        cmap = "RdYlGn_r"
    else:
        cmap = "RdYlGn"
    for i in range(len(panels)):
        ax = grid[i]
        panel = panels[i]
        val = vals[panel]
        im = ax.imshow(val,vmin=vmin,vmax=vmax,origin="lower",cmap=cmap)
        ax.set_title(panel_titles[panel])
        ax.set_xlabel(x_label)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_ylabel(y_label)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels(y_vals)
        for (j,k),label in np.ndenumerate(np.round(val,rounding)):
            ax.text(k,j,label,ha='center',va='center',fontweight="bold")
    grid.cbar_axes[0].colorbar(im)
    plt.savefig(fig_path+filename)
    plt.close()

def plot_reward_comparison_individual(base_path,trials,accepted_panels=None,panel_titles=None,reward_range=[None,None]):
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
    for i in range(len(panels)):
        panel = accepted_panels[i]
        fig, ax = plt.subplots(ncols=1,figsize=(5,4))
        for agent in agents:
            agent_name = agent_names[agent]
            if takes_advice[agent]:
                ax.fill_between(x, reward_low[agent][panel], reward_high[agent][panel], alpha=0.2)
                ax.plot(x,reward_means[agent][panel],label=agent_name)
            else:
                ax.fill_between(x, reward_low[agent], reward_high[agent], alpha=0.2)
                ax.plot(x,reward_means[agent],label=agent_name)
            ax.set_xlabel("Trials")
            ax.set_ylabel("Average Reward")
        if panel_titles is not None:
            ax.set_title(panel_titles[panel])
        else:
            ax.set_title(panel)
        bottom = 0.5
        wspace = 0.45
        fig.subplots_adjust(bottom=bottom, wspace=wspace)
        plt.legend(labels=agent_labels,loc='upper left',
                 bbox_to_anchor=(0, -bottom/2),fancybox=False, shadow=False, ncol=len(agents))
        plt.savefig(file_dir+panel+"_reward_comparison.png",dpi=256)
        plt.close()

def plot_reward_comparison(base_path,trials,accepted_panels=None,panel_titles=None,reward_range=[None,None],fill=True):
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

    if len(panels) == 0:
        panels = ["No Expert"]
        panel_titles["No Expert"] = "No Expert"
        accepted_panels.append("No Expert")

    # Plot
    x = np.arange(trials)
    file_dir = fig_path+"agent_comparison/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    fig, ax = plt.subplots(ncols=len(panels),figsize=(8*len(panels),4.8))
    if len(panels)==1:
        ax = [ax]
    plot_list = []
    for i in range(len(panels)):
        panel = accepted_panels[i]
        for agent in agents:
            agent_name = agent_names[agent]
            if takes_advice[agent]:
                if fill:
                    ax[i].fill_between(x, reward_low[agent][panel], reward_high[agent][panel], alpha=0.2)
                l, = ax[i].plot(x,reward_means[agent][panel],label=agent_name)
            else:
                if fill:
                    ax[i].fill_between(x, reward_low[agent], reward_high[agent], alpha=0.2)
                l, = ax[i].plot(x,reward_means[agent],label=agent_name)
            plot_list.append(l)
            ax[i].set_xlabel("Trials")
            ax[i].set_ylabel("Average Reward")
            if panel_titles is not None:
                ax[i].set_title(panel_titles[panel])
            else:
                ax[i].set_title(panel)
    if len(panels) == 1:
        wspace = -0.5
    else:
        wspace=0.45
    bottom = 0.5

    fig.subplots_adjust(bottom=bottom, wspace=wspace)
    plt.legend(handles = plot_list , labels=agent_labels,loc='upper center',
             bbox_to_anchor=(-wspace, -bottom/2),fancybox=False, shadow=False, ncol=len(agents))
    fname = file_dir+"reward_comparison"
    if not fill:
        fname += "_nofill"
    fname += ".png"
    plt.savefig(fname,dpi=256)
    plt.close()

def plot_rhos(base_path,trials,accepted_panels=None,panel_titles=None):
    path = "results/"+base_path
    fig_path = "figures/"+base_path
    rho_path = path+"rhos/"

    rhos,rho_low,rho_high,agents,panels,rels = read_rhos(rho_path,trials,accepted_panels)

    if accepted_panels is None:
        accepted_panels = panels

    num_experts = []
    all_rels = []
    for panel in rels:
        num_experts.append(len(rels[panel]))
        for rel in rels[panel]:
            if float(rel) not in all_rels:
                all_rels.append(float(rel))
    max_num_rel = np.max(num_experts)
    all_rels.sort()
    all_rels_str = list(map(str,all_rels))

    # Correct agent names
    agent_names = {}
    agent_labels = []
    for agent in agents:
        agent_names[agent] = agent.replace("_"," ")
        agent_labels.append(agent_names[agent])

    # Plot
    x = np.arange(trials)
    file_dir = fig_path+"rho_comparison/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)

    c_map = plt.cm.get_cmap("tab20", len(all_rels))
    colours = {}
    for rel in all_rels_str:
        colours[rel] = c_map(all_rels_str.index(rel))

    for agent in agents:
        fig, ax = plt.subplots(ncols=len(panels),figsize=(4*len(panels),4.8))
        plot_dict = {}
        for i in range(len(panels)):
            panel = panels[i]
            for rel in rels[panel]:
                rel_str = str(float(rel))
                print(agent+" : "+panel+" : "+rel_str+" : converged to "+str(rhos[agent][panel][rel][-1]))
                ax[i].fill_between(x,rho_low[agent][panel][rel], rho_high[agent][panel][rel], alpha=0.2,color=colours[rel_str])
                l, = ax[i].plot(rhos[agent][panel][rel],label=rel_str,color=colours[rel_str])
                if rel_str not in plot_dict:
                    plot_dict[rel_str] = l
                ax[i].set_xlabel("Trials")
                ax[i].set_ylabel("Average Rho")
                if panel_titles is not None:
                    ax[i].set_title(panel_titles[panel])
                else:
                    ax[i].set_title(panel)
        plot_list = []
        for rel in all_rels_str:
            plot_list.append(plot_dict[rel])
        bottom = 0.5
        wspace = 0.45
        fig.subplots_adjust(bottom=bottom, wspace=wspace)
        plt.legend(handles = plot_list , labels=all_rels_str,loc='upper center',
                 bbox_to_anchor=(-wspace, -bottom/2),fancybox=False, shadow=False, ncol=max_num_rel)
        plt.savefig(file_dir+agent+".png",dpi=256)
        plt.close()

def plot_rhos_individual(base_path,trials,accepted_panels=None,panel_titles=None):
    path = "results/"+base_path
    fig_path = "figures/"+base_path
    rho_path = path+"rhos/"

    rhos,rho_low,rho_high,agents,panels,rels = read_rhos(rho_path,trials,accepted_panels)

    if accepted_panels is None:
        accepted_panels = panels

    num_experts = []
    all_rels = []
    for panel in rels:
        num_experts.append(len(rels[panel]))
        for rel in rels[panel]:
            if float(rel) not in all_rels:
                all_rels.append(float(rel))
    max_num_rel = np.max(num_experts)
    all_rels.sort()
    all_rels_str = list(map(str,all_rels))

    # Correct agent names
    agent_names = {}
    agent_labels = []
    for agent in agents:
        agent_names[agent] = agent.replace("_"," ")
        agent_labels.append(agent_names[agent])

    # Plot
    x = np.arange(trials)
    file_dir = fig_path+"rho_comparison/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)

    plt.rcParams["figure.figsize"] = (4,4.8)

    c_map = plt.cm.get_cmap("tab20", len(all_rels))
    colours = {}
    for rel in all_rels_str:
        colours[rel] = c_map(all_rels_str.index(rel))
    bottom = 0.5
    wspace = 0.45

    for agent in agents:
        for panel in panels:
            for rel in rels[panel]:
                rel_str = str(float(rel))
                plt.fill_between(x,rho_low[agent][panel][rel], rho_high[agent][panel][rel], alpha=0.2,color=colours[rel_str])
                plt.plot(rhos[agent][panel][rel],label=rel_str,color=colours[rel_str])
                plt.xlabel("Trials")
                plt.ylabel("Average Rho")
                if panel_titles is not None:
                    plt.title(panel_titles[panel])
                else:
                    plt.title(panel)
            plt.legend(labels=agent_labels,loc='upper center',
                     bbox_to_anchor=(-wspace, -bottom/2),fancybox=False, shadow=False, ncol=len(agents))
            plt.savefig(file_dir+agent+"_"+panel+".png",dpi=256)
            plt.close()

def plot_panel_comparison(base_path,trials,accepted_panels=None,panel_titles=None,reward_range=[None,None],fill=True):
    path = "results/"+base_path
    fig_path = "figures/"+base_path
    reward_path = path+"rewards/"

    reward_means,reward_low,reward_high,takes_advice,agents,panels = read_rewards(reward_path,trials,accepted_panels,reward_range=reward_range)

    if accepted_panels is None:
        accepted_panels = panels

    # Correct agent names


    # Plot
    x = np.arange(trials)
    file_dir = fig_path+"panel_comparison/"

    #print(reward_means)

    advice_agents = []
    for agent in agents:
        #print(agent)
        if takes_advice[agent]:
            advice_agents.append(agent)

    fig, ax = plt.subplots(ncols=len(advice_agents),figsize=(4*len(advice_agents),4.8))
    plot_list = []

    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    for j in range(len(advice_agents)):
        agent = advice_agents[j]
        for i in range(len(panels)):
            panel = accepted_panels[i]
            if fill:
                ax[j].fill_between(x, reward_low[agent][panel], reward_high[agent][panel], alpha=0.2)
            l, = ax[j].plot(x,reward_means[agent][panel],label=panel)
            plot_list.append(l)
        if fill:
            ax[j].fill_between(x, reward_low["Baseline Agent"], reward_high["Baseline Agent"], alpha=0.2)
        l, = ax[j].plot(x,reward_means["Baseline Agent"],label="Baseline Agent")
        plot_list.append(l)
        ax[j].set_xlabel("Trials")
        ax[j].set_ylabel("Average Reward")
        ax[j].set_title(agent)
    bottom = 0.5
    wspace = 0.45
    fig.subplots_adjust(bottom=bottom, wspace=wspace)
    plt.legend(labels=panels+["Baseline"],loc='upper center',
             bbox_to_anchor=(-wspace, -bottom/2),fancybox=False, shadow=False, ncol=len(panels+["Baseline"]))
    name = file_dir+"panel_comparison"
    if not fill:
        name += "_nofill"
    name += ".png"
    plt.savefig(name,dpi=256)
    plt.close()
