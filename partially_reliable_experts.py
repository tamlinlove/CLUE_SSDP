import CLUE
import numpy as np

'''
This script runs the panel comparison experiment with
partially reliable experts
'''

'''
EXPERIMENT DETAILS
'''
# Number of trials
trials = 10000
# Number of runs
runs = 10
# List of agents
agent_list = ["True Policy Agent","Baseline Agent","NAF","CLUE"]
# Expert nums
expert_nums = [0,1,3,5,7]
# Number of variables in env
num_chance = 7 # Number of state variables (|S|=2^num_chance)
num_decision = 3 # Number of action variables (|A|=2^num_decision)
# Name of experiment, for saving and plotting
exp_name = "partially_reliable_experts"
# Environment
env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3)
# Panels of single experts
panel_dict = {}
for num in expert_nums:
    panel_dict[str(num)] = [num]
# Multiple panels
panel_dict["Varied"] = [0,1,3,5,7]

'''
RUN EXPERIMENT
'''
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison_partially_reliable_experts(env,agent_list,panel_dict,trials,runs,display=True,display_interval=1)

print("======Saving results======")
CLUE.Experiment.save_panel_comparison_random_envs_to_csv(rewards,rhos,7,3,agent_list,panel_dict,trials,runs,directory=exp_name)

'''
PLOT RESULTS
'''
print("======Plotting graphs======")
# Set path in figures/ directory
base_path = env.name+"/partially_reliable_experts/"+str(trials)+"_trials_"+str(runs)+"_runs/"
# Set titles above each plot
panel_titles = {}
for num in expert_nums:
    panel_titles[str(num)] = "Number of hidden nodes: "+str(num)
#panel_titles["Varied"] = "Number of hidden nodes: 0,1,3,5,7"
# Set clip range for shaded area
reward_range = [-1,1]
# Plot graphs
CLUE.Plot.plot_panel_comparison(base_path,trials,panel_titles=panel_titles)
CLUE.Plot.plot_panel_comparison(base_path,trials,panel_titles=panel_titles,fill=False)
# Plot Rhos
#CLUE.Plot.plot_rhos_individual(base_path,trials,panel_titles=panel_titles)
