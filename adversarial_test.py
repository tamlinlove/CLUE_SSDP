import CLUE
import numpy as np

'''
This script runs the adversarial test
'''

'''
EXPERIMENT DETAILS
'''
# Number of trials
trials = 80000
#trials = 10000
# Number of runs
runs = 100
#runs = 10
# List of agents
agent_list = ["True Policy Agent","Baseline Agent","NAF","CLUE"]

# Dict of panels
panel_dict = {
"Single_Bad":[0],
"Single_Good":[1],
"Varied_Panel":[0,0.1,0.25,0.5,0.75,0.9,1]
}

# Number of variables in env
num_chance = 10 # Number of state variables (|S|=2^num_chance)
num_decision = 1 # Number of action variables (|A|=2^num_decision)
# Name of experiment, for saving and plotting
exp_name = "adversarial"

'''
RUN EXPERIMENT
'''
# Run the panel comparison experiment
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison_random_envs(agent_list,panel_dict,trials,runs,display=True,display_interval=2,num_chance=num_chance,num_decision=num_decision,threshold=0.25)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_panel_comparison_random_envs_to_csv(rewards,rhos,num_chance,num_decision,agent_list,panel_dict,trials,runs,directory=exp_name)
'''
PLOT RESULTS
'''
print("======Plotting graphs======")
# Set path in figures/ directory
base_path = "Random ("+str(num_chance)+","+str(num_decision)+")/"+exp_name+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
# Set titles above each plot
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
# Set clip range for shaded area
reward_range = [-1,1]
# Plot graphs
CLUE.Plot.plot_reward_comparison(base_path,trials,panel_titles=panel_titles)
# Plot Rhos
CLUE.Plot.plot_rhos(base_path,trials,panel_titles=panel_titles)
