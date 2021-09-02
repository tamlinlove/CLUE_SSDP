import CLUE
import numpy as np

'''
This script runs a comparison of regret between agents with different initial beta
distribution parameters
'''

'''
ENVIRONMENT
'''
# Create a random environment with 7 state variables (|S|=128)
# and 3 action variables (|A|=8)
env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3)

'''
EXPERIMENT DETAILS
'''
trials = 10000 # Number of trials each run
runs = 2 # Number of runs, each run the agent learns from scratch
mus = [1,10,100,1000] # Values of mu to be tested
gammas = [0.001,0.01,0.1,1] # Values of gamma to be tested

'''
AGENTS
'''
# Create some agents
agent_list = ["True Policy Agent","Baseline Agent","NAF","CLUE","Decayed Reliance"]
agents = CLUE.Experiment.make_agents(agent_list,env,trials)

'''
PANEL OF EXPERTS
'''
# Dictionary of panels to be tested
panel_dict = {
"Single_Bad":[0],
"Single_Good":[1],
"Varied_Panel":[0,0.1,0.25,0.5,0.75,0.9,1]
}

'''
RUN EXPERIMENT
'''
# Run experiments, calculate regret
print("======Running experiment======")
regrets = CLUE.Experiment.expert_param_test(env,agents,panel_dict,mus,gammas,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_expert_param_test_to_csv(regrets,env.name,agents,panel_dict,trials,runs)
'''
PLOT RESULTS
'''
# Set path in figures/ directory
base_path = env.name+"/expert_param_test/"+str(trials)+"_trials_"+str(runs)+"_runs/expert_param_test/"
# Set titles above each plot
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
print("======Plotting graphs======")
CLUE.Plot.plot_expert_heatmap(base_path,mus,gammas,panel_titles)
