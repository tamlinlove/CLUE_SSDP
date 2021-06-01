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
runs = 10 # Number of runs, each run the agent learns from scratch
alphas = [1,10,100,1000] # Values of alpha to be tested
betas = [1,10,100,1000] # Values of beta to be tested

'''
AGENTS
'''
# Create some agents, do not include CLUE
agent_list = ["True Policy Agent","Baseline Agent","NAF"]
agents = CLUE.Experiment.make_agents(agent_list,env,trials)
# Create different CLUE agents
for a in alphas:
    for b in betas:
        agents["CLUE_"+str(a)+"_"+str(b)] = CLUE.ClueAgent(env,trials=trials,initial_estimate=[a,b])

'''
PANEL OF EXPERTS
'''
# Dictionary of panels to be tested
panel_dict = {
"Single_Bad":[0],
"Single_Good":[1],
"Varied_Panel":[0,0.1,0.25,0.5,0.75,0.9,1]
}

# Create list of panels
panels = CLUE.Experiment.make_panels(panel_dict,env)

'''
RUN EXPERIMENT
'''
# Run experiments, calculate regret
print("======Running experiment======")
regrets = CLUE.Experiment.regret_test(env,agents,panels,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_beta_param_test_to_csv(regrets,env.name,agents,panels,trials,runs)
'''
PLOT RESULTS
'''
# Set path in figures/ directory
base_path = env.name+"/beta_param_test/"+str(trials)+"_trials_"+str(runs)+"_runs/beta_param_test/"
# Set titles above each plot
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
print("======Plotting graphs======")
CLUE.Plot.plot_beta_heatmap(base_path,alphas,betas,panel_titles)
