import CLUE
import numpy as np
import sys

experiment_name = "nonuniform"

'''
ENVIRONMENT
'''
# Create a random environment with 7 state variables (|S|=128)
# and 3 action variables (|A|=8)
reward_range = [-1,1]
env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3,reward_range=reward_range)


'''
EXPERIMENT DETAILS
'''
trials = 10000 # Number of trials each run
runs = 10 # Number of runs, each run the agent learns from scratch


'''
AGENTS
'''
agents = {
    "Baseline Agent":CLUE.BaselineAgent(env,trials),
    "NAF":CLUE.NaiveAdviceFollower(env,agent=CLUE.BaselineAgent(env,trials)),
    "CLUE":CLUE.ClueAgent(env,trials,agent=CLUE.BaselineAgent(env,trials))
}

'''
DIVIDE INTO REGIONS
'''
region_list = [0,1] # 0 = easy, 1 = hard
regions = CLUE.StateTable(env.state_space,random_init_vals=region_list)

'''
PANEL OF EXPERTS
'''
# Dictionary of panels to be tested
panel_dict = {
    "Single_Extreme":[[1,0]]
}

# Create list of panels
panels = CLUE.Experiment.make_panels_nonuniform(panel_dict,env,regions)

'''
RUN EXPERIMENT
'''
# Run the panel comparison experiment
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison(env,agents,panels,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
#CLUE.Experiment.save_panel_comparison_to_csv(rewards,rhos,env.name,agents,panels,trials,runs,directory=experiment_name)
'''
PLOT RESULTS
'''
'''
print("======Plotting graphs======")
# Set path in figures/ directory
base_path = env.name+"/"+experiment_name+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
# Set titles above each plot
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
# Plot graphs
CLUE.Plot.plot_reward_comparison(base_path,trials,panel_titles=panel_titles,reward_range=reward_range)
# Plot Rhos
#CLUE.Plot.plot_rhos(base_path,trials,panel_titles=panel_titles)
'''
