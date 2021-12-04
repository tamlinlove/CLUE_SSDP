import CLUE
import numpy as np
import sys

experiment_name = "baseline_experiment"

'''
ENVIRONMENT
'''
# Create a random environment with 7 state variables (|S|=128)
# and 3 action variables (|A|=8)
reward_range = [-1,1]
env = CLUE.make("RandomSSDP",num_chance=10,num_decision=3,reward_range=reward_range)


'''
EXPERIMENT DETAILS
'''
trials = 80000 # Number of trials each run
runs = 100 # Number of runs, each run the agent learns from scratch

if len(sys.argv) == 0:
    test = "Epsilon Greedy"
else:
    test = sys.argv[1]

'''
AGENTS
'''
# Create some agents, store in dictionary
agents = {}
if test == "Epsilon Greedy":
    experiment_name = "epsilon_greedy_test"
    agents["Baseline Agent"] = CLUE.BaselineAgent(env,trials)
    clue_base = CLUE.BaselineAgent(env,trials)
    naf_base = CLUE.BaselineAgent(env,trials)
elif test == "ETE":
    experiment_name = "ete_test"
    agents["ETE Baseline Agent"] = CLUE.ETEAgent(env,trials)
    clue_base = CLUE.ETEAgent(env,trials)
    naf_base = CLUE.ETEAgent(env,trials)
elif test == "Adaptive Greedy":
    experiment_name = "adaptive_greedy_test"
    agents["ETE Baseline Agent"] = CLUE.AdaptiveGreedyAgent(env,trials)
    clue_base = CLUE.AdaptiveGreedyAgent(env,trials)
    naf_base = CLUE.AdaptiveGreedyAgent(env,trials)
elif test == "UCB":
    experiment_name = "ucb_test"
    agents["UCB Baseline Agent"] = CLUE.UCBAgent(env,c=0.25)
    clue_base = CLUE.UCBAgent(env,c=0.25)
    naf_base = CLUE.UCBAgent(env,c=0.25)
else:
    error_message = "Test argument '{}' invalid. Must be 'Epsilon Greedy', 'ETE', 'Adaptive Greedy' or 'UCB'".format(test)
    raise Exception(error_message)
agents["CLUE"] = CLUE.ClueAgent(env,trials,agent=clue_base)
agents["NAF"] = CLUE.NaiveAdviceFollower(env,agent=naf_base)

#agents["Adaptive Greedy Baseline Agent"] = CLUE.AdaptiveGreedyAgent(env,trials)
#agents["ETE Baseline Agent"] = CLUE.ETEAgent(env,trials)
#agents["UCB Baseline Agent"] = CLUE.UCBAgent(env,c=0.25)
#agents["Baseline Agent"] = CLUE.BaselineAgent(env,trials)



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
# Run the panel comparison experiment
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison(env,agents,panels,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_panel_comparison_to_csv(rewards,rhos,env.name,agents,panels,trials,runs,directory=experiment_name)
'''
PLOT RESULTS
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
#CLUE.Plot.plot_reward_comparison(base_path,trials,panel_titles=panel_titles,reward_range=reward_range)
# Plot Rhos
#CLUE.Plot.plot_rhos(base_path,trials,panel_titles=panel_titles)
