import CLUE

# Parameters
parameters = {
"agents":["True_Policy_Agent","Baseline_Agent","NAF","CLUE"], # The agents to be tested
"trials":80000, # The number of consecutive trials over which an agent learns
"runs":100, # The number of runs over which the reward curves etc. are averaged
"eps_start":1, # The starting value of the epsilon parameter
"eps_end":0, # The final value of epsilon
"eps_fraction":0.8, # The fraction of the total number of trials over which epsilon decays from initial to final value
"initial_estimate":0, # Initial value of Q(s,a) for each s,a
"moving_average_weight":"count_based", # Weight of moving average for learning, often denoted alpha. "count_based" = 1/k
"initial_beta":[1,1], # The initial [alpha,beta] values for estimating every expert's reliability
"prob_threshold":0.25, # T parameter, the threshold under which agent will never follow expert advice
"expert_interval":10, # mu, the minimum number of trials between advice givings
"expert_tolerance": 0.01, # gamma, the expert's tolerance for improvement
"display":True, # Whether or not the experiment will print out updates
"display_interval":10 # Number of runs between displays during experiment
}

env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3,seed=1)
