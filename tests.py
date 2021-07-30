import CLUE
import numpy as np
import matplotlib.pyplot as plt

env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3)
oracle = CLUE.TruePolicyAgent(env)

#hidden_nodes = []
#hidden_nodes = ["C0"]
#hidden_nodes = ["C1"]
hidden_nodes = ["C0","C2"]
#hidden_nodes = env.chance
expert = CLUE.PartiallyReliableExpert(env,hidden_nodes)

num_trials = 1000
num_runs = 100
optimality = []

for n in range(8):
    print("Testing partially reliable expert with "+str(n)+" hidden nodes")
    tot_num = 0
    for i in range(num_runs):
        hidden_nodes = np.random.choice(env.chance,n,replace=False)
        expert = CLUE.PartiallyReliableExpert(env,hidden_nodes)
        num_correct = 0
        for j in range(num_trials):
            state = env.reset()
            optimal = oracle.act(state)
            advised = expert.advise(state)
            if optimal == advised:
                num_correct += 1
        tot_num += num_correct/num_trials
    optimality.append(tot_num/num_runs)

plt.plot(range(8),optimality)
plt.title("Testing a partially reliable expert\n Averaged over 100 node configurations with 1000 trials each")
plt.ylabel("Percentage of states for which advice is optimal")
plt.xlabel("Number of state nodes hidden from expert")
plt.show()
