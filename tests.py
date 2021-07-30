import CLUE
import numpy as np

env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3)

env.prune_state_nodes([])
#env.prune_state_nodes(['C0'])
#env.prune_state_nodes(['C0','C1','C3'])
#env.prune_state_nodes(env.chance)
