from Agent import Agent
import PGM

import numpy as np
import networkx as nx

class TruePolicyAgent(Agent):
    '''
    The class for an SSDP True Policy Agent
    i.e. an agent that always acts optimally
    '''

    def __init__(self,env):
        '''
        Initialise True Policy Agent

        Input:
            env - an instance of the InfluenceDiagram class, representing the
                  SSDP environment
        '''
        self.name = "True Policy Agent"
        self.env = env

        # Set up graph
        parents = {}
        children = {}
        for node in env.nodes+["reward"]:
            children[node] = []
            parents[node] = []
        # Add parents and children for each chance node and the reward node
        for factor in env.dn.factors:
            if not isinstance(factor,PGM.Utility): # Chance node
                for parent in factor.parents:
                    parents[factor.child.name].append(parent.name)
                    children[parent.name].append(factor.child.name)
            else: # Reward node
                for parent in factor.variables:
                    parents["reward"].append(parent.name)
                    children[parent.name].append("reward")
        # Add parents and children for each decision node
        for node in env.decision:
            for parent in env.variables[node].parents:
                parents[node].append(parent.name)
                children[parent.name].append(node)
        # Create graph
        graph = nx.DiGraph()
        edges = []
        for node in env.nodes:
            for child in children[node]:
                edges.append((node,child))
        graph.add_edges_from(edges)

        reward_ancestors = nx.ancestors(graph,'reward') # All ancestors of reward node
        order = self.getEliminationOrder(parents,children,reward_ancestors) # Elimination order for Variable Elimination
        ve = PGM.VE_DN(env.dn) # Variable Eliminator
        val,pol = ve.optimize(order,model=graph) # Get true policy
        self.policy = pol

        # Store policy in data structure
        self.decision_functions = {}
        for decision_function in self.policy:
            self.decision_functions[decision_function.dvar.name] = decision_function

    def act(self,state):
        '''
        Select an action based on state

        Input:
            state - dict mapping state variable names to values
        Output:
            action - dict mapping action variable names to values
        '''
        # Convert state to use variables as keys
        state_vars = {}
        for node in state:
            state_vars[self.env.variables[node]] = state[node]

        # Assign decision nodes
        action = {}
        running_state = state_vars.copy()
        for decision in self.env.action_space:
            decision_function = self.decision_functions[decision]
            action[decision] = decision_function.get_value(running_state)
            running_state[decision_function.dvar] = action[decision] # For inference
        return action

    def expected_utility(self,state,action):
        '''
        Compute the expected reward of a given state-action pair

        Input:
            state - dict mapping state variable names to values
            action - dict mapping action variable names to values

        Output:
            eu - the expected utility for state-action
        '''
        # Convert state and action to use variables as keys
        state_vars = {}
        for node in self.env.state_space:
            state_vars[self.env.variables[node]] = state[node]
        for node in action:
            state_vars[self.env.variables[node]] = action[node]

        # Create Variable eliminator
        gm_ve = PGM.VE(self.env.gm)
        gm_ve.max_display_level = 0 # Variable eliminator unaware of reward
        ve = PGM.VE(self.env.dn)
        ve.max_display_level = 0 # Variable eliminator used to get reward factor
        reward_fac = ve.gm.factors[-1] # Fetch reward factor
        reward_fac = ve.project_observations(reward_fac,state_vars) # Project observations
        # Calculate the probability of each remaining reward parent
        probs = {}
        for var in reward_fac.variables:
            probs[var] = gm_ve.query(var,obs=state_vars)
        # Calculate EU by Multiplying probability of each assignment by the reward
        eu = 0
        for i in range(reward_fac.size):
            assignment = reward_fac.index_to_assignment(i)
            assignment_prob = 1
            for var in probs:
                assignment_prob *= probs[var][assignment[var]]
            eu += assignment_prob * reward_fac.get_value(assignment)
        return eu

    def getEliminationOrder(self,parents,children,reward_ancestors):
        '''
        Order each node in the ID in order of elimination

        Input:
            parents - dict mapping node name to list of parent node names
            children - dict mapping node name to list of children node names
            reward_ancestors - list of node names that are ancestors to the 'reward' node

        Output:
            order - a list of node variables in order of elimination
        '''
        order = []
        nodes_left = self.env.nodes.copy()
        # Order decision nodes
        dec_nodes_left = self.env.decision.copy()
        dec_nodes = []
        while len(dec_nodes_left)>0:
            for dec_node in dec_nodes_left:
                can_add = True
                # Check if parents haven't been added yet
                for parent in parents[dec_node]:
                    if parent in dec_nodes_left:
                        can_add = False
                        break
                if can_add:
                    dec_nodes.append(dec_node)
                    dec_nodes_left.remove(dec_node)
        # Create visited dictionary
        added = {}
        for node in nodes_left:
            added[node] = False

        while len(nodes_left)>0:
            added_node = False
            if len(dec_nodes)>0:
                last_dec = dec_nodes[-1]
                for node in reversed(nodes_left):
                    can_add = True
                    for child in children[node]:
                        if child != "reward" and not added[child] and child==last_dec:
                            can_add = False
                            break
                    if can_add and (self.env.node_types[node]=="chance"):
                        order.append(self.env.variables[node])
                        nodes_left.remove(node)
                        added[node] = True
                        added_node = True
                if not added_node:
                    order.append(self.env.variables[last_dec])
                    nodes_left.remove(last_dec)
                    added[last_dec] = True
                    dec_nodes.pop()
            else:
                for node in reversed(nodes_left):
                    order.append(self.env.variables[node])
                    nodes_left.remove(node)
                    added[node] = True
        # Remove nodes that aren't reward ancestors
        for node in order:
            if node.name not in reward_ancestors:
                order.remove(node)
        return order
