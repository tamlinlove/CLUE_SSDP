from UnreliableExpert import UnreliableExpert

class Panel:
    '''
    Class for a panel of potentially unreliable experts
    '''
    def __init__(self,env,name,oracle,rhos,mu=10,gamma=0.01):
        '''
        Initialise panel of experts

        Input:
            env - instance of InfluenceDiagram representing environment
            name - a string. The name of the panel
            oracle - instance of TruePolicyAgent. Used to select correct advice
            rhos - list of numbers between 0 and 1 inclusive. Each represents an expert
            mu - interval parameter (number of trials between advice givings)
                default: 10
            gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
                default: 0.01

        Todo:
            allow for each expert in a panel to have different parameters
        '''
        self.name = name
        self.experts = {}
        for rho in rhos:
            self.experts[str(rho)] = UnreliableExpert(env,oracle,rho,mu=mu,gamma=gamma)

    def advise(self,state,action,reward):
        '''
        Asks every expert to decide whether or not to advise the agent, and if so, returns that advice

        Input:
            state - dict mapping state node name to domain value
            action - dict mapping action node names to domain value
            reward - the reward obtained this trial
        Output:
            advice - dict mapping expert name (string of true rho) to the advice they give, which is either:
                    1. a dict mapping decision node name to value
                    2. None
        '''
        advice = {}
        for expert in self.experts:
            _,advice[expert] = self.experts[expert].deliberate(state,action,reward)
        return advice

    def reset(self):
        '''
        Reset every expert in the panel for a fresh run
        '''
        for expert in self.experts:
            self.experts[expert].reset()
