from NonuniformUnreliableExpert import NonuniformUnreliableExpert
from Panel import Panel

class NonuniformPanel(Panel):
    '''
    Class for a panel of potentially unreliable (nonuniform) experts
    '''
    def __init__(self,env,name,oracle,rhos,regions,mu=10,gamma=0.01,experts=None):
        '''
        Initialise panel of experts

        Input:
            env - instance of InfluenceDiagram representing environment
            name - a string. The name of the panel
            oracle - instance of TruePolicyAgent. Used to select correct advice
                Ignored if expert parameter is used
            rhos - list of rhos, each of which is a list of numbers between 0 and 1 inclusive. Each list represents an expert
                Ignored if experts parameter is used
            regions - instance of StateTable mapping state to region (a number, the index of the rho to be used)
            mu - interval parameter (number of trials between advice givings)
                default: 10
            gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
                default: 0.01
            experts - dict of experts in the panel. If None, will create Unreliable Experts using rho list

        Todo:
            allow for each expert in a panel to have different parameters
        '''
        self.name = name

        if experts is None:
            self.experts = {}
            expert_count = 0
            for rho_list in rhos:
                self.experts[str(expert_count)] = NonuniformUnreliableExpert(env,oracle,rho_list,regions,mu=mu,gamma=gamma)
        else:
            self.experts = experts