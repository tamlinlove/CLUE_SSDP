from UnreliableExpert import UnreliableExpert

class DegradingExpert(UnreliableExpert):
    '''
    Class for unreliable expert whose performance degrades over time
    '''
    def __init__(self,env,oracle,rho,degrade_factor=0.99,mu=10,gamma=0.1):
        '''
        Initialises a potentially unreliable expert whose performance degrades over time

        Input:
            env - instance of InfluenceDiagram representing environment
            oracle - instance of TruePolicyAgent, used to get correct advice
            rho - real number between 0 and 1 inclusive, controls how often correct advice is given
                rho = 0, always incorrect
                rho = 1, always correct
            degrade_factor - real number between 0 and 1 inclusive, controls how quickly rho degrades
                rho = rho * degrade_factor
                default: 0.99
            mu - interval parameter (number of trials between advice givings)
                default: 10
            gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
                default: 0.01
        '''
        super().__init__(env,oracle,rho,mu=mu,gamma=gamma)
        self.degrade_factor = degrade_factor
        self.initial_rho = rho

    def deliberate(self,state,action,reward):
        '''
        Decide whether or not to give advice for this trial

        Input:
            state - dict mapping node name to domain value of pre state chance nodes
            action - dict mapping decision node names to domain value
            reward - the reward obtained this trial

        Output:
            advice_given - True if advice is given, false otherwise
            advice - dict mapping decision node names to domain value if advice_given is True, None otherwise
        '''
        advice_given,advice = super().deliberate(state,action,reward)
        self.rho = self.rho * self.degrade_factor
        return advice_given,advice


    def reset(self):
        '''
        Resets the expert for a fresh run
        '''
        super().reset()
        #print("Rho: {} degraded to {}".format(self.initial_rho,self.rho))
        self.rho = self.initial_rho
