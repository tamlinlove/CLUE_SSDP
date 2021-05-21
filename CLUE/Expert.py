class Expert:
    '''
    Parent class for an expert
    '''
    def __init__(self):
        '''
        Initialises an unreliable expert
        '''
        raise NotImplementedError()

    def advise(self,state):
        '''
        Advise on the best action to take given an assignment of state variables

        Input:
            state - dict mapping node name to domain value

        Output:
            advice - dict mapping action node names to domain value
        '''
        raise NotImplementedError()

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
        raise NotImplementedError()

    def reset(self):
        '''
        Resets the expert for a fresh run
        '''
        pass
