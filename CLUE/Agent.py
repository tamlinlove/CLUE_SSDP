class Agent():
    '''
    The parent class for all agents
    '''
    def __init__(self):
        '''
        Constructor - must initialise agent
        '''
        raise NotImplementedError()

    def act(self,state):
        '''
        Select an action based on the state

        Input:
            state - a dict mapping state variable names to values

        Output:
            action - a dict mapping action variable names to values
        '''
        raise NotImplementedError()

    def get_history(self):
        '''
        Return a dict of important stats acquired since the last reset
        By default, returns None (i.e. no important stats)
        Overwrite if you want to track something (e.g. rho values over time)

        Output:
            history - a dict of stats
        '''
        return None

    def learn(self,state,actions,reward):
        '''
        Do some learning given the state, action and reward of a trial
        By default, does nothing
        Overwrite if you want the agent to learn after each trial

        Input:
            state - dict mapping state node names to values
            action - dict mapping action node names to values
            reward - real-valued reward for the given state-action pair
        '''
        pass

    def reset(self):
        '''
        Reset the agent for a fresh run
        By default, does nothing
        Overwrite if you need to reset things between runs
        '''
        pass

    def takes_advice(self):
        '''
        Returns True if the agent incorporates expert advice, False otherwise
        By default, returns False
        Overwrite if you want the agent to use expert advice
        '''
        return False
