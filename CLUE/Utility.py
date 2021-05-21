import numpy as np

class Utility:
    '''
    Class for utility tables
    Based on the Factor class by David L Poole and Alan K Mackworth, 2017
    '''
    def __init__(self,domains,Q0=0,alpha=None):
        '''
        Initialise utility table

        Input:
            domains - dict mapping node names to their domains
            Q0 - initial Q value. Can be real number or array of appropriate size
            alpha - learning rate. Real number or None
                if None, will be based on visit count (see Sutton and Barto, 2018)
        '''
        self.domains = domains
        self.nodes = list(self.domains.keys())
        # Set up offsets and size
        self.size = 1
        self.offsets = {}
        for i in range(len(self.nodes)-1,-1,-1):
            self.offsets[self.nodes[i]]=self.size
            self.size *= len(self.domains[self.nodes[i]]) # Offset by size of domain

        # Set up initial values
        if isinstance(Q0,(int,float)): # is a single number
            self.values = np.ones(self.size) * Q0
        elif isinstance(Q0,(list,np.ndarray)): # is an array-like
            if isinstance(Q0,list):
                Q0_size = len(Q0)
            else:
                Q0_size = Q0.size
            if Q0_size != self.size:
                error_message = "Size of Q0 does not match state-action space size ("+str(Q0_size)+" != "+str(self.size)+")!"
                raise Exception(error_message)
            else:
                self.values = Q0

        # Set up learning rate
        if alpha is None:
            self.count_based = True
            self.counts = np.zeros(self.size)
        elif isinstance(alpha,(int,float)):
            self.count_based = False
            self.alpha = alpha
        else:
            error_message = "Invalid alpha: "+str(alpha)
            raise Exception(error_message)

    def __str__(self):
        '''
        Display utility function
        '''
        text = ""
        for i in range(self.size):
            text += str(self.index_to_assignment(i)) + " : " + str(self.values[i]) + "\n"
        return text

    def assignment_to_index(self,assignment):
        '''
        Takes in an assignment, returns the corresponding index

        Input:
            assignment - a dict mapping node names to values in their domains

        Output:
            index - an int index corresponding to the assignment
        '''
        index = 0
        for node in self.nodes:
            index += self.domains[node].index(assignment[node])*self.offsets[node]
        return index

    def get_value(self,assignment):
        '''
        Gets the value stored for a given assignment

        Input:
            assignment - a dict mapping node name to a value in the domain
        Output:
            value - the value of the assignment
        '''
        return self.values[self.assignment_to_index(assignment)]

    def index_to_assignment(self,index):
        '''
        Takes in an index, returns the corresponding assignment

        Input:
            index - an integer

        Output:
            assignment - a dict mapping node names to values in their domains
        '''
        assignment = {}
        for i in range(len(self.nodes)-1,-1,-1):
            assignment[self.nodes[i]] = self.domains[self.nodes[i]][index % len(self.domains[self.nodes[i]])]
            index = index // len(self.domains[self.nodes[i]])
        return assignment

    def update(self,trial):
        '''
        Update utility based on latest trial

        Input:
            trial - dict mapping node names (including reward) to values
        '''
        index = self.assignment_to_index(trial)
        if self.count_based:
            self.counts[index] += 1
            alpha = 1/self.counts[index]
        else:
            alpha = self.alpha
        self.values[index] = self.values[index] + alpha*(trial["reward"]-self.values[index])
