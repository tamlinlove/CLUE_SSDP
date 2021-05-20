class StateTable:
    '''
    Data structure for recording advice received by an agent
    '''
    def __init__(self,domain,default_value=None):
        '''
        Initialise state table

        Input:
            domain - dict mapping node name to domain
            default_value - dfeault value stored in table
                default: None
        '''
        self.domain = domain
        self.nodes = list(domain.keys())
        # Setup table
        self.size = 1
        self.offsets = {}
        for i in range(len(self.nodes)-1,-1,-1):
            self.offsets[self.nodes[i]]=self.size
            self.size *= len(self.domain[self.nodes[i]]) # Offset by size of domain
        # Fill in values
        self.values = []
        for i in range(self.size):
            self.values.append(default_value)

    def __str__(self):
        '''
        For printing a state table
        '''
        text = "==State Table==\n"
        for i in range(self.size):
            assignment = self.index_to_assignment(i)
            text += str(assignment) + ":" + str(self.values[i]) + "\n"
        return text

    def add_to_table(self,state,value):
        '''
        Adds something for the state to the table

        Input:
            state - dict mapping node names to domain values
            value - the thing to be added to the table
        '''
        index = self.assignment_to_index(state)
        self.values[index] = value

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
            index += self.domain[node].index(assignment[node])*self.offsets[node]
        return index

    def get_value(self,assignment):
        '''
        Gets the value for a given state

        Input:
            assignment - a dict mapping node name to a value in the domain
        Output:
            value - the value at the given assignment
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
            assignment[self.nodes[i]] = self.domain[self.nodes[i]][index % len(self.domain[self.nodes[i]])]
            index = index // len(self.domain[self.nodes[i]])
        return assignment
