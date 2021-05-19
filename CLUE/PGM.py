'''
This file combines and modifies several classes and methods created by
David L Poole and Alan K Mackworth
See their copyright below

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
'''

class Graphical_model(object):
    """The class of graphical models.
    A graphical model consists of a set of variables and a set of factors.

    List vars is a list of variables
    List factors is a list of factors
    """
    def __init__(self,vars=None,factors=None):
        self.variables = vars
        self.factors = factors

class DecisionNetwork(Graphical_model):
    def __init__(self,vars=None,factors=None):
        """vars is a list of variables
        factors is a list of factors (instances of Prob and Utility)
        """
        Graphical_model.__init__(self,vars,factors)

class Variable(object):
    """A random variable.
    name (string) - name of the variable
    domain (list) - a list of the values for the variable.
    Variables are ordered according to their name.
    """

    def __init__(self,name,domain):
        self.name = name
        self.size = len(domain)
        self.domain = domain
        self.val_to_index = {} # map from domain to index
        for i,val in enumerate(domain):
            self.val_to_index[val]=i

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Variable('"+self.name+"')"

class DecisionVariable(Variable):
    def __init__(self,name,domain,parents):
        Variable.__init__(self,name,domain)
        self.parents = parents
        self.all_vars = set(parents) | {self}

class Factor(object):
    nextid=0  # each factor has a unique identifier; for printing

    def __init__(self,variables):
        """variables is the ordered list of variables
        """
        self.variables = variables   # ordered list of variables
        # Compute the size and the offsets for the variables
        self.var_offsets = {}
        self.size = 1
        for i in range(len(variables)-1,-1,-1):
            self.var_offsets[variables[i]]=self.size
            self.size *= variables[i].size
        self.id = Factor.nextid
        self.name = "f"+str(self.id)
        Factor.nextid += 1

    def get_value(self,assignment):
        raise NotImplementedError("get_value")   # abstract method

    def __str__(self, variables=None):
        """returns a string representation of the factor.
        Allows for an arbitrary variable ordering.
        variables is a list of the variables in the factor
        (can contain other variables)"""
        if variables==None:
            variables = self.variables
        else:
            variables = [v for v in variables if v in self.variables]
        res = ""
        for v in variables:
            res += str(v) + "\t"
        res += self.name+"\n"
        for i in range(self.size):
            asst = self.index_to_assignment(i)
            for v in variables:
                res += str(asst[v])+"\t"
            res += str(self.get_value(asst))
            res += "\n"
        return res

    def brief(self):
        """returns a string representing a summary of the factor"""
        res = self.name+"("
        for i in range(0,len(self.variables)-1):
            res += str(self.variables[i])+","
        if len(self.variables)>0:
            res += str(self.variables[len(self.variables)-1])
        res += ")"
        return res

    __repr__ = brief

    def assignment_to_index(self,assignment):
        """returns the index where the variable:value assignment is stored"""

        index = 0
        for var in self.variables:
            index += var.val_to_index[assignment[var]]*self.var_offsets[var]
        return index

    def index_to_assignment(self,index):
        """gives a dict representation of the variable assignment for index
        """
        asst = {}
        for i in range(len(self.variables)-1,-1,-1):
            asst[self.variables[i]] = self.variables[i].domain[index % self.variables[i].size]
            index = index // self.variables[i].size
        return asst

class Factor_stored(Factor):
    def __init__(self,variables,values):
        Factor.__init__(self, variables)
        self.values = values

    def get_value(self,assignment):
        return self.values[self.assignment_to_index(assignment)]

class Prob(Factor_stored):
    """A factor defined by a conditional probability table"""
    def __init__(self,var,pars,cpt):
        """Creates a factor from a conditional probability table, cptf.
        The cpt values are assumed to be for the ordering par+[var]
        """
        Factor_stored.__init__(self,pars+[var],cpt)
        self.child = var
        self.parents = pars
        assert self.size==len(cpt),"Table size incorrect "+str(self)

    def cond_dist(self,par_assignment):
        """returns the distribution (a val:prob dictionary) over the child given
        assignment to the parents

        par_assignment is a variable:value dictionary that assigns values to parents
        """
        index = 0
        for var in self.parents:
            index += var.val_to_index[par_assignment[var]]*self.var_offsets[var]
        # index is the position where the disgribution starts
        return {self.child.domain[i]:self.values[index+i] for i in range(len(self.child.domain))}

    def cond_prob(self,par_assignment,child_value):
        """returns the probability child has child_value given
        assignment to the parents

        par_assignment is a variable:value dictionary that assigns values to parents
        child_value is a value to the child
        """
        index = self.child.val_to_index[child_value]
        for var in self.parents:
            index += var.val_to_index[par_assignment[var]]*self.var_offsets[var]
        return self.values[index]

class Utility(Factor_stored):
    """A factor defined by a utility"""
    def __init__(self,vars,table):
        """Creates a factor on vars from the table.
        The table is ordered according to vars.
        """
        Factor_stored.__init__(self,vars,table)
        assert self.size==len(table),"Table size incorrect "+str(self)
