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

class Factor_observed(Factor):
    def __init__(self,factor,obs):
        Factor.__init__(self, [v for v in factor.variables if v not in obs])
        self.observed = obs
        self.orig_factor = factor

    def get_value(self,assignment):
        ass = assignment.copy()
        for ob in self.observed:
            ass[ob]=self.observed[ob]
        return self.orig_factor.get_value(ass)

class Factor_sum(Factor_stored):
    def __init__(self,var,factors):
        self.var_summed_out = var
        self.factors = factors
        vars = []
        for fac in factors:
            for v in fac.variables:
                if v is not var and v not in vars:
                    vars.append(v)
        Factor_stored.__init__(self,vars,None)
        self.values = [None]*self.size

    def get_value(self,assignment):
        """lazy implementation: if not saved, compute it. Return saved value"""
        index = self.assignment_to_index(assignment)
        if self.values[index]:
            return self.values[index]
        else:
            total = 0
            new_asst = assignment.copy()
            for val in self.var_summed_out.domain:
                new_asst[self.var_summed_out] = val
                prod = 1
                for fac in self.factors:
                    prod *= fac.get_value(new_asst)
                total += prod
            self.values[index] = total
            return total

def factor_times(variable,factors):
    """when factors are factors just on variable (or on no variables)"""
    prods= []
    facs = [f for f in factors if variable in f.variables]
    for val in variable.domain:
        prod = 1
        ast = {variable:val}
        for f in facs:
            prod *= f.get_value(ast)
        prods.append(prod)
    return prods

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

class Factor_max(Factor_stored):
    """A factor obtained by maximizing a variable in a factor.
    Also builds a decision_function. This is based on Factor_sum.
    """

    def __init__(self, dvar, factor):
        """dvar is a decision variable.
        factor is a factor that contains dvar and only parents of dvar
        """
        self.dvar = dvar
        self.factor = factor
        vars = [v for v in factor.variables if v is not dvar]
        Factor_stored.__init__(self,vars,None)
        self.values = [None]*self.size
        self.decision_fun = Factor_DF(dvar,vars,[None]*self.size)

    def get_value(self,assignment):
        """lazy implementation: if saved, return saved value, else compute it"""
        index = self.assignment_to_index(assignment)
        if self.values[index]:
            return self.values[index]
        else:
            max_val = float("-inf")  # -infinity
            new_asst = assignment.copy()
            for elt in self.dvar.domain:
                new_asst[self.dvar] = elt
                fac_val = self.factor.get_value(new_asst)
                if fac_val>max_val:
                    max_val = fac_val
                    best_elt = elt
            self.values[index] = max_val
            self.decision_fun.values[index] = best_elt
            return max_val

class Factor_DF(Factor_stored):
    """A decision function"""
    def __init__(self,dvar, vars, values):
        Factor_stored.__init__(self,vars,values)
        self.dvar = dvar
        self.name = str(dvar)  # Used in printing

class Utility(Factor_stored):
    """A factor defined by a utility"""
    def __init__(self,vars,table):
        """Creates a factor on vars from the table.
        The table is ordered according to vars.
        """
        Factor_stored.__init__(self,vars,table)
        assert self.size==len(table),"Table size incorrect "+str(self)

class Displayable(object):
    """Class that uses 'display'.
    The amount of detail is controlled by max_display_level
    """
    max_display_level = 1   # can be overridden in subclasses

    def display(self,level,*args,**nargs):
        """print the arguments if level is less than or equal to the
        current max_display_level.
        level is an integer.
        the other arguments are whatever arguments print can take.
        """
        if level <= self.max_display_level:
            print(*args, **nargs)  ##if error you are using Python2 not Python3

class Inference_method(Displayable):
    """The abstract class of graphical model inference methods"""
    def query(self,qvar,obs={}):
        raise NotImplementedError("Inference_method query")   # abstract method

class VE(Inference_method):
    """The class that queries Graphical Models using variable elimination.

    gm is graphical model to query
    """
    def __init__(self,gm=None):
        self.gm = gm

    def query(self,var,obs={},elim_order=None):
        """computes P(var|obs) where
        var is a variable
        obs is a variable:value dictionary"""
        if var in obs:
            return [1 if val == obs[var] else 0 for val in var.domain]
        else:
            if elim_order == None:
                elim_order = self.gm.variables
            projFactors = [self.project_observations(fact,obs)
                           for fact in self.gm.factors]
            for v in elim_order:
                if v != var and v not in obs:
                    projFactors = self.eliminate_var(projFactors,v)
            unnorm = factor_times(var,projFactors)
            p_obs=sum(unnorm)
            self.display(1,"Unnormalized probs:",unnorm,"Prob obs:",p_obs)
            return {val:pr/p_obs for val,pr in zip(var.domain, unnorm)}

    def project_observations(self,factor,obs):
        """Returns the resulting factor after observing obs

        obs is a dictionary of variable:value pairs.
        """
        if any((var in obs) for var in factor.variables):
            # a variable in factor is observed
            return Factor_observed(factor,obs)
        else:
            return factor

    def eliminate_var(self,factors,var):
        """Eliminate a variable var from a list of factors.
        Returns a new set of factors that has var summed out.
        """
        self.display(2,"eliminating ",str(var))
        contains_var = []
        not_contains_var = []
        for fac in factors:
            if var in fac.variables:
                contains_var.append(fac)
            else:
                not_contains_var.append(fac)
        if contains_var == []:
            return factors
        else:
            newFactor = Factor_sum(var,contains_var)
            self.display(2,"Multiplying:",[f.brief() for f in contains_var])
            self.display(2,"Creating factor:", newFactor.brief())
            self.display(3, newFactor)  # factor in detail
            not_contains_var.append(newFactor)
            return not_contains_var

class VE_DN(VE):
    """Variable Elimination for Decision Networks"""
    def __init__(self,dn=None):
        """dn is a decision network"""
        VE.__init__(self,dn)
        self.dn = dn

    def optimize(self,elim_order=None,obs={},model=None):
        if elim_order == None:
                elim_order = self.gm.variables
        policy = []
        proj_factors = [self.project_observations(fact,obs)
                           for fact in self.dn.factors]
        trace = [proj_factors]
        for v in elim_order:
            if isinstance(v,DecisionVariable):
                to_max = [fac for fac in proj_factors
                          if v in fac.variables and set(fac.variables) <= v.all_vars]
                if len(to_max)!=1:
                    print("Decvar:",v)
                    print("to_max",to_max)
                assert len(to_max)==1, "illegal variable order "+str(elim_order)+" at "+str(v)
                newFac = Factor_max(v, to_max[0])
                policy.append(newFac.decision_fun)
                proj_factors = [fac for fac in proj_factors if fac is not to_max[0]]+[newFac]
                self.display(2,"maximizing",v,"resulting factor",newFac.brief() )
                self.display(3,newFac)
            else:
                proj_factors = self.eliminate_var(proj_factors, v)
            trace.append(proj_factors)
        if len(proj_factors)!=1:
            '''
            print(trace)
            print(elim_order)
            print(model.edges)
            print("Policy")
            for pol in policy:
                print(pol)
            for fac in proj_factors:
                if isinstance(fac,Factor_sum):
                    print("Factor ",fac.name," = Factor sum of ",fac.var_summed_out)
                    print(fac)
                elif isinstance(fac,Factor_max):
                    print("Factor ",fac.name," = Factor max of ",fac.dvar.name)
                    print(fac)
            '''
            # Hacky fix, will return to this later
            new_proj_factors = []
            for fac in proj_factors:
                if isinstance(fac,Factor_sum):
                    if fac.values[0] is None:
                        fac.get_value(fac.index_to_assignment(0))
                    if fac.values[0]!=1:
                        new_proj_factors.append(fac)
                else:
                    new_proj_factors.append(fac)

            proj_factors = new_proj_factors
            #print(proj_factors)
        assert len(proj_factors)==1,"Should there be only one element of proj_factors?"
        value = proj_factors[0].get_value({})
        return value,policy
