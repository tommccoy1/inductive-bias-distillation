
import multiprocessing
import time

import numpy as np
import random

import sys
sys.setrecursionlimit(100)

data_types = ["C", "S", "Bool", "StrSet", "Prob"]
data_type_probs = [0.2,0.2,0.2,0.2,0.2]

class Primitive:
    def __init__(self, input_types, output_type, function, name, factor=False, memoized=False):
        self.input_types = input_types
        self.output_type = output_type
        self.arity = len(input_types)
        self.name = name
        self.function = function
        self.factor = factor
        self.memoized = memoized
        self.memoization_dict = {}

        self.prnt = False

    def execute(self, *args):
        if self.memoized:
            # Converting to string so it can be hashed
            if str(args) in self.memoization_dict:
                return self.memoization_dict[str(args)]
            
        
        if self.prnt:
            print(self.name, args)

        if self.factor:
            if self.memoized:
                if str(args) in self.memoization_dict:
                    return self.memoization_dict[str(args)]
                
            to_return = self.function(*args)
            
            if self.memoized and str(args) not in self.memoization_dict:
                self.memoization_dict[str(args)] = to_return
                
        else:
            if self.memoized:
                if str(args[1:]) in self.memoization_dict:
                    return self.memoization_dict[str(args[1:])]
                
            # Strip off the first argument: it is x, which
            # is only included in factors
            to_return = self.function(*args[1:])
            
            if self.memoized and str(args[1:]) not in self.memoization_dict:
                self.memoization_dict[str(args[1:])] = to_return

        if self.prnt:
            print(to_return)
            print("")
            

        return to_return


    
def primitive_string(primitive):
    name = primitive.name
    if "(" in name:
        return name[:name.index("(")]
    else:
        return name
    
    
class Grammar:
    def __init__(self):
        # Permanent rules and their weights
        self.rules = {}
        self.weights = {}

        # Temporary rules and their weights (arising from factors)
        self.temporary_rules = {}
        self.temporary_weights = {}
        
        self.string2primitive = {}

    def add(self, primitive, weight):
        if primitive.output_type not in self.rules:
            self.rules[primitive.output_type] = []
            self.weights[primitive.output_type] = []

        self.rules[primitive.output_type].append(primitive)
        self.weights[primitive.output_type].append(weight)
        
        self.string2primitive[primitive_string(primitive)] = primitive

    def add_temporary(self, primitive, weight):
        if primitive.output_type not in self.temporary_rules:
            self.temporary_rules[primitive.output_type] = []
            self.temporary_weights[primitive.output_type] = []

        self.temporary_rules[primitive.output_type].append(primitive)
        self.temporary_weights[primitive.output_type].append(weight)

    def add_basics(self, basic_primitives):
        for primitive, weight in basic_primitives:
            self.add(primitive, weight)

    def sample(self, nonterminal):
        choices = []
        weights = []

        if nonterminal in self.rules:
            choices = choices + self.rules[nonterminal]
            weights = weights + self.weights[nonterminal]

        if nonterminal in self.temporary_rules:
            choices = choices + self.temporary_rules[nonterminal]
            weights = weights + self.temporary_weights[nonterminal]

        if choices == []:
            print("No choices available for nonterminal:", nonterminal)
            14/0

        choice = random.choices(choices, weights=weights)[0]

        f = choice.execute
        name = choice.name

        arg_functions = []
        arg_names = []

        for i in range(choice.arity):
            arg_function, arg_name = self.sample(choice.input_types[i])
            arg_functions.append(arg_function)
            arg_names.append(arg_name)


        if name.startswith("if"):
            # Necessary to deal with recursion
            # within the "if" statement. This way
            # ensures lazy evaluation 
            def new_f(x):
                if arg_functions[0](x):
                    return arg_functions[1](x)
                else:
                    return arg_functions[2](x)

        else:
            new_f = lambda x : f(*[x] + [arg_function(x) for arg_function in arg_functions])

        return new_f, name % tuple(arg_names)


    def pretty_print(self):
        for nonterminal in self.rules:
            lhs = nonterminal

            if nonterminal in self.rules:
                for primitive in self.rules[nonterminal]:
                    to_print = primitive.name % tuple(primitive.input_types)
                    print(lhs + " -> " + to_print)

            if nonterminal in self.temporary_rules:
                for primitive in self.temporary_rules[nonterminal]:
                    to_print = primitive.name % tuple(primitive.input_types)
                    print(lhs + " -> " + to_print)



# Functions to help manage lists of factors to be defined
def list_remove(lst, elt_to_remove):
    new_list = []
    for elt in lst:
        if elt == elt_to_remove:
            pass
        else:
            new_list.append(elt)

    return new_list

def intersect(lst1, lst2):
    set1 = set(lst1)
    set2 = set(lst2)

    in_common = list(set1.intersection(set2))

    return in_common

def strip_comma(string):
    if string[-1] == ",":
        return string[:-1]
    else:
        return string


class Hypothesis:
    def __init__(self, geometric_p=0.5, terminal_w=None, sigma_w=None, prob_w_num=None, 
                 factor_w=None, x_w=None, prob_divisions=100, vocab_size=None, 
                 hypstring=None, basic_primitives=[]):
        
        self.geometric_p = geometric_p

        # This weight is divided equally among all terminals
        self.terminal_w = terminal_w
        self.sigma_w = sigma_w

        # This weight is divided equally among all probabilities
        self.prob_w_num = prob_w_num

        # This weight is divided equally among all factors
        self.factor_w = factor_w
        self.x_w = x_w

        self.vocab = []
        self.grammar = Grammar()
        self.grammar.add_basics(basic_primitives)

        self.init_vocab(geometric_p=self.geometric_p, vocab_size=vocab_size)
        self.init_probs(divisions=prob_divisions)
        
        if hypstring is None:
	    # If a hypothesis string is not provided, initialize randomly
            self.init_factors()
        else:
	    # If a hypothesis string is provided, use that
            self.init_from_string(hypstring)
 

    def init_vocab(self, geometric_p=None, vocab_size=None):
        if vocab_size is None:
            vocab_size = np.random.geometric(geometric_p)
        self.vocab = [[str(i)] for i in range(vocab_size)]

        for elt in self.vocab:
            # elt=elt is necessary to avoid late binding
            self.grammar.add(Primitive([], "C", lambda elt=elt: elt, elt[0]), self.terminal_w/vocab_size)

        sigma = Primitive([], "StrSet", lambda : self.vocab, "Sigma")
        self.grammar.add(sigma, self.sigma_w)

    def init_probs(self, divisions=100):
        for i in range(divisions-1):
            prob = (i+1.0)/divisions
            prob_rule = Primitive([], "Prob", lambda prob=prob: prob, str(prob))
            self.grammar.add(prob_rule, self.prob_w_num/divisions)

    def init_factors(self, geometric_p=0.5):

        factor_count = np.random.geometric(geometric_p)

        input_types = [random.choices(data_types, weights=data_type_probs)[0] for _ in range(factor_count)]
        output_types = [random.choices(data_types, weights=data_type_probs)[0] for _ in range(factor_count)]

        # The final factor is the outermost one and
        # must be string in, string out
        input_types[-1] = "S"
        output_types[-1] = "S"

        # Pointers to the primitives so we can
        # update their functions later
        factor_primitives = []
        factor_m_primitives = []

        for fci in range(factor_count):
            # Right now, the function is just a dummy
            # We will update it later
            factor_primitive = Primitive([input_types[fci]], output_types[fci], None, "F" + str(fci) + "(%s)")
            factor_primitives.append(factor_primitive)

            factor_m_primitive = Primitive([input_types[fci]], output_types[fci], None, "Fm" + str(fci) + "(%s)", memoized=True)
            factor_m_primitives.append(factor_m_primitive)
            
            self.grammar.add(factor_primitive, self.factor_w/factor_count)
            self.grammar.add(factor_m_primitive, self.factor_w/factor_count)
            
        # To be populated
        factor_functions = [None for _ in range(factor_count)]
        factor_names = [None for _ in range(factor_count)]

        for fi in range(factor_count):

	    # x is temporary because it can have a different type in different factors
            x_primitive = Primitive([], input_types[fi], lambda x : x, "x", factor=True)
            self.grammar.add_temporary(x_primitive, self.x_w)

            factor_function, factor_name = self.grammar.sample(output_types[fi])
            factor_functions[fi] = factor_function


	    # Fill in the factor functions now that they are defined
            factor_primitives[fi].function = factor_function
            factor_m_primitives[fi].function = factor_function
            factor_names[fi] = factor_name


            # Clear out the temporary rules
            self.grammar.temporary_rules = {}
            self.grammar.temporary_weights = {}

        self.factor_names = factor_names
       
        def to_call(x):
	    # Clear out the memoization dict (memoization only 
	    # happens within a call)
            for primitive in factor_m_primitives:
                primitive.memoization_dict = {}
               
            return " ".join(factor_functions[-1](x))

        self.to_call = to_call

   
    # factors is a string listing the factors         
    def init_from_string(self, factors):
        factor_list = factors.split("; ")
       
	# Add necessary primitives to the grammar 
        factor_primitives = []
        factor_m_primitives = []
        
        for fci in range(len(factor_list)):
            # Using "S" for convenience; doesn't really matter
            factor_primitive = Primitive(["S"], "S", None, "F" + str(fci) + "(%s)")
            factor_m_primitive = Primitive(["S"], "S", None, "Fm" + str(fci) + "(%s)", memoized=True)
            
            factor_primitives.append(factor_primitive)
            factor_m_primitives.append(factor_m_primitive)
            
            self.grammar.add(factor_primitive, self.factor_w)
            self.grammar.add(factor_m_primitive, self.factor_w)
          
        # Using "S" for convenience; doesn't really matter
        x_primitive = Primitive([], "S", lambda x : x, "x", factor=True)
        self.grammar.add(x_primitive, self.x_w)
       
	# Create the function for each factor 
        for factor in factor_list:
            sides = factor.split(" = ")
            lhs = sides[0]
            rhs = sides[1]
            
            factor_number = int(lhs[1:-3])
            
            rhs = rhs.replace("(", "( ").replace(")", " )").split()
            
            function = self.function_from_list(rhs, [])
            
            factor_primitives[factor_number].function = function
            factor_m_primitives[factor_number].function = function
            
        def to_call(x):
            for primitive in factor_m_primitives:
                primitive.memoization_dict = {}
                
            return " ".join(factor_primitives[-1].function(x))
        
        self.to_call = to_call
            
           
    # Create a function from a list of tokens in
    # the string defining it
    # E.g.: ["append(", "head(", "1", ")", "sample(", "union(", "Sigma", "set(", "3", ")", ")", ")", ")"]
    def function_from_list(self, to_parse, stack):
       
		# Base case: We've parsed everything 
        if to_parse == []:
            if len(stack) != 1:
                print("STACK WRONG LENGTH", len(stack))
                14/0
            return stack[0][0]
        
        else:
            next_arg = strip_comma(to_parse[0])
            to_parse = to_parse[1:]
            if next_arg[-1] == "(":
		# Function whose arguments are yet to be parsed

                # True means it has an open paren
                stack.append((next_arg[:-1], True))
                
            elif next_arg == ")":
				# Time to reduce

                last_open_in_stack = -1
                for index, elt in enumerate(stack):
                    if elt[1]:
                        last_open_in_stack = index
                        
                to_pop = stack[last_open_in_stack][0]
                
                # Only need the functions, not the Boolean tags
                arg_functions = [x[0] for x in stack[last_open_in_stack+1:]]

		# Create the function that we will reduce the last few
		# elements on the stack to                
                if to_pop == "if":
                    def new_f(x):
                        if arg_functions[0](x):
                            return arg_functions[1](x)
                        else:
                            return arg_functions[2](x)
                else:
                    f = self.grammar.string2primitive[to_pop].execute
                    new_f = lambda x : f(*[x] + [arg_function(x) for arg_function in arg_functions])
                    
                # False because it's not open
                stack = stack[:last_open_in_stack] + [(new_f, False)]
                
            else:
                if "/" in next_arg:
                    parts = next_arg.split("/")
                    numerator = int(parts[0])
                    denominator = int(parts[1])
                    prob = numerator*1.0/denominator

                    # A fraction that needs to be parsed
                    prim = Primitive([], "Prob", lambda : prob, str(prob))
                    arg_function = lambda x : prim.execute(*[x])
                else:
                    # Just some argument that doesn't call other arguments of its own
                    arg_function = lambda x : self.grammar.string2primitive[strip_comma(next_arg)].execute(*[x])
                stack.append((arg_function, False))
            
            return self.function_from_list(to_parse, stack)
                
 
    def pretty_print(self):
        for i, factor_name in enumerate(self.factor_names):
            print("F" + str(i) + "(x) = " + factor_name)
                


# Basic primitives

def head_f(s):
    if s == []:
        return []
    else:
        return [s[0]]

def insert_f(s1, s2):
    length = len(s1)
    if length == 0:
        return s2
    else:
        position = length // 2

        new_s = s1[:position] + s2 + s1[position:]

        return new_s

def union_f(l1, l2):
    l1_prime = l1[:]
    l2_prime = l2[:]

    if isinstance(l1_prime, str):
        l1_prime = [l1_prime]
    if isinstance(l2_prime, str):
        l2_prime = [l2_prime]
   
    if len(l1_prime) == 0:
        l1_prime = [l1_prime]
    if len(l2_prime) == 0:
        l2_prime = [l2_prime]

    if len(l1_prime) > 0:
        if isinstance(l1_prime[0], str):
            l1_prime = [l1_prime]
    if len(l2_prime) > 0:
        if isinstance(l2_prime[0], str):
            l2_prime = [l2_prime]
        
    l1_set = set([" ".join(x) for x in l1_prime])
    l2_set = set([" ".join(x) for x in l2_prime])

    unioned = list(l1_set.union(l2_set))

    return [x.split() for x in unioned]

def setminus_f(l1, l2):
    l1_prime = l1[:]
    l2_prime = l2[:]

    if isinstance(l1_prime, str):
        l1_prime = [l1_prime]
    if isinstance(l2_prime, str):
        l2_prime = [l2_prime]
 
    if len(l1_prime) == 0:
        l1_prime = [l1_prime]
    if len(l2_prime) == 0:
        l2_prime = [l2_prime]

    if len(l1_prime) > 0:
        if isinstance(l1_prime[0], str):
            l1_prime = [l1_prime]
    if len(l2_prime) > 0:
        if isinstance(l2_prime[0], str):
            l2_prime = [l2_prime]
    
    l1_set = set([" ".join(x) for x in l1_prime])
    l2_set = set([" ".join(x) for x in l2_prime])

    diffed = list(l1_set.difference(l2_set))

    return [x.split() for x in diffed]

def if_f(cond, s1, s2):
    if cond:
        return s1

    else:
        return s2

def pair_f(arg1, arg2):
    l1 = arg1
    l2 = arg2
    if isinstance(l1, str):
        l1 = [l1]
    if isinstance(l2, str):
        l2 = [l2]

    return l1 + l2


def setsample_f(set_arg):
    settified = set_arg
    if isinstance(settified, str):
        settified = [settified]

    if len(settified) > 0:
        if isinstance(settified[0], str):
            settified = [settified]

    if len(settified) == 0:
        return []
    else:
        return random.choice(settified)

tail = Primitive(["S"], "S", lambda x : x[1:], "tail(%s)")
head = Primitive(["S"], "C", lambda x : head_f(x), "head(%s)")
pair = Primitive(["S", "C"], "S", lambda x,y : pair_f(x, y), "pair(%s, %s)")
append = Primitive(["S", "S"], "S", lambda x,y : pair_f(x, y), "append(%s, %s)")
epsilon = Primitive([], "S", lambda : [], "epsilon")
equals = Primitive(["S", "S"], "Bool", lambda x,y : x == y, "equals(%s, %s)")
empty = Primitive(["S"], "Bool", lambda x : x == [], "empty(%s)")
insert = Primitive(["S", "S"], "S", lambda x,y : insert_f(x, y), "insert(%s, %s)")
settify = Primitive(["S"], "StrSet", lambda x : [x], "set(%s)")
union = Primitive(["StrSet", "StrSet"], "StrSet", lambda x,y : union_f(x,y), "union(%s, %s)")
setminus = Primitive(["StrSet", "StrSet"], "StrSet", lambda x,y : setminus_f(x,y), "setminus(%s, %s)")
and_p = Primitive(["Bool", "Bool"], "Bool", lambda x,y : x and y, "and(%s, %s)")
or_p = Primitive(["Bool", "Bool"], "Bool", lambda x,y : x or y, "or(%s, %s)")
not_p = Primitive(["Bool"], "Bool", lambda x : not x, "not(%s)")
if_s = Primitive(["Bool", "S", "S"], "S", lambda x,y,z : y if x else z, "if(%s, %s, %s)")
if_char = Primitive(["Bool", "C", "C"], "S", lambda x,y,z : y if x else z, "if(%s, %s, %s)")
if_strset = Primitive(["Bool", "StrSet", "StrSet"], "StrSet", lambda x,y,z : y if x else z, "if(%s, %s, %s)")
if_prob = Primitive(["Bool", "Prob", "Prob"], "Prob", lambda x,y,z : y if x else z, "if(%s, %s, %s)")
flip = Primitive(["Prob"], "Bool", lambda x : random.random() < x, "flip(%s)")
setsample = Primitive(["StrSet"], "S", lambda x : setsample_f(x), "sample(%s)")



def primitives_from_file(filename):
    param_dict = {}

    fi = open(filename)
    for line in fi:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue

        parts = line.split()
        param_dict[parts[0]] = float(parts[1])

    basic_primitives = [(tail, param_dict["TAIL_W"]), (head, param_dict["HEAD_W"]), (append, param_dict["APPEND_W"]), (pair, param_dict["PAIR_W"]),
        (epsilon, param_dict["EPSILON_W"]), (equals, param_dict["EQUALS_W"]), (empty, param_dict["EMPTY_W"]), (insert, param_dict["INSERT_W"]),
        (settify, param_dict["SETTIFY_W"]), (union, param_dict["UNION_W"]), (setminus, param_dict["SETMINUS_W"]), (and_p, param_dict["AND_W"]),
        (or_p, param_dict["OR_W"]), (not_p, param_dict["NOT_W"]), (if_s, param_dict["IFS_W"]), (if_char, param_dict["IFC_W"]), (if_strset, param_dict["IFSET_W"]),
        (if_prob, param_dict["IFP_W"]), (flip, param_dict["FLIP_W"]), (setsample, param_dict["SETSAMPLE_W"])]

    return param_dict, basic_primitives






if __name__ == "__main__":
    _, basic_primitives = primitives_from_file("yandp_weights/yandp_params_uniform.txt")
    for _ in range(5):
        dataset_created = False
        while not dataset_created:
            try:
                sys.setrecursionlimit(40)
                hyp = Hypothesis(geometric_p=0.5, terminal_w=1.0, sigma_w=1.0,
                        prob_w_num=1.0, factor_w=1.0, x_w=1.0,
                        prob_divisions=100, basic_primitives=basic_primitives)
                sys.setrecursionlimit(1000)

                data_list = []
                for attempt_number in range(100):
                    if len(data_list) == 5:
                        break

                    try:
                        sys.setrecursionlimit(40)
                        example = hyp.to_call([])
                        sys.setrecursionlimit(1000)
                        data_list.append(example)
                    except:
                        pass

                if len(data_list) == 5:
                    dataset_created = True

            except:
                # Failed to generate a grammar. Loop back to the start of the
                # while loop and try again
                continue

        hyp.pretty_print()
        for example in data_list:
            print(example)
