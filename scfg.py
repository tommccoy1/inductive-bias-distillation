

import multiprocessing
import time

import numpy as np
import random
import math
import ast

import sys
sys.setrecursionlimit(100)


# Make the first-appearing one 0, the
# the next-appearing one 1, etc.
# e.g., [3,3,3,0,1,0,3,7] -> [0,0,0,1,2,1,0,3]
def standardize_sync_pattern(sync_pattern):
    vocab_size = 0
    seen = {}
    new_pattern = []

    for elt in sync_pattern:
        if elt in seen:
            new_pattern.append(seen[elt])
        else:
            seen[elt] = vocab_size
            vocab_size += 1
            new_pattern.append(seen[elt])

    return new_pattern

# Compute the prior for a synchrony pattern
def prior_sync_pattern(sync_pattern):
    n_options = len(sync_pattern)
    total_sync_patterns = n_options**len(sync_pattern)
    
    unique_count = 0
    seen = {}
    for elt in sync_pattern:
        if elt not in seen:
            unique_count += 1
        seen[elt] = 1
    
    # Suppose all are unique. Then the number of patterns that lead to
    # this standardized sync_pattern is the number of permutations.
    # If there's only one unique, then the number of patterns is just
    # the number of options - i.e., n_options! / (n_options-1)!
    possibilities = math.factorial(n_options) / math.factorial(n_options - unique_count)

    prob = math.log(possibilities) - math.log(total_sync_patterns)
    return prob



# How many heads fall in each synchrony pattern
def counts_from_sync_pattern(sync_pattern):
    counts = {}
    for elt in sync_pattern:
        if elt not in counts:
            counts[elt] = 0
        counts[elt] += 1

    return counts

class SyncGrammar:
    def __init__(self):
        self.rules = {}
        self.weights = {}

        self.string2primitive = {}

        # basic_primitives is defined near the bottom of the file
        for primitive, weight in basic_primitives:
            self.add(primitive, weight)

        self.init_probs()

    def add(self, primitive, weight):
        if primitive.output_type not in self.rules:
            self.rules[primitive.output_type] = []
            self.weights[primitive.output_type] = []

        self.rules[primitive.output_type].append(primitive)
        self.weights[primitive.output_type].append(weight)
        
        self.string2primitive[primitive_string(primitive)] = primitive

    def init_probs(self, divisions=100):
        for i in range(divisions-1):
            prob = (i+1.0)/divisions
            prob_rule = SyncPrimitive([], "Prob", lambda prob=prob: prob, name=str(prob))
            self.add(prob_rule, 1.0)


    def sample(self, nonterminal, count=None, vocab=None, node_index=0, input_nested_list=None):
        nonterminal = nonterminal.split("_")[0]

        choices = []
        weights = []

        prior = 0

        name = None
        if input_nested_list is not None:
            nonterminal = input_nested_list[0]
            nonterminal = nonterminal.split("_")[0]
            name = input_nested_list[1]

        if nonterminal == "T":

            sigma = False
            epsilon = False
            terminal = None
            if input_nested_list is not None:
                if name.startswith("sigma"):
                    sigma = True
                elif name.startswith("epsilon"):
                    epsilon = True
                else:
                    terminal = ast.literal_eval(name)

            # Either sigma or a terminal
            # Treat sigma as a vocab item, with uniform prob over vocab items
            if (input_nested_list is None and random.random() < 1.0/(len(vocab)+2)) or sigma:
                def sigma_f(k=count):
                    vocab_item = random.choice(vocab)[0]
                    vocab_tuple = [tuple([vocab_item for _ in range(k)])]
                    return vocab_tuple
                choice = SyncPrimitive(input_types=[], output_type="T", function=(lambda : sigma_f()), name="sigma" + str(len(vocab)))
                prior += math.log(1.0/(len(vocab)+2))

            elif (input_nested_list is None and random.random() < 2.0/(len(vocab)+2)) or epsilon:

                choice = SyncPrimitive(input_types=[], output_type="T", function=(lambda : []), name="epsilon")
                prior += math.log(1.0/(len(vocab)+2))


            else:
                if input_nested_list is not None:
                    terminals = [terminal]
                    count = len(terminals[0])
                else:
                    terminals = [tuple([x[0] for x in random.choices(vocab, k=count)])]
                
                choice = SyncPrimitive(input_types=[], output_type="T", function=(lambda terminals=terminals: terminals), name=str(terminals[0]))

                prior += math.log(len(vocab)/(len(vocab)+2)) - count*math.log(len(vocab))

        else:
            choice_arity = None
            choice_prob = None
            choice_star_position = None
            if input_nested_list is not None:
                trimmed_name = name
                if "(" in name:
                    trimmed_name = name[:name.index("(")]

                if trimmed_name in self.string2primitive:
                    choice = self.string2primitive[trimmed_name]
                else:
                    if name.startswith("concat"):
                        choice = self.string2primitive["CONCAT"]
                        choice_arity = name.count("%s")
                    elif name.startswith("plus"):
                        choice = self.string2primitive["PLUS"]
                        choice_arity = name.count("%s")
                        name_split = name.replace("(", ",")
                        name_split = name_split.split(",")
                        choice_prob = float(name_split[1])
                        choice_star_position = int(name_split[2])
                    elif name.startswith("star"):
                        choice = self.string2primitive["STAR"]
                        choice_arity = name.count("%s")
                        name_split = name.replace("(", ",")
                        name_split = name_split.split(",")
                        choice_prob = float(name_split[1])
                        choice_star_position = int(name_split[2])

                    intermediate_name = name[:name.index("(")].upper()
                    if nonterminal == "S":
                        intermediate_name = intermediate_name + "_T"

                    choice = self.string2primitive[intermediate_name]
            else:
                if nonterminal in self.rules:
                    choices = choices + self.rules[nonterminal]
                    weights = weights + self.weights[nonterminal]

                if choices == []:
                    print("No choices available for nonterminal:", nonterminal)
                    14/0

                choice = random.choices(choices, weights=weights)[0]

            prior += math.log(choice.weight)

        if choice.special_type is not None:
            if choice.special_type == "PLUS":
                choice = SyncPrimitive(input_type="L", output_type="L", function=(lambda prob, plus_ind, *x : plus_f(prob, plus_ind, *x)), name="plus(%s)", recurse=True, prob=choice_prob, arity=choice_arity, star_position=choice_star_position)
                prior += math.log(prob2prob[choice.prob]) # Probability
                prior += choice.arity*math.log(0.5) # Arity
                prior += -1*math.log(choice.arity) # Star positions

            elif choice.special_type == "STAR":
                choice = SyncPrimitive(input_type="L", output_type="L", function=(lambda prob, plus_ind, *x : star_f(prob, plus_ind, *x)), name="star(%s)", recurse=True, prob=choice_prob, arity=choice_arity, star_position=choice_star_position)
                prior += math.log(prob2prob(choice.prob))
                prior += choice.arity*math.log(0.5)
                prior += -1*math.log(choice.arity)

            elif choice.special_type == "CONCAT":
                choice = SyncPrimitive(input_type="L", output_type="L", function=(lambda *x : concat_f(*x)), name="concat(%s)", recurse=False, prob=choice_prob, arity=choice_arity, star_position=choice_star_position)
                prior += choice.arity*math.log(0.5)

        f = choice.execute
        name = choice.name
        nested_list = [[nonterminal + "_" + str(count), name, []]]
        nodes = [[node_index]]
        top_coordinate = 2

        arg_functions = []
        arg_names = []

        for i in range(choice.arity):
            if input_nested_list is not None:
                arg_function, arg_name, arg_nested_list, arg_nodes, arg_prior = self.sample(choice.input_types[i], count=count, vocab=vocab, node_index=i, input_nested_list=input_nested_list[2][i])
            else:
                arg_function, arg_name, arg_nested_list, arg_nodes, arg_prior = self.sample(choice.input_types[i], count=count, vocab=vocab, node_index=i)
            arg_functions.append(arg_function)
            arg_names.append(arg_name)
            nested_list[0][2] += arg_nested_list
            nodes += [[node_index,top_coordinate] + node for node in arg_nodes]
            prior += arg_prior

        new_f = lambda : f(*arg_functions)

        return new_f, name % tuple(arg_names), nested_list, nodes, prior


    def pretty_print(self):
        for nonterminal in self.rules:
            lhs = nonterminal

            for primitive in self.rules[nonterminal]:
                to_print = primitive.name % tuple(primitive.input_types)
                print(lhs + " -> " + to_print)

# The probability of each probability of recursion
prob2prob = {0.15: 0.0003, 0.16: 0.0004, 0.17: 0.0005, 0.18: 0.0007, 0.19: 0.0014, 0.2: 0.0026, 0.21: 0.0036, 0.22: 0.0069, 0.23: 0.0115, 0.24: 0.0145, 0.25: 0.0238, 0.26: 0.0282, 0.27: 0.0394, 0.28: 0.0491, 0.29: 0.0571, 0.3: 0.065, 0.31: 0.0768, 0.32: 0.0733, 0.33: 0.0801, 0.34: 0.0797, 0.35: 0.0744, 0.36: 0.0646, 0.37: 0.0581, 0.38: 0.0485, 0.39: 0.0387, 0.4: 0.0295, 0.41: 0.0261, 0.42: 0.0176, 0.43: 0.01, 0.44: 0.0073, 0.45: 0.0053, 0.46: 0.0019, 0.47: 0.0018, 0.48: 0.0003, 0.49: 0.0005, 0.5: 0.0004, 0.52: 0.0001}
all_probs = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.52]
all_probs_weights = [0.0003, 0.0004, 0.0005, 0.0007, 0.0014, 0.0026, 0.0036, 0.0069, 0.0115, 0.0145, 0.0238, 0.0282, 0.0394, 0.0491, 0.0571, 0.065, 0.0768, 0.0733, 0.0801, 0.0797, 0.0744, 0.0646, 0.0581, 0.0485, 0.0387, 0.0295, 0.0261, 0.0176, 0.01, 0.0073, 0.0053, 0.0019, 0.0018, 0.0003, 0.0005, 0.0004, 0.0001]
def select_prob():
    #prob = np.random.normal(loc=0.33, scale=0.05)
    #if prob <= 0.15:
    #    prob = 0.15
    #if prob >= 1:
    #    prob = 0.99

    #prob = round(prob, 2)

    # Replacing the above with a discretized version using stored values
    prob = random.choices(all_probs, weights=all_probs_weights)[0]

    return prob

class SyncPrimitive:
    def __init__(self, input_types=None, special_type=None, input_type=None, output_type=None, function=None, name=None, recurse=False, weight=None, prob=None, arity=None, star_position=None):
        
        self.recurse = recurse
        self.special_type = special_type

        if input_types is not None:
            self.input_types = input_types
            if arity is None:
                self.arity = len(input_types)
            else:
                self.arity = arity
        elif input_type is not None:
            if arity is None:
                self.arity = np.random.geometric(0.5)
            else:
                self.arity = arity

            self.input_types = [input_type for _ in range(self.arity)]
            if self.recurse:
                if prob is None:
                    self.prob = select_prob()
                else:
                    self.prob = prob

                if star_position is None:
                    self.star_position = random.choice(list(range(self.arity)))
                else:
                    self.star_position = star_position

                name = name.replace("(", "(" + str(self.prob) + ", " + str(self.star_position) + ", ")
        elif special_type is None:
            raise ValueError("Need input_type or input_types or special_type")

        self.output_type = output_type
        self.name = name

        if input_type is not None:
            new_string_template = " ".join(["%s" for _ in range(self.arity)])
            self.name = self.name.replace("%s", new_string_template)

        self.function = function
        self.weight = weight
        
    def execute(self, *args):
        
        if self.recurse:
            to_return = self.function(self.prob, self.star_position, *args)
        else:
            to_return = self.function(*args)
            
        return to_return

# Defining the primitives
def aorb_f(x, y):
    return lambda : random.choice([x()(), y()()])

def question_mark_f(x):
    return lambda : random.choice([x()(), []])

def combine_lists(list_of_lists, combine_index):
    combined = []

    for elt in list_of_lists:
        combined = elt[:combine_index] + combined + elt[combine_index:]

    # Flatten list
    return [item for sublist in combined for item in sublist]

def plus_f(prob, plus_index, *x):
    number_iterations = np.random.geometric(prob)

    levels = []

    for _ in range(number_iterations):
        levels.append(x)

    return lambda : combine_lists([[elt()() for elt in x] for x in levels], plus_index)


def star_f(prob, plus_index, *x):
    number_iterations = np.random.geometric(prob) - 1

    levels = []

    for _ in range(number_iterations):
        levels.append(x)

    return lambda : combine_lists([[elt()() for elt in x] for x in levels], plus_index)


def flatten(lst):
    new_lst = []
    for elt in lst:
        for item in elt:
            new_lst.append(item)

    return new_lst

def concat_f(*x):

    return lambda : flatten([elt()() for elt in x])

def stringify_tuple_index(lst, index):
    stringified = []

    for elt in lst:
        if isinstance(elt, tuple):
            stringified.append(elt[index])
        else:
            stringified.append(elt)

    return " ".join(stringified)

def stringify_tuple(lst, count):
    strings = []

    for index in range(count):
        strings.append(stringify_tuple_index(lst, index))

    return strings


# Insert the component strings into the synchrony pattern
def insert_into_pattern(string_list, sync_pattern):
    output = []

    count_used_per_pattern = {}

    for pattern_id in sync_pattern:
        if pattern_id not in count_used_per_pattern:
            count_used_per_pattern[pattern_id] = 0
        next_string = string_list[pattern_id][count_used_per_pattern[pattern_id]]
        if next_string != "":
            output.append(next_string)
        count_used_per_pattern[pattern_id] += 1

    return " ".join(output)


class SyncHypothesis:
    def __init__(self, input_nested_list=None, vocab=None, sync_pattern=None):
        
        self.grammar = SyncGrammar()

        self.init_vocab(vocab=vocab)
    
        self.prior = len(self.vocab) * math.log(0.5)

        if sync_pattern is not None:
            # Vocab should be introduced as a list of single tokens, not tuples: [0,1,2]
            # sync_pattern should be introduced as normal
            self.sync_pattern = sync_pattern
            self.n_heads = len(self.sync_pattern)
        else:
            # Number of positions to write from
            # 1: 0.5, 2: 0.25, 3: 0.125, ...
            self.n_heads = np.random.geometric(0.5)

            # Create the pattern of synchrony across the heads
            self.sync_pattern = []
            choices = list(range(self.n_heads))
            for i in range(self.n_heads):
                self.sync_pattern.append(random.choice(choices))
            self.sync_pattern = standardize_sync_pattern(self.sync_pattern)
 
        self.prior += self.n_heads * math.log(0.5)
        self.prior += prior_sync_pattern(self.sync_pattern)

        # How many times each unique synchrony pattern is used
        counts_per_pattern = counts_from_sync_pattern(self.sync_pattern)

        self.rule_names = []
        self.rule_to_calls = []
        self.rule_nested_lists = [[["SYNC", sync_id] for sync_id in self.sync_pattern], []]
        self.head_node = [[-1]]
        self.sync_nodes = []
        for index in range(len(self.sync_pattern)):
            self.sync_nodes.append([0, index])
        self.rule_nodes = []

        for pattern in counts_per_pattern:
            if input_nested_list is None:
                rule_function, rule_name, rule_nested_list, rule_nodes, rule_prior = self.init_rule(count=counts_per_pattern[pattern])
            else:
                rule_function, rule_name, rule_nested_list, rule_nodes, rule_prior = self.init_rule(count=counts_per_pattern[pattern], input_nested_list=input_nested_list[1][pattern][0])

            self.rule_names.append(rule_name)
            self.rule_to_calls.append(rule_function)
            self.rule_nested_lists[1].append(rule_nested_list)
            this_rule_node_list = []
            for x in rule_nodes:
                this_rule_node_list.append([1,pattern] + x)
            self.rule_nodes.append(this_rule_node_list)
            self.prior += rule_prior

        self.to_call = lambda x : insert_into_pattern([rule(x) for rule in self.rule_to_calls], self.sync_pattern)
        self.n_nonterminals = len(self.all_nodes())

    def init_vocab(self, vocab=None):
        if vocab is None:
            vocab_size = np.random.geometric(0.5) + 1
            if vocab_size > 10:
                # Cap the vocab size at 10
                vocab_size = 10
            vocab = list(range(vocab_size))

        self.vocab = [[str(x)] for x in vocab]
   

    def init_rule(self, count=None, input_nested_list=None):

        rule_function, rule_name, rule_nested_list, rule_nodes, rule_prior = self.grammar.sample("S", count=count, vocab=self.vocab, input_nested_list=input_nested_list)

        def to_call(x, count=count):
            return stringify_tuple(rule_function()(), count)

        return to_call, rule_name, rule_nested_list, rule_nodes, rule_prior

    def all_nodes(self):
        return self.head_node + self.sync_nodes + [rule_node for rule_node_list in self.rule_nodes for rule_node in rule_node_list]

    def pretty_print(self):
        print("VOCAB", self.vocab)
        print("PATTERN: " + ",".join([str(x) for x in self.sync_pattern]))
        for index, name in enumerate(self.rule_names):
            print("RULE " + str(index) + ": " + name)
        



def random_sync():
    hyp = SyncHypothesis()
    return hyp


def primitive_string(primitive):
    name = primitive.name
    if "(" in name:
        return name[:name.index("(")]
    else:
        return name

terminal_top_w = 0.0204
aorb_top_w = 0.3265
plus_top_w = 0.3265
concat_top_w = 0.3265
terminal_w = 0.8421
aorb_w = 0.0526
plus_w = 0.0526
concat_w = 0.0526

terminal_top = SyncPrimitive(input_types=["T"], output_type="S", function=(lambda x : x), name="%s_T", weight=terminal_top_w)
aorb_top = SyncPrimitive(input_types=["L", "L"], output_type="S", function=(lambda x,y : aorb_f(x, y)), name="aorb_T(%s, %s)", weight=aorb_top_w)
plus_top = SyncPrimitive(special_type="PLUS", output_type="S", name="PLUS_T", weight=plus_top_w)
star_top = SyncPrimitive(special_type="STAR", output_type="S", name="STAR_T")
concat_top = SyncPrimitive(special_type="CONCAT", output_type="S", name="CONCAT_T", weight=concat_top_w)


terminal = SyncPrimitive(input_types=["T"], output_type="L", function=(lambda x : x), name="%s", weight=terminal_w)
aorb = SyncPrimitive(input_types=["L", "L"], output_type="L", function=(lambda x,y : aorb_f(x, y)), name="aorb(%s, %s)", weight=aorb_w)
plus = SyncPrimitive(special_type="PLUS", output_type="L", name="PLUS", weight=plus_w)
star = SyncPrimitive(special_type="STAR", output_type="L", name="STAR")
concat = SyncPrimitive(special_type="CONCAT", output_type="L", name="CONCAT", weight=concat_w)

basic_primitives = [(terminal_top, terminal_top_w), (aorb_top, aorb_top_w), (plus_top, plus_top_w), (concat_top, concat_top_w), (terminal, terminal_w), (aorb, aorb_w), (plus, plus_w), (concat, concat_w)]


if __name__ == "__main__":
    for _ in range(5):
        hyp = random_sync()
        hyp.pretty_print()
        for _ in range(5):
            print(hyp.to_call([]))
        print("")


