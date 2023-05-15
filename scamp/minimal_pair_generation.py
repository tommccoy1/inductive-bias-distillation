
import random
from collections import defaultdict

# To-do: 
# - Don't have 2 of same content word?

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--implausible", help="Have the generated sentences be semantically implausible", action='store_true')
parser.add_argument("--to_command_line", help="Print to the command line, instead of to a file", action='store_true')
parser.add_argument("--n_examples", help="Number of examples to generate", type=int, default=10)
parser.add_argument("--category", help="Category of examples to produce", type=str, default=None)
parser.add_argument("--directory_name", help="Name of directory where examples will be saved", type=str, default=None)
args = parser.parse_args()

###################################################################
# Tools for checking if words are in vocab
###################################################################

vocab_file = open("childes_vocab_counts.txt", "r")
vocab_dict = defaultdict(int)

for line in vocab_file:
    parts = line.strip().split("\t")
    vocab_dict[parts[0]] = int(parts[1])

def freq(example, threshold=50):
    for word in example.split():
        if vocab_dict[word] < threshold:
            return False
    return True

def filter_freq(word_list, threshold=50):
    new_list = []
    for word in word_list:
        if vocab_dict[word] >= threshold:
            new_list.append(word)

    return new_list

###################################################################
# Vocabulary: Determiners
###################################################################

det_sg = ["the", "this", "that"]
det_pl = ["the", "these", "those"]


###################################################################
# Vocabulary: Nouns
###################################################################

male_names = ["Liam", "Bill", "Steve", "Matt", "Noah", "Donald", "William", "Charles", "Clyde", "Joseph", "Robert", "Michael", "David", "Andrew", "James", "John", "Christopher", "Brian", "Mark", "Richard", "Jeffrey", "Scott", "Jason", "Kevin", "Steven", "Thomas", "Eric", "Daniel", "Timothy", "Paul", "Gregory", "Stephen", "Todd", "Ronald", "Patrick", "Gary", "Douglas", "Keith", "Craig", "George", "Larry", "Peter", "Jerry", "Dennis", "Brad", "Frank", "Aaron", "Russell", "Roger", "Carl", "Travis", "Adam", "Marcus", "Derek", "Vincent", "Wayne", "Benjamin", "Walter", "Alan", "Martin", "Bruce", "Brett", "Alexander", "Guy", "Dan"]
female_names = ["Jennifer", "Lisa", "Michelle", "Amy", "Angela", "Melissa", "Tammy", "Mary", "Tracy", "Julie", "Karen", "Laura", "Christine", "Nina", "Susan", "Dawn", "Stephanie", "Elizabeth", "Heather", "Tina", "Lori", "Patricia", "Sandra", "Wendy", "Rebecca", "Becca", "Nicole", "Donna", "Deborah", "Christina", "Denise", "Sharon", "Linda", "Maria", "Brenda", "Barbara", "Stacy", "Andrea", "Cheryl", "Jessica", "Kathleen", "Debra", "Nancy", "Ruth", "Grace", "Jill", "Theresa", "Dana", "Paula", "Rachel", "Catherine", "Sherry", "Gina", "Ann", "Anne", "Anna", "Kristin", "Leslie", "Sarah", "Sara", "Katherine", "Renee", "Diane", "Diana", "Carol", "Cindy", "Sally", "Holly", "Tanya", "Margaret", "Heidi", "Kristen", "Kirsten", "April", "Lissa", "Regina", "Suzanne", "Laurie", "Melanie", "Beth", "Melinda", "Janet", "Valerie", "Kayla", "Amelia", "Amanda", "Erin", "Danielle", "Connie", "Claire", "Colleen", "Julia", "Sabrina", "Alicia", "Naomi", "Allison", "Martha", "Vanessa", "Samantha", "Beverly", "Carmen", "Marie", "Becky", "Emily", "Rose", "Judy", "Ellen", "Helen", "Jane", "Alice", "Elaine", "Caroline", "Meredith", "Irene", "Ella"]

male_person_sg = ["boy", "man", "guy", "king"]
female_person_sg = ["girl", "woman", "lady", "queen"]
male_person_pl = ["boys", "men", "guys", "kings"]
female_person_pl = ["girls", "women", "ladies", "queens"]
male_relation_sg = ["brother", "son", "father", "dad", "nephew", "grandfather", "husband"]
male_relation_pl = ["brothers", "sons", "fathers", "dads", "nephews", "grandfathers", "husbands"]
female_relation_sg = ["sister", "daughter", "mother", "mom", "niece", "grandmother", "wife"]
female_relation_pl = ["sisters", "daughters", "mothers", "moms", "nieces", "grandmothers", "wives"]
relation_sg = ["cousin", "friend", "boss", "teacher", "student", "doctor", "dentist", "guest"]
relation_pl = ["cousins", "friends", "bosses", "teachers", "students", "doctors", "dentists", "guests"]

person_sg = ["dancer", "child", "teenager", "adult", "person", "doctor", "patient", "customer", "cashier", "teacher", "student", "guest", "driver", "lawyer", "dentist"]
person_pl = ["dancers", "children", "teenagers", "adults", "people", "doctors", "patients", "customers", "cashiers", "teachers", "students", "guests", "drivers", "lawyers", "dentists"]

animal_sg = ["dog", "cat", "bird", "snake", "rabbit", "hamster", "horse", "turtle", "mouse"]
animal_pl = ["dogs", "cats", "birds", "snakes", "rabbits", "hamsters", "horses", "turtles", "mice"]

nouns_person_sg = person_sg + male_person_sg + female_person_sg 
nouns_person_pl = person_pl + male_person_pl + female_person_pl
nouns_animate_sg = nouns_person_sg + animal_sg
nouns_animate_pl = nouns_person_pl + animal_pl
nouns_place_sg = ["school", "cafe", "hospital", "bank", "library", "school", "mall", "restaurant", "museum", "university", "park", "room", "kitchen"]
nouns_place_pl = ["schools", "cafes", "hospitals", "banks", "libraries", "schools", "malls", "restaurants", "museums", "universities", "parks", "rooms", "kitchens"]
nouns_inanimate_sg = ["cup", "plate", "fork", "glass", "dish", "book", "chair", "couch", "rug", "car", "book", "window", "vase", "mirror", "car", "bicycle", "plane", "truck", "ship", "toy", "bed", "box", "picture", "door", "hat", "spoon", "shoe", "bottle", "doll", "bag", "boat", "bowl", "rocket", "crayon", "refrigerators", "candle", "shirt", "button", "ring"]
nouns_inanimate_pl = ["cups", "plates", "forks", "glasses", "dishes", "books", "chairs", "couches", "rugs", "cars", "books", "windows", "vases", "mirrors", "cars", "bicycles", "planes", "trucks", "ships", "toys", "beds", "boxes", "pictures", "doors", "hats", "spoons", "shoes", "bottles", "dolls", "bags", "boats", "bowls", "rockets", "crayons", "refrigerators", "candles", "shirts", "buttons", "rings"]

selectional_categories = {}
selectional_categories["animate"] = [nouns_animate_sg, nouns_animate_pl]
selectional_categories["physent"] = [nouns_animate_sg+nouns_inanimate_sg, nouns_animate_pl+nouns_inanimate_pl]
selectional_categories["person"] = [nouns_person_sg, nouns_person_pl]
selectional_categories["place"] = [nouns_place_sg, nouns_place_pl]
selectional_categories["inanimate"] = [nouns_inanimate_sg, nouns_inanimate_pl]

selectional_categories_implausible = {}
selectional_categories_implausible["animate"] = [nouns_inanimate_sg+nouns_place_sg, nouns_inanimate_pl+nouns_place_pl]
selectional_categories_implausible["person"] = [nouns_inanimate_sg+nouns_place_sg, nouns_inanimate_pl+nouns_place_pl]
selectional_categories_implausible["inanimate"] = [nouns_person_sg, nouns_person_pl]

###################################################################
# Vocabulary: Adjectives
###################################################################

adjectives = {}
adjectives["physent"] = ["boring", "popular", "important"]
adjectives["animate"] = ["bored", "healthy", "surprised", "happy", "confident", "excited", "wrong", "lucky", "upset", "clever", "busy", "smart", "young", "tall", "kind", "nice", "hungry", "sleepy", "tired", "strong", "careful", "gentle", "angry", "friendly", "helpful", "curious", "confused", "quiet"] + adjectives["physent"]
adjectives["person"] = adjectives["animate"]
adjectives["place"] = ["clean", "messy", "big", "small", "new", "old"] + adjectives["physent"]
adjectives["inanimate"] = ["whole", "broken", "big", "small", "new", "old", "cheap", "expensive", "green", "blue", "red", "yellow", "purple", "orange", "brown", "white", "black", "gray", "great", "hot", "cold", "cute", "fine", "dry", "missing", "cool", "amazing", "awesome"] + adjectives["physent"]




###################################################################
# Vocabulary: Verbs
###################################################################

v_animate_physent = ["see", "like", "love", "hate", "remember", "forget", "notice", "think about"]
vs_animate_physent = ["sees", "likes", "loves", "hates", "remembers", "forgets", "notices", "thinks about"]
ving_animate_physent = ["remembering", "forgetting", "noticing", "thinking about"]
ved_animate_physent = ["saw", "liked", "loved", "hated", "remembered", "forgot", "noticed", "thought about"]
ven_animate_physent = ["seen", "remembered", "forgotten", "noticed", "thought about"]
allv_animate_physent = [v_animate_physent, vs_animate_physent, ving_animate_physent, ved_animate_physent, ven_animate_physent, {"subject" : "animate", "object" : "physent"}]

v_person_physent = ["describe", "discuss", "talk about"]
vs_person_physent = ["describes", "discusses", "talks about"]
ving_person_physent = ["describing", "discussing", "talking about"]
ved_person_physent = ["described", "discussed", "talked about"]
ven_person_physent = ["described", "discussed", "talked about"]
allv_person_physent = [v_person_physent, vs_person_physent, ving_person_physent, ved_person_physent, ven_person_physent, {"subject" : "person", "object" : "physent"}]

if args.implausible:
    v_animate_animate = ["know", "respect", "observe", "hear"]
    vs_animate_animate = ["knows", "respects", "observes", "hears"]
    ving_animate_animate = ["observing"]
    ved_animate_animate = ["knew", "respected", "observed", "heard"]
    ven_animate_animate = ["respected", "observed", "heard"]
    allv_animate_animate = [v_animate_animate, vs_animate_animate, ving_animate_animate, ved_animate_animate, ven_animate_animate, {"subject" : "person", "object" : "animate"}]
else:
    v_animate_animate = ["know", "respect", "look like", "sound like", "observe", "hear"]
    vs_animate_animate = ["knows", "respects", "looks like", "sounds like", "observes", "hears"]
    ving_animate_animate = ["observing"]
    ved_animate_animate = ["knew", "respected", "looked like", "sounded like", "observed", "heard"]
    ven_animate_animate = ["respected", "observed", "heard"]
    allv_animate_animate = [v_animate_animate, vs_animate_animate, ving_animate_animate, ved_animate_animate, ven_animate_animate, {"subject" : "person", "object" : "animate"}]


v_person_person = ["care for", "help", "talk to", "visit", "appreciate", "listen to", "work with", "praise", "bother", "confuse", "scare", "worry", "answer", "forgive", "teach", "understand"]
vs_person_person = ["cares for", "helps", "talks to", "visits", "appreciates", "listens to", "works with", "praises", "bothers", "confuses", "scares", "worries", "answers", "forgives", "teaches", "understands"]
ving_person_person = ["caring for", "helping", "talking to", "visiting", "listening to", "working with", "praising", "bothering", "confusing", "scaring", "worrying", "answering", "teaching"]
ved_person_person = ["cared for", "helped", "talked to", "visited", "appreciated", "listened to", "worked with", "praised", "bothered", "confused", "scared", "worried", "answered", "found", "lost", "forgave", "taught", "understood"]
ven_person_person = ["cared for", "helped", "talked to", "visited", "appreciated", "listened to", "worked with", "praised", "bothered", "confused", "scared", "worried", "answered", "found", "lost", "forgiven", "taught", "understood"]
allv_person_person = [v_person_person, vs_person_person, ving_person_person, ved_person_person, ven_person_person, {"subject" : "person", "object" : "person"}]

v_person_person_not_clausal = ["care for", "help", "talk to", "visit", "listen to", "work with", "praise", "bother", "confuse", "scare"]
vs_person_person_not_clausal = ["cares for", "helps", "talks to", "visits", "listens to", "works with", "praises", "bothers", "confuses", "scares"]
ving_person_person_not_clausal = ["caring for", "helping", "talking to", "visiting", "listening to", "working with", "praising", "bothering", "confusing", "scaring"]
ved_person_person_not_clausal = ["cared for", "helped", "talked to", "visited", "listened to", "worked with", "praised", "bothered", "confused", "scared"]
ven_person_person_not_clausal = ["cared for", "helped", "talked to", "visited", "listened to", "worked with", "praised", "bothered", "confused", "scared"]
allv_person_person_not_clausal = [v_person_person_not_clausal, vs_person_person_not_clausal, ving_person_person_not_clausal, ved_person_person_not_clausal, ven_person_person_not_clausal, {"subject" : "person", "object" : "person"}]

v_person_place = ["arrive at", "explore", "go to", "leave", "pass", "return to", "drive to", "run around", "walk through"]
vs_person_place = ["arrives at", "explores", "goes to", "leaves", "passes", "returns to", "drives to", "runs around", "walks through"]
ving_person_place = ["arriving at", "exploring", "going to", "leaving", "passing", "returning to", "driving to", "running around", "walking through"]
ved_person_place = ["arrived at", "explored", "went to", "left", "passed", "returned to", "drove to", "ran around", "walked through"]
ven_person_place = ["arrived at", "explored", "gone to", "left", "passed", "returned to", "driven to", "run around", "walked through"]
allv_person_place = [v_person_place, vs_person_place, ving_person_place, ved_person_place, ven_person_place, {"subject" : "person", "object" : "place"}]

v_person_object = ["clean", "buy", "sell", "lift", "have", "take", "bring", "observe", "find", "lose", "fix", "repair", "destroy", "drop", "hold", "keep", "make", "shake", "steal", "throw", "move"]
vs_person_object = ["cleans", "buys", "sells", "lifts", "has", "takes", "brings", "observes", "finds", "loses", "fixes", "repairs", "destroys", "drops", "holds", "keeps", "makes", "shakes", "steals", "throws", "moves"]
ving_person_object = ["cleaning", "buying", "selling", "lifting", "taking", "bringing", "observing", "finding", "losing", "fixing", "repairing", "destroying", "dropping", "holding", "keeping", "making", "shaking", "stealing", "throwing", "moving"]
ved_person_object = ["cleaned", "bought", "sold", "lifted", "had", "took", "brought", "observed", "found", "lost", "fixed", "repaired", "destroyed", "dropped", "held", "kept", "made", "shook", "stole", "threw", "moved"]
ven_person_object = ["cleaned", "bought", "sold", "lifted", "taken", "brought", "observed", "found", "lost", "fixed", "repaired", "destroyed", "dropped", "held", "kept", "made", "shaken", "stolen", "thrown", "moved"]
allv_person_object = [v_person_object, vs_person_object, ving_person_object, ved_person_object, ven_person_object, {"subject" : "person", "object" : "inanimate"}]

v_person_object_not_ditransitive = ["clean", "lift", "observe", "lose", "repair", "destroy", "drop", "hold", "keep", "shake", "move"]
vs_person_object_not_ditransitive = ["cleans", "lifts", "observes", "loses", "repairs", "destroys", "drops", "holds", "keeps", "shakes", "moves"]
ving_person_object_not_ditransitive = ["cleaning", "lifting", "observing", "losing", "repairing", "destroying", "dropping", "holding", "keeping", "shaking", "moving"]
ved_person_object_not_ditransitive = ["cleaned", "lifted", "observed", "lost", "repaired", "destroyed", "dropped", "held", "kept", "shook", "moved"]
ven_person_object_not_ditransitive = ["cleaned", "lifted", "observed", "lost", "repaired", "destroyed", "dropped", "held", "kept", "shaken", "moved"]
allv_person_object_not_ditransitive = [v_person_object_not_ditransitive, vs_person_object_not_ditransitive, ving_person_object_not_ditransitive, ved_person_object_not_ditransitive, ven_person_object_not_ditransitive, {"subject" : "person", "object" : "inanimate"}]


v_physent_object = ["break",]
vs_physent_object = ["breaks"]
ving_physent_object = ["breaking"]
ved_physent_object = ["broke"]
ven_physent_object = ["broken"]
allv_physent_object = [v_physent_object, vs_physent_object, ving_physent_object, ved_physent_object, ven_physent_object, {"subject" : "physent", "object" : "inanimate"}]

v_intrans_person = ["complain", "cry", "exercise", "grin", "laugh", "lie", "lose", "nod", "paint", "practice", "read", "scream", "shout", "sigh", "sing", "smile", "stretch", "talk", "wave", "whisper", "win", "yell", "sneeze", "cough", "yawn"]
vs_intrans_person = ["complains", "cries", "exercises", "grins", "laughs", "lies", "loses", "nods", "paints", "practices", "reads", "screams", "shouts", "sighs", "sings", "smiles", "stretches", "talks", "waves", "whispers", "wins", "yells", "sneezes", "coughs", "yawns"]
ving_intrans_person = ["complaining", "crying", "exercising", "grinning", "laughing", "lying", "losing", "nodding", "painting", "practicing", "reading", "screaming", "shouting", "sighing", "singing", "smiling", "stretching", "talking", "waving", "whispering", "winning", "yelling", "sneezing", "coughing", "yawning"]
ved_intrans_person = ["complained", "cried", "exercised", "grinned", "laughed", "lied", "lost", "nodded", "painted", "practiced", "read", "screamed", "shouted", "sighed", "sang", "smiled", "stretched", "talked", "waved", "whispered", "won", "yelled", "sneezed", "coughed", "yawned"]
ven_intrans_person = ["complained", "cried", "exercised", "grinned", "laughed", "lied", "lost", "nodded", "painted", "practiced", "read", "screamed", "shouted", "sighed", "sung", "smiled", "stretched", "talked", "waved", "whispered", "won", "yelled", "sneezed", "coughed", "yawned"]
allv_intrans_person = [v_intrans_person, vs_intrans_person, ving_intrans_person, ved_intrans_person, ven_intrans_person, {"subject" : "person"}]

v_intrans_animate = ["blink", "eat", "learn", "arrive", "grow", "run"]
vs_intrans_animate = ["blinks", "eats", "learns", "arrives", "grows", "runs"]
ving_intrans_animate = ["blinking", "eating", "learning", "arriving", "growing", "running"]
ved_intrans_animate = ["blinked", "ate", "learned", "arrived", "grew", "ran"]
ven_intrans_animate = ["blinked", "eaten", "learned", "arrived", "grown", "run"]
allv_intrans_animate = [v_intrans_animate, vs_intrans_animate, ving_intrans_animate, ved_intrans_animate, ven_intrans_animate, {"subject" : "animate"}]

v_intrans_inanimate = ["break", "roll away", "drop", "move", "collapse"]
vs_intrans_inanimate = ["breaks", "rolls away", "drops", "moves", "collapses"]
ving_intrans_inanimate = ["breaking", "rolling away", "dropping", "moving", "collapsing"]
ved_intrans_inanimate = ["broke", "rolled away", "dropped", "moved", "collapsed"]
ven_intrans_inanimate = ["broken", "rolled away", "dropped", "moved", "collapsed"]
allv_intrans_inanimate = [v_intrans_inanimate, vs_intrans_inanimate, ving_intrans_inanimate , ved_intrans_inanimate, ven_intrans_inanimate, {"subject" : "inanimate"}]

v_intrans_physent = ["disappear", "appear"]
vs_intrans_physent = ["disappears", "appears"]
ving_intrans_physent = ["disappearing", "appearing"]
ved_intrans_physent = ["disappeared", "appeared"]
ven_intrans_physent = ["disappeared", "appeared"]
allv_intrans_physent = [v_intrans_physent, vs_intrans_physent, ving_intrans_physent, ved_intrans_physent, ven_intrans_physent, {"subject" : "physent"}]

v_person_person_obligatory_transitive = ["like", "hate", "describe", "discuss", "respect", "appreciate", "praise", "bother", "confuse", "scare"]
vs_person_person_obligatory_transitive = ["likes", "hates", "describes", "discusses", "respects", "appreciates", "praises", "bothers", "confuses", "scares"]
ving_person_person_obligatory_transitive = ["liking", "describing", "respecting", "appreciating", "praising", "bothering", "scaring"]
ved_person_person_obligatory_transitive = ["liked", "hated", "described", "discussed", "respected", "appreciated", "praised", "bothered", "confused", "scared"]
ven_person_person_obligatory_transitive = ["liked", "hated", "described", "discussed", "respected", "appreciated", "praised", "bothered", "confused", "scared"]
allv_person_person_obligatory_transitive = [v_person_person_obligatory_transitive, vs_person_person_obligatory_transitive, ving_person_person_obligatory_transitive, ved_person_person_obligatory_transitive, ven_person_person_obligatory_transitive, {"subject" : "person", "object" : "person"}]

if args.implausible:
    v_person_obligatory_intransitive = ["blink", "complain", "grin", "laugh", "lie", "nod", "scream", "shout", "sigh", "smile", "talk", "whisper", "yell", "sneeze", "cough", "yawn"]
    vs_person_obligatory_intransitive = ["blinks", "complains", "grins", "laughs", "lies", "nods", "screams", "shouts", "sighs", "smiles", "talks", "whispers", "yells", "sneezes", "coughs", "yawns"]
    ving_person_obligatory_intransitive = ["blinking", "arriving", "complaining", "grinning", "laughing", "lying", "nodding", "screaming", "shouting", "sighing", "smiling", "talking", "whispering", "yelling", "sneezing", "coughing", "yawning"]
    ved_person_obligatory_intransitive = ["blinked", "complained", "grinned", "laughed", "lied", "nodded", "screamed", "shouted", "sighed", "smiled", "talked", "whispered", "yelled", "sneezed", "coughed", "yawned"]
    ven_person_obligatory_intransitive = ["blinked", "complained", "grinned", "laughed", "lied", "nodded", "screamed", "shouted", "sighed", "smiled", "talked", "whispered", "yelled", "sneezed", "coughed", "yawned"]
    allv_person_obligatory_intransitive = [v_person_obligatory_intransitive, vs_person_obligatory_intransitive, ving_person_obligatory_intransitive, ved_person_obligatory_intransitive, ven_person_obligatory_intransitive, {"subject" : "person"}]
else:
    v_person_obligatory_intransitive = ["blink", "arrive", "disappear", "appear", "complain", "grin", "laugh", "lie", "nod", "scream", "shout", "sigh", "smile", "talk", "whisper", "yell", "sneeze", "cough", "yawn"]
    vs_person_obligatory_intransitive = ["blinks", "arrives", "disappears", "appears", "complains", "grins", "laughs", "lies", "nods", "screams", "shouts", "sighs", "smiles", "talks", "whispers", "yells", "sneezes", "coughs", "yawns"]
    ving_person_obligatory_intransitive = ["blinking", "arriving", "disappearing", "appearing", "complaining", "grinning", "laughing", "lying", "nodding", "screaming", "shouting", "sighing", "smiling", "talking", "whispering", "yelling", "sneezing", "coughing", "yawning"]
    ved_person_obligatory_intransitive = ["blinked", "arrived", "disappeared", "appeared", "complained", "grinned", "laughed", "lied", "nodded", "screamed", "shouted", "sighed", "smiled", "talked", "whispered", "yelled", "sneezed", "coughed", "yawned"]
    ven_person_obligatory_intransitive = ["blinked", "arrived", "disappeared", "appeared", "complained", "grinned", "laughed", "lied", "nodded", "screamed", "shouted", "sighed", "smiled", "talked", "whispered", "yelled", "sneezed", "coughed", "yawned"]
    allv_person_obligatory_intransitive = [v_person_obligatory_intransitive, vs_person_obligatory_intransitive, ving_person_obligatory_intransitive, ved_person_obligatory_intransitive, ven_person_obligatory_intransitive, {"subject" : "person"}]

v_person_object_obligatory_transitive = ["buy", "sell", "take", "bring", "find", "fix", "repair", "destroy", "hold", "keep", "make", "steal"]
vs_person_object_obligatory_transitive = ["buys", "sells", "takes", "brings", "finds", "fixes", "repairs", "destroys", "holds", "keeps", "makes", "steals"]
ving_person_object_obligatory_transitive = ["buying", "selling", "taking", "bringing", "finding", "fixing", "repairing", "destroying", "holding", "keeping", "making", "stealing"]
ved_person_object_obligatory_transitive = ["bought", "sold", "took", "brought", "found", "fixed", "repaired", "destroyed", "held", "kept", "made", "stole"]
ven_person_object_obligatory_transitive = ["bought", "sold", "taken", "brought", "found", "fixed", "repaired", "destroyed", "held", "kept", "made", "stolen"]
allv_person_object_obligatory_transitive = [v_person_object_obligatory_transitive, vs_person_object_obligatory_transitive, ving_person_object_obligatory_transitive, ved_person_object_obligatory_transitive, ven_person_object_obligatory_transitive, {"subject" : "person", "object" : "inanimate"}]


allv_can_take_clausal = ["see", "sees", "seeing", "saw", "seen", "remember", "remembers", "remembering", "remembered", "forget", "forgets", "forgetting", "forgot", "forgotten", "notice", "notices", "noticing", "noticed", "know", "knows", "knowing", "knew", "known", "observe", "observes", "observing", "observed", "hear", "hears", "hearing", "heard", "appreciate", "appreciates", "appreciating", "appreciated", "worry", "worries", "worrying", "worried", "understand", "understands", "understanding", "understood", "buy", "buys", "buying", "bought", "observe", "observes", "observing", "observed", "find", "finds", "finding", "found", "teach", "teaches", "teaching", "taught"]
dict_v_can_take_clausal = {}
for verb in allv_can_take_clausal:
    dict_v_can_take_clausal[verb] = 1

v_wh_object = ["believe", "know", "forget", "learn", "remember", "realize", "figure out", "say", "think", "hope"]
vs_wh_object = ["believes", "knows", "forgets", "learns", "remembers", "realizes", "figures out", "says", "thinks", "hopes"]
ving_wh_object = ["forgetting", "learning", "figuring out", "saying", "thinking", "hoping"]
ved_wh_object = ["believed", "knew", "forgot", "learned", "remembered", "realized", "figured out", "said", "thought", "hoped"]
ven_wh_object = ["believed", "known", "forgotten", "learned", "remembered", "realized", "figured out", "said", "thought", "hoped"]
allv_wh_object = [v_wh_object, vs_wh_object, ving_wh_object, ved_wh_object, ven_wh_object, {}]
allv_wh_object = [v_wh_object, vs_wh_object, ving_wh_object, ved_wh_object, ven_wh_object, {}]

v_wh_or_np = ["know", "forget", "remember", "notice", "discover", "see", "learn", "realize"]
vs_wh_or_np = ["knows", "forgets", "remembers", "notices", "discovers", "sees", "learns", "realizes"]
ving_wh_or_np = ["forgetting", "remembering", "noticing", "discovering"]
ved_wh_or_np = ["knew", "forgot", "remembered", "noticed", "discovered", "saw", "learned", "realized"]
ven_wh_or_np = ["known", "forgotten", "remembered", "noticed", "discovered", "seen", "learned", "realized"]
allv_wh_or_np = [v_wh_or_np, vs_wh_or_np, ving_wh_or_np, ved_wh_or_np, ven_wh_or_np, {}]

v_wh_or_person = ["know", "forget", "remember", "notice", "discover", "see"]
vs_wh_or_person = ["knows", "forgets", "remembers", "notices", "discovers", "sees"]
ving_wh_or_person = ["forgetting", "remembering", "noticing", "discovering"]
ved_wh_or_person = ["knew", "forgot", "remembered", "noticed", "discovered", "saw"]
ven_wh_or_person = ["known", "forgotten", "remembered", "noticed", "discovered", "seen"]
allv_wh_or_person = [v_wh_or_person, vs_wh_or_person, ving_wh_or_person, ved_wh_or_person, ven_wh_or_person, {}]



v_clausal = ["believe", "discover", "know", "forget", "learn", "remember", "realize", "say", "think", "explain", "learn", "notice", "shout", "whisper", "argue", "suggest", "hope", "admit"]
vs_clausal = ["believes", "discovers", "knows", "forgets", "learns", "remembers", "realizes", "says", "thinks", "explains", "learns", "notices", "shouts", "whispers", "argues", "suggests", "hopes", "admits"]
ving_clausal = ["discovering", "forgetting", "learning", "remembering", "realizing", "saying", "thinking", "explaining", "learning", "noticing", "shouting", "whispering", "arguing", "suggesting", "hoping", "admitting"]
ved_clausal = ["believed", "discovered", "knew", "forgot", "learned", "remembered", "realized", "said", "thought", "explained", "learned", "noticed", "shouted", "whispered", "argued", "suggested", "hoped", "admitted"]
ven_clausal = ["believed", "discovered", "forgotten", "learned", "remembered", "realized", "said", "thought", "explained", "learned", "noticed", "shouted", "whispered", "argued", "suggested", "hoped", "admitted"]
allv_clausal = [v_clausal, vs_clausal, ving_clausal, ved_clausal, ven_clausal, {}]

v_ditransitive = ["show", "give", "offer", "sell", "buy", "send", "mail"]
vs_ditransitive = ["shows", "gives", "offers", "sells", "buys", "sends", "mails"]
ving_ditransitive = ["showing", "giving", "offering", "selling", "buying", "sending", "mailing"]
ved_ditransitive = ["showed", "gave", "offered", "sold", "bought", "sent", "mailed"]
ven_ditransitive = ["shown", "given", "offered", "sold", "bought", "sent", "mailed"]
allv_ditransitive = [v_ditransitive, vs_ditransitive, ving_ditransitive, ved_ditransitive, ven_ditransitive, ven_ditransitive, {}]

v_not_anaphor = ["say", "try", "begin", "run", "write", "continue", "learn", "read", "build"]
vs_not_anaphor = ["says", "trys", "begins", "runs", "writes", "continues", "learns", "reads", "builds"]
ving_not_anaphor = ["saying", "trying", "beginning", "running", "writing", "continuing", "learning", "reading", "building"]
ved_not_anaphor = ["said", "trid", "began", "ran", "wrote", "continued", "learned", "read", "built"]
ven_not_anaphor = ["said", "tried", "begun", "run", "written", "continued", "learned", "read", "built"]
allv_not_anaphor = [v_not_anaphor, vs_not_anaphor, ving_not_anaphor, ved_not_anaphor, ven_not_anaphor, {}]

vs_locative = ["sits", "rests"]
ving_locative = ["sitting", "resting"]



v_category_to_list = {}
v_category_to_list["animate_physent"] = allv_animate_physent
v_category_to_list["person_physent"] = allv_person_physent
v_category_to_list["animate_animate"] = allv_animate_animate
v_category_to_list["person_person"] = allv_person_person
v_category_to_list["person_place"] = allv_person_place
v_category_to_list["person_object"] = allv_person_object
v_category_to_list["physent_object"] = allv_physent_object
v_category_to_list["intrans_person"] = allv_intrans_person
v_category_to_list["intrans_animate"] = allv_intrans_animate
v_category_to_list["intrans_inanimate"] = allv_intrans_inanimate
v_category_to_list["intrans_physent"] = allv_intrans_physent
v_category_to_list["person_person_obl"] = allv_person_person_obligatory_transitive
v_category_to_list["person_object_obl"] = allv_person_object_obligatory_transitive






###################################################################
# Vocabulary: Auxiliaries
###################################################################

aux_types_sg = ["inf_sg", "vs", "ving_sg", "ven_sg", "pst_sg"]
aux_types_pl = ["inf_pl", "v", "ving_pl", "ven_pl", "pst_pl"]
auxes = {}
auxes["inf_sg"] = ["might", "must", "will", "can", "may", "would", "should", "could", "does n't", "did n't", "wo n't", "can n't", "would n't", "could n't", "should n't"]
auxes["inf_pl"] = ["might", "must", "will", "can", "may", "would", "should", "could", "do n't", "did n't", "wo n't", "can n't", "would n't", "could n't", "should n't"]
auxes["ving_sg"] = ["is", "was", "is n't", "was n't"]
auxes["ving_pl"] = ["are", "were", "are n't", "were n't"]
auxes["ven_sg"] = ["has", "had", "has n't", "had n't"]
auxes["ven_pl"] = ["have", "had", "have n't", "had n't"]

auxes["inf_sg_nonneg"] = ["might", "must", "will", "can", "may", "would", "should", "could"] 
auxes["inf_pl_nonneg"] = ["might", "must", "will", "can", "may", "would", "should", "could"] 
auxes["ving_sg_nonneg"] = ["is", "was"] 
auxes["ving_pl_nonneg"] = ["are", "were"] 
auxes["ven_sg_nonneg"] = ["has", "had"] 
auxes["ven_pl_nonneg"] = ["have", "had"] 
aux_type_nonnegative = ["inf_sg_nonneg", "inf_pl_nonneg", "ving_sg_nonneg", "ving_pl_nonneg", "ven_sg_nonneg", "ven_pl_nonneg", "vs", "v"]
aux_type_nonnegative_sg = ["inf_sg_nonneg", "ving_sg_nonneg", "ven_sg_nonneg", "vs"]
aux_type_nonnegative_pl = ["inf_pl_nonneg", "ving_pl_nonneg", "ven_pl_nonneg", "v"]

all_auxes_oneword = ["might", "must", "will", "can", "may", "would", "should", "could", "is", "was", "are", "were", "has", "have", "had", "do", "does", "did"]

aux_qinv_sg = ["inf_sg", "ving_sg", "ven_sg"]
aux_qinv_pl = ["inf_pl", "ving_pl", "ven_pl"]
auxes_qinv = {}
auxes_qinv["inf_sg"] = ["will", "can", "would", "should", "could", "does", "did"]
auxes_qinv["inf_pl"] = ["will", "can", "would", "should", "could", "do", "did"]
auxes_qinv["ving_sg"] = ["is", "was"]
auxes_qinv["ving_pl"] = ["are", "were"]
auxes_qinv["ven_sg"] = ["has", "had"]
auxes_qinv["ven_pl"] = ["have", "had"]

aux_qclause_sg = ["inf_sg", "ving_sg", "ven_sg"]
aux_qclause_pl = ["inf_pl", "ving_pl", "ven_pl"]
auxes_qclause = {}
auxes_qclause["inf_sg"] = ["will", "can", "would", "should", "could", "does", "did", "wo n't", "can n't", "would n't", "should n't", "could n't", "does n't", "did n't"]
auxes_qclause["inf_pl"] = ["will", "can", "would", "should", "could", "do", "did", "wo n't", "can n't", "would n't", "should n't", "could n't", "do n't", "did n't"]
auxes_qclause["ving_sg"] = ["is", "was", "is n't", "was n't"]
auxes_qclause["ving_pl"] = ["are", "were", "are n't", "were n't"]
auxes_qclause["ven_sg"] = ["has", "had", "has n't", "had n't"]
auxes_qclause["ven_pl"] = ["have", "had", "have n't", "had n't"]

auxes_no_number = ["will", "can", "would", "should", "could", "did", "wo n't", "can n't", "would n't", "should n't", "could n't", "did n't"]



###################################################################
# Vocabulary: Adverbs
###################################################################

adv_manner = ["rarely", "happily", "slowly", "calmly", "elegantly", "honestly", "lazily", "boldly", "perfectly", "loudly", "quietly", "quickly", "carefully"]
adv_time = ["rarely", "never", "often", "always", "sometimes"]
adv_opinion = ["fortunately", "unfortunately", "luckily", "really", "probably", "possibly", "definitely", "potentially"] 


###################################################################
# Vocabulary: Prepositions
###################################################################

preps = ["next to", "near", "by", "in front of", "behind"]
prep_ing_adjunct = ["after", "without", "while", "before"]





###################################################################
# Functions for creating phrases
###################################################################

def create_matrix_clause(aux_types=aux_types_sg + aux_types_pl, v_category=None, qinv=False, object_number=None, implausible=False):
    if v_category is None:
        if implausible:
            v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object", "intrans_person", "intrans_person", "intrans_person", "intrans_person", "intrans_animate"])
        else:
            v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object", "intrans_person", "intrans_person", "intrans_person", "intrans_person", "intrans_animate", "intrans_animate", "intrans_inanimate", "intrans_inanimate", "intrans_physent"])

    verb_list = v_category_to_list[v_category]
    aux_type = random.choice(aux_types)
    if aux_type.replace("_nonneg", "") in aux_types_sg + aux_qinv_sg:
        number = "sg"
        subject_det = random.choice(det_sg)
    else:
        number = "pl"
        subject_det = random.choice(det_pl)

    if qinv:
        verb = inflect_qinv_verb_group(verb_list, number=number, v_type=aux_type)
    else:
        verb = inflect_verb_group(verb_list, number=number, v_type=aux_type)
    subject_type = verb_list[5]["subject"]
    if number == "sg":
        if implausible:
            subj = random.choice(selectional_categories_implausible[subject_type][0])
        else:
            subj = random.choice(selectional_categories[subject_type][0])
    else:
        if implausible:
            subj = random.choice(selectional_categories_implausible[subject_type][1])
        else:
            subj = random.choice(selectional_categories[subject_type][1])
    subj = subject_det + " " + subj

    if "object" in verb_list[5]:
        object_type = verb_list[5]["object"]
        if object_number is None:
            object_number = random.choice(["sg", "pl"])

        if object_number == "sg":
            if implausible:
                obj = random.choice(selectional_categories_implausible[object_type][0])
            else:
                obj = random.choice(selectional_categories[object_type][0])
            obj_det = random.choice(det_sg)
        else:
            if implausible:
                obj = random.choice(selectional_categories_implausible[object_type][1])
            else:
                obj = random.choice(selectional_categories[object_type][1])
            obj_det = random.choice(det_pl)

        obj = obj_det + " " + obj
    else:
        object_type = None
        object_number = None
        obj = None

    to_return = {"subject" : subj, "verb" : verb, "object" : obj, "subject_type" : subject_type, "object_type" : object_type, "subject_number" : number, "object_number" : object_number, "verb_category" : v_category, "aux_type" : aux_type}

    return to_return


def filter_selections(selections, position):
    new_selections = []
    for selection in selections:
        if selection[1] == position:
            new_selections.append(selection)

    return new_selections


def create_rc(noun_type, number, aux_types=None, only_transitive=False, position=None, object_number=None, implausible=False):
    if implausible:
        if noun_type == "animate":
            if only_transitive:
                selections = [("animate_animate", "subj"), ("animate_animate", "obj")]
            else:
                selections = [("animate_animate", "subj"), ("animate_animate", "obj"), ("intrans_animate", "subj")]
        elif noun_type == "person":
            if only_transitive:
                selections = [("person_object", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj"), ("person_person", "subj"), ("person_person", "obj")]
            else:
                selections = [("person_object", "subj"), ("intrans_person", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj")]
        elif noun_type == "inanimate":
            if only_transitive:
                selections = [("person_object", "obj")]
            else:
                selections = [("person_object", "obj"), ("animate_physent", "obj")]
        elif noun_type == "double_person_subj":
            selections = [("person_person", "subj")]

    else:
        if noun_type == "animate":
            if only_transitive:
                selections = [("animate_physent", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj"), ("person_physent", "obj")]
            else:
                selections = [("animate_physent", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj"), ("intrans_animate", "subj"), ("person_physent", "obj")]
        elif noun_type == "physent":
            if only_transitive:
                selections = [("animate_physent", "obj"), ("person_physent", "obj")]
            else:
                selections = [("animate_physent", "obj"), ("person_physent", "obj"), ("intrans_physent", "subj")]
        elif noun_type == "person":
            if only_transitive:
                selections = [("person_physent", "subj"), ("person_place", "subj"), ("person_object", "subj"), ("animate_physent", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj"), ("person_physent", "obj"), ("person_person", "subj"), ("person_person", "obj")]
            else:
                selections = [("person_physent", "subj"), ("person_place", "subj"), ("person_object", "subj"), ("intrans_person", "subj"), ("animate_physent", "subj"), ("animate_animate", "subj"), ("animate_animate", "obj"), ("person_physent", "obj")]
        elif noun_type == "place":
            selections = [("person_place", "obj")]
        elif noun_type == "inanimate":
            if only_transitive:
                selections = [("person_object", "obj"), ("animate_physent", "obj"), ("person_physent", "obj")]
            else:
                selections = [("intrans_inanimate", "subj"), ("person_object", "obj"), ("animate_physent", "obj"), ("person_physent", "obj"), ("intrans_physent", "subj")]
        elif noun_type == "double_person_subj":
            selections = [("person_person", "subj")]

    if position is None:
        verb_type, position = random.choice(selections)
    else:
        selections = filter_selections(selections, position)
        verb_type, _ = random.choice(selections)

    if aux_types is None:
        if position == "obj":
            aux_types = aux_types_sg + aux_types_pl
        else:
            if number == "sg":
                aux_types = aux_types_sg
            else:
                aux_types = aux_types_pl

    if noun_type.startswith("double"):
        clause = create_matrix_clause(aux_types=aux_types, v_category=verb_type, object_number=number, implausible=implausible)
    else:
        clause = create_matrix_clause(aux_types=aux_types, v_category=verb_type, object_number=object_number, implausible=implausible)

    subj = clause["subject"]
    verb = clause["verb"]
    obj = clause["object"]

    if noun_type == "person" or noun_type == "double_person_subj":
        relativizer = "who"
        wh_word = "who"
    elif noun_type == "place":
        relativizer = "that"
        wh_word = "where"
    else:
        relativizer = "that"
        wh_word = "what"

    to_return = {}
    to_return["relativizer"] = relativizer
    to_return["verb"] = verb
    to_return["wh_word"] = wh_word

    if position == "subj":
        if obj is None:
            to_return["subject"] = None
            to_return["object"] = None
        else:
            to_return["subject"] = None
            to_return["object"] = obj
            to_return["object_number"] = clause["object_number"]
            to_return["object_type"] = clause["object_type"]
    else:
        to_return["subject"] = subj
        to_return["subject_number"] = clause["subject_number"]
        to_return["subject_type"] = clause["subject_type"]
        to_return["object"] = None

    return to_return

def stringify_rc(rc):

    if rc["subject"] is None:
        if rc["object"] is None:
            return " ".join([rc["relativizer"], rc["verb"]])
        else:
            return " ".join([rc["relativizer"], rc["verb"], rc["object"]])
    else:
        return " ".join([rc["relativizer"], rc["subject"], rc["verb"]])


def create_opposite_number_rc(noun_type, number, no_clausal=False, implausible=False):

    if noun_type in ["inanimate", "physent", "place"]:
        position = "obj"
    else:
        position = random.choice(["subj", "obj"])

    if position == "subj":
        aux_types = None
        if number == "sg":
            object_number = "pl"
        else:
            object_number = "sg"
    else:
        object_number = None
        if number == "sg":
            aux_types = aux_types_pl
        else:
            aux_types = aux_types_sg

    rc_found = False
    while not rc_found:
        rc = create_rc(noun_type, number, aux_types=aux_types, only_transitive=True, position=position, object_number=object_number, implausible=implausible)
        aux, v = v_to_aux_v(rc["verb"])
        if no_clausal:
            if v not in dict_v_can_take_clausal:
                rc_found = True
        else:
            rc_found = true

    return rc


def create_opposite_number_pp(noun_type, number, implausible=False):
    
    prep = random.choice(preps)

    det = "the"
   
    if number == "pl":
        pobj = random.choice(selectional_categories[noun_type][0])
    else:
        pobj = random.choice(selectional_categories[noun_type][1])
    pp = prep + " " + det + " " + pobj

    return pp


in_nouns = nouns_place_sg
on_nouns = ["floor", "table", "chair", "bed", "stool", "couch", "ground", "street"]
by_nouns = nouns_inanimate_sg
def create_pp_locative():
    prep = random.choice(["in", "on", "by"])

    if prep == "in":
        noun = random.choice(in_nouns)
    elif prep == "on":
        noun = random.choice(on_nouns)
    else:
        noun = random.choice(by_nouns)

    pp = prep + " the " + noun
    return pp


 
def person_dp(number="sg", name_allowed=False, gender=None, implausible=False):

    if implausible:
        category = random.choice(["object"])

        if category == "object":
            if number == "sg":
                det = random.choice(det_sg)
                noun = random.choice(filter_freq(nouns_inanimate_sg))
            else:
                det = random.choice(det_pl)
                noun = random.choice(filter_freq(nouns_inanimate_pl))
    else:
        if name_allowed and number == "sg":
            category = random.choice(["relation", "person", "name"])
        else:
            category = random.choice(["relation", "person"])
    
        if category == "relation":
            det = random.choice(["my", "your"])
        elif number == "sg":
            det = random.choice(det_sg)
        else:
            det = random.choice(det_pl)

        if gender is None:
            if number == "sg":
                if category == "relation":
                    noun = random.choice(filter_freq(relation_sg))
                elif category ==  "person":
                    noun = random.choice(filter_freq(person_sg))
                else:
                    noun = random.choice(filter_freq(male_names) + filter_freq(female_names))
            else:
                if category == "relation":
                    noun = random.choice(filter_freq(relation_pl))
                elif category == "person":
                    noun = random.choice(filter_freq(person_pl))
        elif gender == "m":
            if number == "sg":
                if category == "relation":
                    noun = random.choice(filter_freq(male_relation_sg))
                elif category ==  "person":
                    noun = random.choice(filter_freq(male_person_sg))
                else:
                    noun = random.choice(filter_freq(male_names))
            else:
                if category == "relation":
                    noun = random.choice(filter_freq(male_relation_pl))
                else:
                    noun = random.choice(filter_freq(male_person_pl))
        else:
            if number == "sg":
                if category == "relation":
                    noun = random.choice(filter_freq(female_relation_sg))
                elif category ==  "person":
                    noun = random.choice(filter_freq(female_person_sg))
                else:
                    noun = random.choice(filter_freq(female_names))
            else:
                if category == "relation":
                    noun = random.choice(filter_freq(female_relation_pl))
                else:
                    noun = random.choice(filter_freq(female_person_pl))


    if category == "name":
        return noun
    else:
        return det + " " + noun


def male_dp():
    categories = ["name", "name", "name", "person", "relation", "relation"]
    category = random.choice(categories)

    if category == "name":
        return random.choice(male_names)
    elif category == "person":
        return random.choice(det_sg) + " " + random.choice(male_person_sg)
    else:
        return random.choice(["my", "your"]) + " " + random.choice(male_relation_sg)


def female_dp():
    categories = ["name", "name", "name", "person", "relation", "relation"]
    category = random.choice(categories)

    if category == "name":
        return random.choice(female_names)
    elif category == "person":
        return random.choice(det_sg) + " " + random.choice(female_person_sg)
    else:
        return random.choice(["my", "your"]) + " " + random.choice(female_relation_sg)


   
###################################################################
# Singular/plural agreement pairs
###################################################################

def noun_sg_pl_pair(noun_type, implausible=False):
    if implausible:
        pairs = list(zip(selectional_categories_implausible[noun_type][0], selectional_categories_implausible[noun_type][1]))
    else:
        pairs = list(zip(selectional_categories[noun_type][0], selectional_categories[noun_type][1]))
    return random.choice(pairs)

def verb_sg_pl_pair(verb_group):
    pairs = list(zip(v_category_to_list[verb_group][1], v_category_to_list[verb_group][0]))
    return random.choice(pairs)


def aux_sg_pl_pair(verb_group):
    if random.choice([True, False, False]):
        auxes = random.choice([["is", "are"], ["is n't", "are n't"], ["was", "were"], ["was n't", "were n't"]])
        verb = random.choice(v_category_to_list[verb_group][2])
    elif random.choice([True, False]):
        auxes = random.choice([["has", "have"], ["has n't", "have n't"]])
        verb = random.choice(v_category_to_list[verb_group][4])
    else:
        auxes = random.choice([["does", "do"], ["does n't", "do n't"]])
        verb = random.choice(v_category_to_list[verb_group][0])
        
    return auxes[0] + " " + verb, auxes[1] + " " + verb





###################################################################
# Functions for post-processing phrases and sentences
###################################################################


def join_verb_groups(group_list):
    new_group = group_list[0][:-1]

    for group in group_list[1:-1]:
        for i in range(len(new_group)):
            new_group[i] = new_group[i][:] + group[i]

    return new_group

def inflect_verb_group(verb_group, number=None, v_type=None):
    
    if v_type is None:
        if number is None:
            v_type = random.choice(aux_types_sg + aux_types_pl)
        elif number == "sg":
            v_type = random.choice(aux_types_sg)
        else:
            v_type = random.choice(aux_types_pl)
    
    v_type_orig = v_type
    v_type = v_type.replace("_nonneg", "")
    if v_type == "v":
        verb = random.choice(verb_group[0])
    elif v_type == "vs":
        verb = random.choice(verb_group[1])
    elif v_type == "pst_sg" or v_type == "pst_pl":
        verb = random.choice(verb_group[3])
    else:
        aux = random.choice(auxes[v_type_orig])

        if v_type == "inf_sg" or v_type == "inf_pl":
            verb = random.choice(verb_group[0])
        elif v_type == "ving_sg" or v_type == "ving_pl":
            verb = random.choice(verb_group[2])
        elif v_type == "ven_sg" or v_type == "ven_pl":
            verb = random.choice(verb_group[4])

        verb = aux + " " + verb

    return verb

def inflect_qinv_verb_group(verb_group, number=None, v_type=None):
    
    if v_type is None:
        if number is None:
            v_type = random.choice(aux_qinv_sg + aux_qinv_pl)
        elif number == "sg":
            v_type = random.choice(aux_qinv_sg)
        else:
            v_type = random.choice(aux_qinv_pl)

    if v_type == "v":
        verb = random.choice(verb_group[0])
    elif v_type == "vs":
        verb = random.choice(verb_group[1])
    elif v_type == "pst_sg" or v_type == "pst_pl":
        verb = random.choice(verb_group[3])
    else:
        aux = random.choice(auxes_qinv[v_type])

        if v_type == "inf_sg" or v_type == "inf_pl":
            verb = random.choice(verb_group[0])
        elif v_type == "ving_sg" or v_type == "ving_pl":
            verb = random.choice(verb_group[2])
        elif v_type == "ven_sg" or v_type == "ven_pl":
            verb = random.choice(verb_group[4])

        verb = aux + " " + verb

    return verb

def v_to_aux_v(v):
    parts = v.split()
    if "n't" in v:
        aux = " ".join(parts[:2])
        just_v = " ".join(parts[2:])
    elif parts[0] in all_auxes_oneword:
        aux = " ".join(parts[:1])
        just_v = " ".join(parts[1:])
    else:
        aux = ""
        just_v = v

    return aux, just_v


def a_or_an(noun):
    if noun[0].lower() in ["a", "e", "i", "o", "u"]:
        return "an"
    return "a"

















######################################################################################
# GENERATING MINIMAL PAIRS
######################################################################################


###########################################
# Anaphor agreement
###########################################

def anaphor_gender_agreement(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = join_verb_groups([allv_not_anaphor])
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            anaphor = random.choice(["herself", "himself"])
            verb = inflect_verb_group(verb_candidates, number="sg")

            if anaphor == "herself":
                subj = female_dp()
            else:
                subj = male_dp()

            s1 = " ".join([subj, verb, anaphor, "."])
            if anaphor == "herself":
                s2 = " ".join([subj, verb, "himself", "."])
            else:
                s2 = " ".join([subj, verb, "herself", "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def anaphor_number_agreement(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = join_verb_groups([allv_person_person])
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            verb = inflect_verb_group(verb_candidates, number=number)

            subj = person_dp(number=number, implausible=implausible)

            if number == "sg":
                correct_anaphor = random.choice(["himself", "herself"])
                incorrect_anaphor = "themselves"
            else:
                correct_anaphor = "themselves"
                incorrect_anaphor = random.choice(["himself", "herself"])


            s1 = " ".join([subj, verb, correct_anaphor, "."])
            s2 = " ".join([subj, verb, incorrect_anaphor, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


###########################################
# Negative Polarity Items
############################################

def matrix_question_npi_licensor_present(n_examples=1000, implausible=False):


    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]
            if "n't" in v:
                continue

            v_parts = v.split()
            aux = v_parts[0]
            just_v = " ".join(v_parts[1:])

            if aux in auxes["inf_sg"]:
                aux = random.choice(["will", "can", "could", "would", "should"])
            elif aux in auxes["inf_pl"]:
                aux = random.choice(["will", "can", "could", "would", "should"])

            if o is None:
                s1 = " ".join([aux, s, "ever", just_v, "?"])
                s2 = " ".join([s, aux, "ever", just_v, "."])
            else:
                s1 = " ".join([aux, s, "ever", just_v, o, "?"])
                s2 = " ".join([s, aux, "ever", just_v, o, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def npi_present(n_examples=1000, implausible=False):


    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]
            if "n't" in v:
                continue

            v_parts = v.split()
            aux = v_parts[0]
            just_v = " ".join(v_parts[1:])

            if aux in auxes["inf_sg"]:
                aux = random.choice(["will", "can", "could", "would", "should"])
            elif aux in auxes["inf_pl"]:
                aux = random.choice(["will", "can", "could", "would", "should"])

            adv = random.choice(adv_time)

            if o is None:
                s1 = " ".join([s, aux, adv, just_v, "."])
                s2 = " ".join([s, aux, "ever", just_v, "."])
            else:
                s1 = " ".join([s, aux, adv, just_v, o, "."])
                s2 = " ".join([s, aux, "ever", just_v, o, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def only_npi_licensor_present(n_examples=1000, implausible=False):


    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            if "n't" in v:
                continue

            v_parts = v.split()
            aux = v_parts[0]
            just_v = " ".join(v_parts[1:])

            if aux in auxes["inf_sg"]:
                aux = random.choice(["will", "can", "could", "would", "should"])
            elif aux in auxes["inf_pl"]:
                aux = random.choice(["will", "can", "could", "would", "should"])

            if o is None:
                s1 = " ".join(["only", s, aux, "ever", just_v, "."])
                s2 = " ".join(["even", s, aux, "ever", just_v, "."])
            else:
                s1 = " ".join(["only", s, aux, "ever", just_v, o, "."])
                s2 = " ".join(["even", s, aux, "ever", just_v, o, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def sentential_negation_npi_licensor_present(n_examples=1000, implausible=False):


    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            v_parts = v.split()
            aux = " ".join(v_parts[:2])
            just_v = " ".join(v_parts[2:])

            if "n't" not in aux or "wo " in aux:
                continue


            aux_positive = aux.split()[0] + " " + random.choice(adv_opinion)

            if o is None:
                s1 = " ".join([s, aux, "ever", just_v, "."])
                s2 = " ".join([s, aux_positive, "ever", just_v, "."])
            else:
                s1 = " ".join([s, aux, "ever", just_v, o, "."])
                s2 = " ".join([s, aux_positive, "ever", just_v, o, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def only_npi_scope(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            subject_type = clause["subject_type"]

            if "n't" in v:
                continue

            if clause["subject_number"] == "sg":
                rc_aux_types = ["inf_sg", "ven_sg"]
            else:
                rc_aux_types = ["inf_pl", "ven_pl"]

            rc = create_rc(subject_type, clause["subject_number"], aux_types=rc_aux_types, implausible=implausible)
            if "n't" in rc["verb"]:
                continue

            v_parts = v.split()
            aux = v_parts[0]
            just_v = " ".join(v_parts[1:])


            if aux in auxes["inf_sg"]:
                aux = random.choice(["will", "can", "could", "would", "should"])
            elif aux in auxes["inf_pl"]:
                aux = random.choice(["will", "can", "could", "would", "should"])

            rc_verb_parts = rc["verb"].split()
            rc["aux"] = rc_verb_parts[0]
            rc["just_verb"] = " ".join(rc_verb_parts[1:])

            if rc["subject"] is None and rc["object"] is None:
                s1 = ["only", s, rc["relativizer"], rc["verb"], aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], rc["aux"], "only", rc["just_verb"], aux, "ever", just_v, "."]
            elif rc["subject"] is None:
                s1 = ["only", s, rc["relativizer"], rc["verb"], rc["object"], aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], rc["aux"], "only", rc["just_verb"], rc["object"], aux, "ever", just_v, "."]
            else:
                s1 = ["only", s, rc["relativizer"], rc["subject"], rc["verb"], aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], "only", rc["subject"], rc["verb"], aux, "ever", just_v, "."]

            if o is not None:
                s1 = s1[:-1] + [o] + ["."]
                s2 = s2[:-1] + [o] + ["."]

            s1 = " ".join(s1)
            s2 = " ".join(s2)

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def sentential_negation_npi_scope(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            subject_type = clause["subject_type"]

            v_parts = v.split()
            aux = " ".join(v_parts[:2])
            just_v = " ".join(v_parts[2:])

            if ("n't" not in aux) or (aux == "wo n't"):
                continue

            aux_positive = aux.split()[0]


            if clause["subject_number"] == "sg":
                rc_aux_types = ["inf_sg", "ven_sg"]
            else:
                rc_aux_types = ["inf_pl", "ven_pl"]

            rc = create_rc(subject_type, clause["subject_number"], aux_types=rc_aux_types, implausible=implausible)

            rc_v_parts = rc["verb"].split()
            rc_aux = " ".join(rc_v_parts[:2])
            rc_just_v = " ".join(rc_v_parts[2:])
            if ("n't" not in rc_aux) or (rc_aux == "wo n't"):
                continue

            rc_aux_positive = rc_aux.split()[0] 

            if rc["subject"] is None and rc["object"] is None:
                s1 = [s, rc["relativizer"], rc_aux_positive, rc_just_v, aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], rc["verb"], aux_positive, "ever", just_v, "."]
            elif rc["subject"] is None:
                s1 = [s, rc["relativizer"], rc_aux_positive, rc_just_v, rc["object"], aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], rc["verb"], rc["object"], aux_positive, "ever", just_v, "."]
            else:
                s1 = [s, rc["relativizer"], rc["subject"], rc_aux_positive, rc_just_v, aux, "ever", just_v, "."]
                s2 = [s, rc["relativizer"], rc["subject"], rc["verb"], aux_positive, "ever", just_v, "."]

            if o is not None:
                s1 = s1[:-1] + [o] + ["."]
                s2 = s2[:-1] + [o] + ["."]

            s1 = " ".join(s1)
            s2 = " ".join(s2)

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



##########################################
# Argument structure
##########################################

def intransitive(n_examples=1000, implausible=False):

    intransitive_candidates = allv_person_obligatory_intransitive
    transitive_candidates = join_verb_groups([allv_person_person_obligatory_transitive, allv_person_object_obligatory_transitive])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            
            verb_i = inflect_verb_group(intransitive_candidates, number=number)
            verb_t = inflect_verb_group(transitive_candidates, number=number)

            if number == "sg":
                if implausible:
                    subject = random.choice(nouns_inanimate_sg)
                else:
                    subject = random.choice(nouns_person_sg)
                subject_det = random.choice(det_sg)
                v_type = random.choice(aux_types_sg)
            else:
                if implausible:
                    subject = random.choice(nouns_inanimate_pl)
                else:
                    subject = random.choice(nouns_person_pl)
                subject_det = random.choice(det_pl)
                v_type = random.choice(aux_types_pl)

            verb_i = inflect_verb_group(intransitive_candidates, number=number, v_type=v_type)
            verb_t = inflect_verb_group(transitive_candidates, number=number, v_type=v_type)

            if "n't" in verb_i:
                verb_i_aux = " ".join(verb_i.split()[:2])
                verb_i_just_verb = " ".join(verb_i.split()[2:])
            elif v_type not in ["vs", "v", "pst_sg", "pst_pl"]:
                verb_i_aux = " ".join(verb_i.split()[:1])
                verb_i_just_verb = " ".join(verb_i.split()[1:])
            else:
                verb_i_aux = ""
                verb_i_just_verb = verb_i


            if "n't" in verb_t:
                verb_t_aux = " ".join(verb_t.split()[:2])
                verb_t_just_verb = " ".join(verb_t.split()[2:])
            elif v_type not in ["vs", "v", "pst_sg", "pst_pl"]:
                verb_t_aux = " ".join(verb_t.split()[:1])
                verb_t_just_verb = " ".join(verb_t.split()[1:])
            else:
                verb_t_aux = ""
                verb_t_just_verb = verb_t

            if verb_i_aux == "":
                verb_t = verb_t_just_verb
            else:
                verb_t = verb_i_aux + " " + verb_t_just_verb


            s1 = " ".join([subject_det, subject, verb_i, "."])
            s2 = " ".join([subject_det, subject, verb_t, "."])

            

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples






def transitive(n_examples=1000, implausible=False):

    intransitive_candidates = allv_person_obligatory_intransitive
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
 
            if number == "sg":
                if implausible:
                    subject = random.choice(nouns_inanimate_sg)
                else:
                    subject = random.choice(nouns_person_sg)
                subject_det = random.choice(det_sg)
                v_type = random.choice(aux_types_sg)
            else:
                if implausible:
                    subject = random.choice(nouns_inanimate_pl)
                else:
                    subject = random.choice(nouns_person_pl)
                subject_det = random.choice(det_pl)
                v_type = random.choice(aux_types_pl)

           
            
            obj_number = random.choice(["sg", "pl"])
            if random.choice([True, False]):
                verb_t = inflect_verb_group(allv_person_person_obligatory_transitive, number=number, v_type=v_type)
 
                if obj_number == "sg":
                    if implausible:
                        obj = random.choice(nouns_inanimate_sg)
                    else:
                        obj = random.choice(nouns_person_sg)
                    obj_det = random.choice(det_sg)
                else:
                    if implausible:
                        obj = random.choice(nouns_inanimate_pl)
                    else:
                        obj = random.choice(nouns_person_pl)
                    obj_det = random.choice(det_pl)

            
            else:
                verb_t = inflect_verb_group(allv_person_object_obligatory_transitive, number=number, v_type=v_type)
 
                if obj_number == "sg":
                    if implausible:
                        obj = random.choice(nouns_person_sg)
                    else:
                        obj = random.choice(nouns_inanimate_sg)
                    obj_det = random.choice(det_sg)
                else:
                    if implausible:
                        obj = random.choice(nouns_person_pl)
                    else:
                        obj = random.choice(nouns_inanimate_pl)
                    obj_det = random.choice(det_pl)

            verb_i = inflect_verb_group(intransitive_candidates, number=number, v_type=v_type)

            if "n't" in verb_i:
                verb_i_aux = " ".join(verb_i.split()[:2])
                verb_i_just_verb = " ".join(verb_i.split()[2:])
            elif v_type not in ["vs", "v", "pst_sg", "pst_pl"]:
                verb_i_aux = " ".join(verb_i.split()[:1])
                verb_i_just_verb = " ".join(verb_i.split()[1:])
            else:
                verb_i_aux = ""
                verb_i_just_verb = verb_i


            if "n't" in verb_t:
                verb_t_aux = " ".join(verb_t.split()[:2])
                verb_t_just_verb = " ".join(verb_t.split()[2:])
            elif v_type not in ["vs", "v", "pst_sg", "pst_pl"]:
                verb_t_aux = " ".join(verb_t.split()[:1])
                verb_t_just_verb = " ".join(verb_t.split()[1:])
            else:
                verb_t_aux = ""
                verb_t_just_verb = verb_t

            if verb_i_aux == "":
                verb_t = verb_t_just_verb
            else:
                verb_t = verb_i_aux + " " + verb_t_just_verb


            s1 = " ".join([subject_det, subject, verb_t, obj_det, obj, "."])
            s2 = " ".join([subject_det, subject, verb_i, obj_det, obj, "."])

            

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


#############################
# Islands
#############################

def left_branch_island_simple_question(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            clause = create_matrix_clause(qinv=True, aux_types=aux_qinv_sg+aux_qinv_pl, v_category=v_category, implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            o_det = o.split()[0]
            o_rest = " ".join(o.split()[1:])
            
            wh_det = random.choice(["which", "what"])

            aux, v = v_to_aux_v(v)

            
            s1 = " ".join([wh_det, o_rest, aux, s, v, "?"]) 
            s2 = " ".join([wh_det, aux, s, v, o_rest, "?"]) 

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def coordinate_structure_constrain_object_extraction(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            main_aux = random.choice(auxes_no_number)
            
            v = random.choice(v_person_person)
            number1 = random.choice(["sg", "pl"])
            number2 = random.choice(["sg", "pl"])

            subj1 = person_dp(number=number1, name_allowed=True, implausible=implausible)
            subj2 = person_dp(number=number1, name_allowed=True, implausible=implausible)

            
            s1 = " ".join(["who", main_aux, subj1, "and", subj2, v, "?"])
            s2 = " ".join(["who", main_aux, subj1, v, "and", subj2, "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def wh_island(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])
                main_aux_type = random.choice(aux_qclause_sg)
                aux_types = aux_types_sg
            else:
                main_aux_type = random.choice(aux_qclause_pl)
                aux_types = aux_types_pl

            main_aux = random.choice(auxes_qclause[main_aux_type])
           
            main_verb = inflect_verb_group(allv_wh_object, v_type=main_aux_type)
            main_aux, main_v = v_to_aux_v(main_verb)

            if number == "pl":
                pronoun = "they"
            else:
                if gender == "m":
                    pronoun = "he"
                else:
                    pronoun = "she"

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person"])
            clause = create_matrix_clause(qinv=True, aux_types=aux_types, v_category=v_category, implausible=implausible)
            v = clause["verb"]
            o = clause["object"]

            if v_category in ["animate_animate", "person_person"]:
                wh_word = "who"
            elif v_category == "person_place":
                wh_word = "where"
            else:
                wh_word = "what"
            
            s1 = " ".join([wh_word, main_aux, main_subj, main_v, pronoun, v,  "?"])
            s2 = " ".join([wh_word, main_aux, main_subj, main_v, "who", v,  "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def complex_np_island(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person"])
            clause = create_matrix_clause(qinv=True, aux_types=aux_qinv_sg+aux_qinv_pl, v_category=v_category, implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]
            aux, v = v_to_aux_v(v)

            if clause["subject_number"] == "sg":
                rc_aux_types = aux_types_sg
            else:
                rc_aux_types = aux_types_pl


            rc = create_rc("double_person_subj", clause["subject_number"], aux_types=rc_aux_types, implausible=implausible)
            rc_s = rc["subject"]
            rc_v = rc["verb"]
            rc_o = rc["object"]
            rc_r = rc["relativizer"]

            if v_category in ["animate_animate", "person_person"]:
                wh_word = "who"
            elif v_category == "person_place":
                wh_word = "where"
            else:
                wh_word = "what"
            
            s1 = " ".join([wh_word, aux, s, rc_r, rc_v, rc_o, v, "?"])
            s2 = " ".join([wh_word, aux, rc_o, v, s, rc_r, rc_v, "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def adjunct_island(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])
            clause = create_matrix_clause(qinv=True, aux_types=aux_qinv_sg+aux_qinv_pl, v_category=v_category, implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]
            aux, v = v_to_aux_v(v)

            clause_adjunct = create_matrix_clause(qinv=True, aux_types=["ving_sg"], v_category=v_category, implausible=implausible)
            adjunct_s = clause_adjunct["subject"]
            adjunct_v = clause_adjunct["verb"]
            adjunct_o = clause_adjunct["object"]
            adjunct_aux, adjunct_v = v_to_aux_v(adjunct_v)
            
            if v_category in ["animate_animate", "person_person"]:
                wh_word = "who"
            elif v_category == "person_place":
                wh_word = "where"
            else:
                wh_word = "what"
            adv = random.choice(prep_ing_adjunct)

            s1 = " ".join([wh_word, aux, s, v, adv, adjunct_v, adjunct_o, "?"])
            s2 = " ".join([wh_word, aux, s, v, adjunct_o, adv, adjunct_v, "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


###########################################
# Filler gap
###########################################

def wh_vs_that_with_gap_long_distance(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])

            main_verb = inflect_verb_group(allv_wh_or_np, number=number)

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            v_type = random.choice(["person_person_obl", "person_object_obl"])
            clause = create_matrix_clause(v_category=v_type, implausible=implausible)

            if clause["object_type"] == "person":
                wh_word = "who"
            else:
                wh_word = "what"


            rc = create_rc(clause["subject_type"], clause["subject_number"], implausible=implausible)
            if rc["subject"] is None:
                if rc["object"] is None:
                    rc = " ".join([rc["relativizer"], rc["verb"]])
                else:
                    rc = " ".join([rc["relativizer"], rc["verb"], rc["object"]])
            else:
                rc = " ".join([rc["relativizer"], rc["subject"], rc["verb"]])

            clause = " ".join([clause["subject"], rc, clause["verb"]])

            s1 = " ".join([main_subj, main_verb, wh_word, clause, "." ])
            s2 = " ".join([main_subj, main_verb, "that", clause, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def wh_vs_that_with_gap(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])

            main_verb = inflect_verb_group(allv_wh_or_np, number=number)

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            v_type = random.choice(["person_person_obl", "person_object_obl"])
            clause = create_matrix_clause(v_category=v_type, implausible=implausible)

            if clause["object_type"] == "person":
                wh_word = "who"
            else:
                wh_word = "what"

            clause = " ".join([clause["subject"], clause["verb"]])

            s1 = " ".join([main_subj, main_verb, wh_word, clause, "." ])
            s2 = " ".join([main_subj, main_verb, "that", clause, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def wh_vs_that_no_gap_long_distance(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])

            main_verb = inflect_verb_group(allv_wh_or_np, number=number)

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            v_type = random.choice(["person_person_obl"])
            clause = create_matrix_clause(v_category=v_type, implausible=implausible)

            if clause["object_type"] == "person":
                wh_word = "who"
            else:
                wh_word = "what"


            rc = create_rc(clause["subject_type"], clause["subject_number"], implausible=implausible)
            if rc["subject"] is None:
                if rc["object"] is None:
                    rc = " ".join([rc["relativizer"], rc["verb"]])
                else:
                    rc = " ".join([rc["relativizer"], rc["verb"], rc["object"]])
            else:
                rc = " ".join([rc["relativizer"], rc["subject"], rc["verb"]])

            clause = " ".join([clause["subject"], rc, clause["verb"], clause["object"]])

            s1 = " ".join([main_subj, main_verb, "that", clause, "." ])
            s2 = " ".join([main_subj, main_verb, wh_word, clause, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples




def wh_vs_that_no_gap(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])

            main_verb = inflect_verb_group(allv_wh_or_np, number=number)

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            v_type = random.choice(["person_person_obl"])
            clause = create_matrix_clause(v_category=v_type, implausible=implausible)

            if clause["object_type"] == "person":
                wh_word = "who"
            else:
                wh_word = "what"

            clause = " ".join([clause["subject"], clause["verb"], clause["object"]])

            s1 = " ".join([main_subj, main_verb, "that", clause, "." ])
            s2 = " ".join([main_subj, main_verb, wh_word, clause, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def wh_questions_subject_gap_long_distance(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])
                main_aux_type = random.choice(aux_qclause_sg)
                aux_types = aux_types_sg
            else:
                main_aux_type = random.choice(aux_qclause_pl)
                aux_types = aux_types_pl

            main_aux = random.choice(auxes_qclause[main_aux_type])
           
            main_verb = inflect_verb_group(allv_wh_or_person, v_type=main_aux_type)
            main_aux, main_v = v_to_aux_v(main_verb)

            if number == "pl":
                pronoun = "they"
            else:
                if gender == "m":
                    pronoun = "he"
                else:
                    pronoun = "she"

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)


            object_type = random.choice(["animate", "person", "person", "person", "person", "person"])
            object_number = random.choice(["sg", "pl"])
            if object_number == "sg":
                obj = random.choice(selectional_categories[object_type][0])
                obj_det = random.choice(det_sg)
            else:
                obj = random.choice(selectional_categories[object_type][1])
                obj_det = random.choice(det_pl)

            obj = obj_det + " " + obj
            rc = create_rc(object_type, object_number, only_transitive=True, position="subj", implausible=implausible)

            extra_rc = create_rc(object_type, object_number, implausible=implausible)
            if extra_rc["subject"] is None:
                if extra_rc["object"] is None:
                    extra_rc = " ".join([extra_rc["relativizer"], extra_rc["verb"]])
                else:
                    extra_rc = " ".join([extra_rc["relativizer"], extra_rc["verb"], extra_rc["object"]])
            else:
                extra_rc = " ".join([extra_rc["relativizer"], extra_rc["subject"], extra_rc["verb"]])

            correct_rc = " ".join([obj, extra_rc, rc["relativizer"], rc["verb"], rc["object"]])
            incorrect_rc = " ".join([rc["wh_word"], obj, extra_rc, rc["verb"], rc["object"]])
            
            s1 = " ".join([main_subj, main_verb, correct_rc, "." ])
            s2 = " ".join([main_subj, main_verb, incorrect_rc, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def wh_questions_object_gap_long_distance(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])
                main_aux_type = random.choice(aux_qclause_sg)
                aux_types = aux_types_sg
            else:
                main_aux_type = random.choice(aux_qclause_pl)
                aux_types = aux_types_pl

            main_aux = random.choice(auxes_qclause[main_aux_type])
           
            main_verb = inflect_verb_group(allv_wh_or_person, v_type=main_aux_type)
            main_aux, main_v = v_to_aux_v(main_verb)

            if number == "pl":
                pronoun = "they"
            else:
                if gender == "m":
                    pronoun = "he"
                else:
                    pronoun = "she"

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            if implausible:
                object_type = random.choice(["animate", "person", "person", "person", "person", "person", "inanimate", "inanimate", "inanimate", "inanimate", "inanimate"])
            else:
                object_type = random.choice(["animate", "physent", "person", "person", "person", "person", "person", "place", "inanimate", "inanimate", "inanimate", "inanimate", "inanimate"])
            object_number = random.choice(["sg", "pl"])
            if object_number == "sg":
                obj = random.choice(selectional_categories[object_type][0])
                obj_det = random.choice(det_sg)
            else:
                obj = random.choice(selectional_categories[object_type][1])
                obj_det = random.choice(det_pl)

            obj = obj_det + " " + obj
            rc = create_rc(object_type, object_number, only_transitive=True, position="obj", implausible=implausible)

            extra_rc = create_rc(rc["subject_type"], rc["subject_number"], implausible=implausible)
            if extra_rc["subject"] is None:
                if extra_rc["object"] is None:
                    extra_rc = " ".join([extra_rc["relativizer"], extra_rc["verb"]])
                else:
                    extra_rc = " ".join([extra_rc["relativizer"], extra_rc["verb"], extra_rc["object"]])
            else:
                extra_rc = " ".join([extra_rc["relativizer"], extra_rc["subject"], extra_rc["verb"]])


            correct_rc = " ".join([obj, rc["relativizer"], rc["subject"], extra_rc, rc["verb"]])
            incorrect_rc = " ".join([rc["wh_word"], rc["subject"], extra_rc, rc["verb"], obj])
            
            s1 = " ".join([main_subj, main_verb, correct_rc, "." ])
            s2 = " ".join([main_subj, main_verb, incorrect_rc, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def wh_questions_subject_gap(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])
                main_aux_type = random.choice(aux_qclause_sg)
                aux_types = aux_types_sg
            else:
                main_aux_type = random.choice(aux_qclause_pl)
                aux_types = aux_types_pl

            main_aux = random.choice(auxes_qclause[main_aux_type])
           
            main_verb = inflect_verb_group(allv_wh_or_person, v_type=main_aux_type)
            main_aux, main_v = v_to_aux_v(main_verb)

            if number == "pl":
                pronoun = "they"
            else:
                if gender == "m":
                    pronoun = "he"
                else:
                    pronoun = "she"

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)


            object_type = random.choice(["animate", "person", "person", "person", "person", "person"])
            object_number = random.choice(["sg", "pl"])
            if object_number == "sg":
                obj = random.choice(selectional_categories[object_type][0])
                obj_det = random.choice(det_sg)
            else:
                obj = random.choice(selectional_categories[object_type][1])
                obj_det = random.choice(det_pl)

            obj = obj_det + " " + obj
            rc = create_rc(object_type, object_number, only_transitive=True, position="subj", implausible=implausible)

            if rc["subject"] is None:
                correct_rc = " ".join([obj, rc["relativizer"], rc["verb"], rc["object"]])
                incorrect_rc = " ".join([rc["wh_word"], obj, rc["verb"], rc["object"]])
            else:
                correct_rc = " ".join([obj, rc["relativizer"], rc["subject"], rc["verb"]])
                incorrect_rc = " ".join([rc["wh_word"], rc["subject"], rc["verb"], obj])
            
            s1 = " ".join([main_subj, main_verb, correct_rc, "." ])
            s2 = " ".join([main_subj, main_verb, incorrect_rc, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def wh_questions_object_gap(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            number = random.choice(["sg", "pl"])
            gender = None
            if number == "sg":
                gender = random.choice(["m", "f"])
                main_aux_type = random.choice(aux_qclause_sg)
                aux_types = aux_types_sg
            else:
                main_aux_type = random.choice(aux_qclause_pl)
                aux_types = aux_types_pl

            main_aux = random.choice(auxes_qclause[main_aux_type])
           
            main_verb = inflect_verb_group(allv_wh_or_person, v_type=main_aux_type)
            main_aux, main_v = v_to_aux_v(main_verb)

            if number == "pl":
                pronoun = "they"
            else:
                if gender == "m":
                    pronoun = "he"
                else:
                    pronoun = "she"

            main_subj = person_dp(number=number, name_allowed=True, gender=gender, implausible=implausible)

            
            if implausible:
                object_type = random.choice(["animate", "person", "person", "person", "person", "person", "inanimate", "inanimate", "inanimate", "inanimate", "inanimate"])
            else:
                object_type = random.choice(["animate", "physent", "person", "person", "person", "person", "person", "place", "inanimate", "inanimate", "inanimate", "inanimate", "inanimate"])
            object_number = random.choice(["sg", "pl"])
            if object_number == "sg":
                obj = random.choice(selectional_categories[object_type][0])
                obj_det = random.choice(det_sg)
            else:
                obj = random.choice(selectional_categories[object_type][1])
                obj_det = random.choice(det_pl)

            obj = obj_det + " " + obj
            rc = create_rc(object_type, object_number, only_transitive=True, position="obj", implausible=implausible)

            if rc["subject"] is None:
                correct_rc = " ".join([obj, rc["relativizer"], rc["verb"], rc["object"]])
                incorrect_rc = " ".join([rc["wh_word"], obj, rc["verb"], rc["object"]])
            else:
                correct_rc = " ".join([obj, rc["relativizer"], rc["subject"], rc["verb"]])
                incorrect_rc = " ".join([rc["wh_word"], rc["subject"], rc["verb"], obj])
            
            s1 = " ".join([main_subj, main_verb, correct_rc, "." ])
            s2 = " ".join([main_subj, main_verb, incorrect_rc, "." ])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

#######################
# Binding
#######################

def principle_A_domain_3(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = allv_not_anaphor
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            anaphor, pronoun = random.choice([("herself", "her"), ("himself", "him")])
            if anaphor == "themselves":
                number = "pl"
            else:
                number = "sg"

            verb = inflect_verb_group(allv_clausal, number=number)

            if anaphor == "herself":
                subj = person_dp(gender="f", name_allowed=True)
                inner_subj = person_dp(number="sg", gender="m", name_allowed=True)
                inner_number = "sg"
                correct_anaphor = "himself"
            elif anaphor == "himself":
                subj = person_dp(gender="m", name_allowed=True)
                inner_subj = person_dp(number="sg", gender="f", name_allowed=True)
                inner_number = "sg"
                correct_anaphor = "herself"

            inner_verb = inflect_verb_group(verb_candidates, number=inner_number)
            

            s1 = " ".join([subj, verb, "that", inner_subj, inner_verb, correct_anaphor, "."])
            s2 = " ".join([inner_subj, verb, "that", subj, inner_verb, correct_anaphor, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def principle_A_domain_2(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = allv_not_anaphor
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            anaphor, pronoun = random.choice([("herself", "her"), ("himself", "him"), ("themselves", "them")])
            if anaphor == "themselves":
                number = "pl"
            else:
                number = "sg"

            verb = inflect_verb_group(allv_clausal, number=number)

            if anaphor == "herself":
                subj = person_dp(gender="f", name_allowed=True)
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="m", name_allowed=True)
                    inner_number = "sg"
                    correct_anaphor = "himself"
                else:
                    inner_subj = person_dp(number="pl")
                    inner_number = "pl"
                    correct_anaphor = "themselves"
            elif anaphor == "himself":
                subj = person_dp(gender="m", name_allowed=True)
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="f", name_allowed=True)
                    inner_number = "sg"
                    correct_anaphor = "herself"
                else:
                    inner_subj = person_dp(number="pl")
                    inner_number = "pl"
                    correct_anaphor = "themselves"
            else:
                subj = person_dp(number="pl")
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="m", name_allowed=True)
                    inner_number = "sg"
                    correct_anaphor = "himself"
                else:
                    inner_subj = person_dp(number="sg", gender="f", name_allowed=True)
                    inner_number = "sg"
                    correct_anaphor = "herself"

            inner_verb = inflect_verb_group(verb_candidates, number=inner_number)
            

            s1 = " ".join([subj, verb, "that", inner_subj, inner_verb, correct_anaphor, "."])
            s2 = " ".join([subj, verb, "that", inner_subj, inner_verb, anaphor, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def principle_A_domain_1(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = allv_not_anaphor
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            anaphor, pronoun = random.choice([("herself", "her"), ("himself", "him"), ("themselves", "them")])
            if anaphor == "themselves":
                number = "pl"
            else:
                number = "sg"

            verb = inflect_verb_group(allv_clausal, number=number)

            if anaphor == "herself":
                subj = person_dp(gender="f", name_allowed=True)
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="m", name_allowed=True)
                    inner_number = "sg"
                else:
                    inner_subj = person_dp(number="pl")
                    inner_number = "pl"
            elif anaphor == "himself":
                subj = person_dp(gender="m", name_allowed=True)
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="f", name_allowed=True)
                    inner_number = "sg"
                else:
                    inner_subj = person_dp(number="pl")
                    inner_number = "pl"
            else:
                subj = person_dp(number="pl")
                if random.choice([True, False]):
                    inner_subj = person_dp(number="sg", gender="m", name_allowed=True)
                    inner_number = "sg"
                else:
                    inner_subj = person_dp(number="sg", gender="f", name_allowed=True)
                    inner_number = "sg"

            inner_verb = inflect_verb_group(verb_candidates, number=inner_number)
            

            s1 = " ".join([subj, verb, "that", inner_subj, inner_verb, pronoun, "."])
            s2 = " ".join([subj, verb, "that", inner_subj, inner_verb, anaphor, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def principle_A_c_command(n_examples=1000, implausible=False):

    if implausible:
        verb_candidates = allv_not_anaphor
    else:
        verb_candidates = join_verb_groups([allv_animate_physent, allv_person_physent, allv_animate_animate, allv_person_person])
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            anaphor = random.choice(["herself", "himself", "themselves"])
            if anaphor == "themselves":
                number = "pl"
            else:
                number = "sg"

            verb = inflect_verb_group(verb_candidates, number=number)

            if anaphor == "herself":
                subj = person_dp(gender="f")
                if random.choice([True, False]):
                    rc_obj = person_dp(number="sg", gender="m", name_allowed=True)
                    incorrect_anaphor = "himself"
                else:
                    rc_obj = person_dp(number="pl")
                    incorrect_anaphor = "themselves"
            elif anaphor == "himself":
                subj = person_dp(gender="m")
                if random.choice([True, False]):
                    rc_obj = person_dp(number="sg", gender="f", name_allowed=True)
                    incorrect_anaphor = "herself"
                else:
                    rc_obj = person_dp(number="pl")
                    incorrect_anaphor = "themselves"
            else:
                subj = person_dp(number="pl")
                if random.choice([True, False]):
                    rc_obj = person_dp(number="sg", gender="m", name_allowed=True)
                    incorrect_anaphor = "himself"
                else:
                    rc_obj = person_dp(number="sg", gender="f", name_allowed=True)
                    incorrect_anaphor = "herself"

            relativizer = random.choice(["who", "that"])
            rc_verb = inflect_verb_group(verb_candidates, number=number)
            

            s1 = " ".join([subj, relativizer, rc_verb, rc_obj, verb, anaphor, "."])
            s2 = " ".join([subj, relativizer, rc_verb, rc_obj, verb, incorrect_anaphor, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


##################################################
# Subject-verb agreement
##################################################

def subj_aux_vary_subj(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            clause = create_matrix_clause(aux_types=["ving_sg", "ving_pl", "ven_sg", "ven_pl", "inf_sg", "inf_pl"], implausible=implausible)

            sg_subj, pl_subj = noun_sg_pl_pair(clause["subject_type"], implausible=implausible)
            det = random.choice(["the"])

            aux_type = clause["aux_type"]
        
            aux, v = v_to_aux_v(clause["verb"])
            if aux_type == "ving_sg":
                aux = random.choice(["is", "is n't", "was", "was n't"])
            elif aux_type == "ving_pl":
                aux = random.choice(["are", "are n't", "were", "were n't"])
            elif aux_type == "ven_sg":
                aux = random.choice(["has", "has n't"])
            elif aux_type == "ven_pl":
                aux = random.choice(["have", "have n't"])
            elif aux_type == "inf_sg":
                aux = random.choice(["does", "does n't"])
            else:
                aux = random.choice(["do", "do n't"])

            clause["verb"] = aux + " " + v

            if clause["subject_number"] == "sg":
                correct_subj = det + " " + sg_subj
                incorrect_subj = det + " " + pl_subj

            else:
                correct_subj = det + " " + pl_subj
                incorrect_subj = det + " " + sg_subj

            if clause["object"] is not None:
                s1 = " ".join([correct_subj, clause["verb"], clause["object"], "."])
                s2 = " ".join([incorrect_subj, clause["verb"], clause["object"], "."])
            else:
                s1 = " ".join([correct_subj, clause["verb"], "."])
                s2 = " ".join([incorrect_subj, clause["verb"], "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def subj_aux_vary_verb(n_examples=1000, distractor=None, distractor_count=1, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            clause = create_matrix_clause(implausible=implausible)

            sg_verb, pl_verb = aux_sg_pl_pair(clause["verb_category"])
            
            if clause["subject_number"] == "sg":
                correct_verb = sg_verb
                incorrect_verb = pl_verb
            else:
                correct_verb = pl_verb
                incorrect_verb = sg_verb

            if distractor == "rc":
                for _ in range(distractor_count):
                    clause["subject"] = clause["subject"] + " " + stringify_rc(create_opposite_number_rc(clause["subject_type"], clause["subject_number"], no_clausal=True, implausible=implausible))
            elif distractor == "pp":
                for _ in range(distractor_count):
                    clause["subject"] = clause["subject"] + " " + create_opposite_number_pp(clause["subject_type"], clause["subject_number"], implausible=implausible)

            if clause["object"] is not None:
                s1 = " ".join([clause["subject"], correct_verb, clause["object"], "."])
                s2 = " ".join([clause["subject"], incorrect_verb, clause["object"], "."])
            else:
                s1 = " ".join([clause["subject"], correct_verb, "."])
                s2 = " ".join([clause["subject"], incorrect_verb, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def subj_verb_vary_verb(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            clause = create_matrix_clause(implausible=implausible)

            sg_verb, pl_verb = verb_sg_pl_pair(clause["verb_category"])
            
            if clause["subject_number"] == "sg":
                correct_verb = sg_verb
                incorrect_verb = pl_verb
            else:
                correct_verb = pl_verb
                incorrect_verb = sg_verb


            if clause["object"] is not None:
                s1 = " ".join([clause["subject"], correct_verb, clause["object"], "."])
                s2 = " ".join([clause["subject"], incorrect_verb, clause["object"], "."])
            else:
                s1 = " ".join([clause["subject"], correct_verb, "."])
                s2 = " ".join([clause["subject"], incorrect_verb, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def subj_aux_agreement_question_vary_subj(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            clause = create_matrix_clause(aux_types=["ving_sg", "ving_pl", "inf_sg", "inf_pl", "ven_sg", "ven_pl"], implausible=implausible)
            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            aux_type = clause["aux_type"]
            aux, v = v_to_aux_v(v)
            if aux_type == "ving_sg":
                aux = random.choice(["is", "is n't", "was", "was n't"])
            elif aux_type == "ving_pl":
                aux = random.choice(["are", "are n't", "were", "were n't"])
            elif aux_type == "ven_sg":
                aux = random.choice(["has", "has n't"])
            elif aux_type == "ven_pl":
                aux = random.choice(["have", "have n't"])
            elif aux_type == "inf_sg":
                aux = random.choice(["does", "does n't"])
            else:
                aux = random.choice(["do", "do n't"])


            det = "the"
            subj1, subj2 = noun_sg_pl_pair(clause["subject_type"], implausible=implausible)

            if clause["subject_number"] == "sg":
                correct_subj = subj1
                incorrect_subj = subj2
            else:
                correct_subj = subj2
                incorrect_subj = subj1

            if o is None:
                s1 = " ".join([aux, det, correct_subj, v, "?"])
                s2 = " ".join([aux, det, incorrect_subj, v, "?"])
            else:
                s1 = " ".join([aux, det, correct_subj, v, o, "?"])
                s2 = " ".join([aux, det, incorrect_subj, v, o, "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



def subj_aux_agreement_question_vary_aux(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            clause = create_matrix_clause(aux_types=["ving_sg", "ving_pl", "ven_sg", "ven_pl", "inf_sg", "inf_pl"], implausible=implausible)
            
            v1, v2 = aux_sg_pl_pair(clause["verb_category"])
            
            det = random.choice(["the"])

            s = clause["subject"]
            v = clause["verb"]
            o = clause["object"]

            aux1, v1 = v_to_aux_v(v1)
            aux2, v2 = v_to_aux_v(v2)

            if clause["subject_number"] == "sg":
                correct_aux = aux1
                incorrect_aux = aux2
            else:
                correct_aux = aux2
                incorrect_aux = aux1

            if o is None:
                s1 = " ".join([correct_aux, s, v1, "?"])
                s2 = " ".join([incorrect_aux, s, v2, "?"])
            else:
                s1 = " ".join([correct_aux, s, v1, o, "?"])
                s2 = " ".join([incorrect_aux, s, v2, o, "?"])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples




################################
# Determiner-noun agreement
################################

def determiner_noun_agreement_1(n_examples=1000, adjs=0, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            clause = create_matrix_clause(v_category=v_category, implausible=implausible)

            clause["subject"] = "the " + " ".join(clause["subject"].split()[1:])
            sg_object, pl_object = noun_sg_pl_pair(clause["object_type"], implausible=implausible)
            if clause["object_number"] == "sg":
                det = random.choice(["this", "that"])
                correct_obj = sg_object
                incorrect_obj = pl_object
            else:
                det = random.choice(["these", "those"])
                correct_obj = pl_object
                incorrect_obj = sg_object

            adj_list = []
            for _ in range(adjs):
                adj_list.append(random.choice(adjectives[clause["object_type"]]))
            if adjs > 0:
                det = det + " " + " ".join(adj_list)

            s1 = " ".join([clause["subject"], clause["verb"], det, correct_obj, "."])
            s2 = " ".join([clause["subject"], clause["verb"], det, incorrect_obj, "."])
            
            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def determiner_noun_agreement_2(n_examples=1000, adjs=0, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            clause = create_matrix_clause(v_category=v_category, implausible=implausible)

            clause["subject"] = "the " + " ".join(clause["subject"].split()[1:])
            clause_object = " ".join(clause["object"].split()[1:])

            if clause["object_number"] == "sg":
                correct_det, incorrect_det = random.choice([["this", "these"], ["that", "those"]])
            else:
                incorrect_det, correct_det = random.choice([["this", "these"], ["that", "those"]])

            adj_list = []
            for _ in range(adjs):
                adj_list.append(random.choice(adjectives[clause["object_type"]]))
            if adjs > 0:
                clause_object = " ".join(adj_list) + " " + clause_object


            s1 = " ".join([clause["subject"], clause["verb"], correct_det, clause_object, "."])
            s2 = " ".join([clause["subject"], clause["verb"], incorrect_det, clause_object, "."])
            
            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples




###################################
# Basic word order
###################################

def svo_vos(n_examples=1000, implausible=False):
   
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            clause = create_matrix_clause(v_category=v_category, implausible=implausible)

            s1 = " ".join([clause["subject"], clause["verb"], clause["object"], "."])
            s2 = " ".join([clause["verb"], clause["object"], clause["subject"], "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def green_ideas(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            adjective = random.choice(adjectives["person"])
            if implausible:
                subject = random.choice(nouns_inanimate_pl)
            else:
                subject = random.choice(person_pl)
            verb = random.choice(v_intrans_person)
            adv = random.choice(adv_manner)

            words_correct = [("adj", adjective), ("n", subject), ("v", verb), ("adv", adv)]

            words_incorrect = words_correct[:]
            satisfied = False
            while not satisfied:
                random.shuffle(words_incorrect)
                order = [x[0] for x in words_incorrect]
                if order != ["adj", "n", "v", "adv"] and order != ["adv", "adj", "n", "v"] and order != ["adj", "n", "adv", "v"]:
                    satisfied = True

            s1 = " ".join([x[1] for x in words_correct]) + " ."
            s2 = " ".join([x[1] for x in words_incorrect]) + " ."

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def adv_order(n_examples=1000, implausible=False):
   
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            adv = random.choice(adv_manner)
            subj = person_dp(implausible=implausible)
            obj = person_dp(implausible=implausible)
            verb = random.choice(ved_person_person)

            if random.choice([True, False]):
                s1 = " ".join([subj, adv, verb, obj, "."])
            else:
                s1 = " ".join([subj, verb, obj, adv, "."])

            s2 = " ".join([subj, verb, adv, obj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def pp_order(n_examples=1000, implausible=False):

    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            pp = create_pp_locative()
            subj = person_dp(implausible=implausible)
            obj = person_dp(implausible=implausible)
            verb = random.choice(ved_person_person)

            if random.choice([True, False]):
                s1 = " ".join([subj, pp, verb, obj, "."])
            else:
                s1 = " ".join([subj, verb, obj, pp, "."])

            s2 = " ".join([subj, verb, pp, obj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples







###################################
# Nested dependencies
###################################


def center_embedding_single_1(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if random.choice([True, False]):
                subject = person_dp(number="sg", implausible=implausible)
                other = person_dp(number="pl", implausible=implausible)

                v1 = random.choice(v_person_person)
                v2 = random.choice(v_intrans_person)
                v3 = random.choice(vs_intrans_person)
            else:
                subject = person_dp(number="pl", implausible=implausible)
                other = person_dp(number="sg", implausible=implausible)

                v1 = random.choice(vs_person_person)
                v2 = random.choice(vs_intrans_person)
                v3 = random.choice(v_intrans_person)

            s1 = " ".join([subject, other, v1, v3, "."])
            s2 = " ".join([subject, other, v3, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def center_embedding_single_2(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if random.choice([True, False]):
                subject = person_dp(number="sg", implausible=implausible)
                other = person_dp(number="pl", implausible=implausible)

                v1 = random.choice(v_person_person)
                v2 = random.choice(v_intrans_person)
                v3 = random.choice(vs_intrans_person)
            else:
                subject = person_dp(number="pl", implausible=implausible)
                other = person_dp(number="sg", implausible=implausible)

                v1 = random.choice(vs_person_person)
                v2 = random.choice(vs_intrans_person)
                v3 = random.choice(v_intrans_person)

            s1 = " ".join([subject, other, v1, v3, "."])
            s2 = " ".join([subject, other, v1, v2, v3, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def center_embedding_double_1(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if random.choice([True, False]):
                subject = person_dp(number="sg", implausible=implausible)
                other = person_dp(number="pl", implausible=implausible)
                other2 = person_dp(number="pl", implausible=implausible)

                v1 = random.choice(v_person_person)
                v2 = random.choice(v_person_person)
                v3 = random.choice(v_intrans_person)
                v4 = random.choice(vs_intrans_person)
            else:
                subject = person_dp(number="pl", implausible=implausible)
                other = person_dp(number="sg", implausible=implausible)
                other2 = person_dp(number="sg", implausible=implausible)

                v1 = random.choice(vs_person_person)
                v2 = random.choice(vs_person_person)
                v3 = random.choice(vs_intrans_person)
                v4 = random.choice(v_intrans_person)

            s1 = " ".join([subject, other, other2, v1, v2, v4, "."])
            s2 = " ".join([subject, other, other2, v1, v4, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def center_embedding_double_2(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            if random.choice([True, False]):
                subject = person_dp(number="sg", implausible=implausible)
                other = person_dp(number="pl", implausible=implausible)
                other2 = person_dp(number="pl", implausible=implausible)

                v1 = random.choice(v_person_person)
                v2 = random.choice(v_person_person)
                v3 = random.choice(v_intrans_person)
                v4 = random.choice(vs_intrans_person)
            else:
                subject = person_dp(number="pl", implausible=implausible)
                other = person_dp(number="sg", implausible=implausible)
                other2 = person_dp(number="sg", implausible=implausible)

                v1 = random.choice(vs_person_person)
                v2 = random.choice(vs_person_person)
                v3 = random.choice(vs_intrans_person)
                v4 = random.choice(v_intrans_person)

            s1 = " ".join([subject, other, other2, v1, v2, v4, "."])
            s2 = " ".join([subject, other, other2, v1, v2, v3, v4, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples



###################################
# Case
###################################


def swapped_ditransitive_1(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            subj_pronoun = random.choice(["I", "we", "he", "she", "they"])
            gender = random.choice([None, "f", "m"])
            recipient = person_dp(name_allowed=True, gender=gender, implausible=implausible)
            obj = "the " + random.choice(nouns_inanimate_sg + nouns_inanimate_pl)
            v = random.choice(ved_ditransitive)

            s1 = " ".join([subj_pronoun, v, recipient, obj, "."])
            s2 = " ".join([recipient, v, obj, subj_pronoun, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def swapped_ditransitive_2(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            subj_pronoun = random.choice(["I", "we", "he", "she", "they"])
            gender = random.choice([None, "f", "m"])
            recipient = person_dp(name_allowed=True, gender=gender, implausible=implausible)
            obj = "the " + random.choice(nouns_inanimate_sg + nouns_inanimate_pl)
            v = random.choice(ved_ditransitive)

            s1 = " ".join([subj_pronoun, v, recipient, obj, "."])
            s2 = " ".join([recipient, v, subj_pronoun, obj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


###################################
# Ellipsis
###################################


def ellipsis(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:

            if implausible:
                v_category = random.choice(["animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_object", "person_object", "person_object", "person_object"])
            else:
                v_category = random.choice(["animate_physent", "animate_physent", "person_physent", "animate_animate", "animate_animate", "person_person", "person_person", "person_person", "person_person", "person_place", "person_place", "person_object", "person_object", "person_object", "person_object", "physent_object"])

            clause = create_matrix_clause(v_category=v_category, aux_types=aux_type_nonnegative, implausible=implausible)
            object_type = clause["object_type"]
            subject_type = clause["subject_type"]

            numbers = ["one", "two", "three", "four", "five"]
            random.shuffle(numbers)
            number1 = numbers[0]
            number2 = numbers[1]
            
            if number1 == "one":
                if implausible:
                    obj = random.choice(selectional_categories_implausible[object_type][0])
                else:
                    obj = random.choice(selectional_categories[object_type][0])
            else:
                if implausible:
                    obj = random.choice(selectional_categories_implausible[object_type][1])
                else:
                    obj = random.choice(selectional_categories[object_type][1])

            adjective = random.choice(adjectives[clause["object_type"]])

            if clause["subject_number"] == "sg":
                det2 = random.choice(det_sg)
                if implausible:
                    subj2 = random.choice(selectional_categories_implausible[subject_type][0])
                else:
                    subj2 = random.choice(selectional_categories[subject_type][0])
            else:
                det2 = random.choice(det_pl)
                if implausible:
                    subj2 = random.choice(selectional_categories_implausible[subject_type][1])
                else:
                    subj2 = random.choice(selectional_categories[subject_type][1])

            s1 = " ".join([clause["subject"], clause["verb"], number1, adjective, obj, ",", "and", det2, subj2, clause["verb"], number2, "."])
            s2 = " ".join([clause["subject"], clause["verb"], number1, obj, ",", "and", det2, subj2, clause["verb"], number2, adjective, "."])
            
            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples





###################################
# Recursion
###################################

def poss():
    rels = male_relation_sg + female_relation_sg + relation_sg
    return random.choice(rels) + " 's"


def recursion_intensifier_adj(n_examples=1000, n_ints=None, implausible=False):
    
    n_ints_input = n_ints

    examples = {}
    for example_index in range(n_examples):

        if n_ints_input == "short":
            n_ints = random.choice([0,1,2])
        elif n_ints_input == "medium":
            n_ints = random.choice([3,4,5,6])
        elif n_ints_input == "long":
            n_ints = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            
            intensifier = random.choice(["really", "very"])
            intensifiers = [intensifier for _ in range(n_ints)]

            n_type = "person"
            if implausible:
                noun = random.choice(nouns_inanimate_sg)
            else:
                noun = random.choice(nouns_person_sg)

            adj = random.choice(adjectives[n_type])
            linking_verb = random.choice(["is", "was"])

            v = inflect_verb_group(allv_intrans_person, number="sg", v_type="vs")

            if n_ints == 0:
                s1 = " ".join(["the", noun, linking_verb, adj, "."])
                s2 = " ".join(["the", noun, v, adj, "."])
            else:
                s1 = " ".join(["the", noun, linking_verb, " ".join(intensifiers), adj, "."])
                s2 = " ".join(["the", noun, v, " ".join(intensifiers), adj, "."])

    
            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def recursion_intensifier_adv(n_examples=1000, n_ints=None, implausible=False):
    
    examples = {}

    n_ints_input = n_ints

    for example_index in range(n_examples):

        if n_ints_input == "short":
            n_ints = random.choice([0,1,2])
        elif n_ints_input == "medium":
            n_ints = random.choice([3,4,5,6])
        elif n_ints_input == "long":
            n_ints = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            
            intensifier = random.choice(["really", "very"])
            intensifiers = [intensifier for _ in range(n_ints)]

            n_type = "person" 
            if implausible:
                noun = random.choice(nouns_inanimate_sg)
            else:
                noun = random.choice(nouns_person_sg)

            adj = random.choice(adjectives[n_type])
            linking_verb = random.choice(["is", "was"])

            v = inflect_verb_group(allv_intrans_person, number="sg", v_type="vs")
            adv = random.choice(adv_manner)

            if n_ints == 0:
                s1 = " ".join(["the", noun, v, adv, "."])
                s2 = " ".join(["the", noun, linking_verb, adv, "."])
            else:
                s1 = " ".join(["the", noun, v, " ".join(intensifiers), adv, "."])
                s2 = " ".join(["the", noun, linking_verb, " ".join(intensifiers), adv, "."])
    
            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples

def recursion_pp_verb(n_examples=1000, n_pps=None, implausible=False):
    
    examples = {}

    n_pps_input = n_pps

    for example_index in range(n_examples):

        if n_pps_input == "short":
            n_pps = random.choice([0,1,2])
        elif n_pps_input == "medium":
            n_pps = random.choice([3,4,5,6])
        elif n_pps_input == "long":
            n_pps = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            pps = [create_pp_locative() for _ in range(n_pps)]
            subj = random.choice(nouns_inanimate_sg)

            if implausible:
                adj = random.choice(adjectives["person"])
            else:
                adj = random.choice(adjectives["inanimate"])

            v1, v2 = random.choice([("sits", "sitting"), ("rests", "resting")])

            if n_pps == 0:
                s1 = " ".join(["the", subj, v2, "is", adj, "."])
                s2 = " ".join(["the", subj, v1, "is", adj, "."])
            else:
                s1 = " ".join(["the", subj, v2, " ".join(pps), "is", adj, "."])
                s2 = " ".join(["the", subj, v1, " ".join(pps), "is", adj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True
    
    return examples

def recursion_pp_is(n_examples=1000, n_pps=None, implausible=False):
    
    examples = {}

    n_pps_input = n_pps

    for example_index in range(n_examples):

        if n_pps_input == "short":
            n_pps = random.choice([0,1,2])
        elif n_pps_input == "medium":
            n_pps = random.choice([3,4,5,6])
        elif n_pps_input == "long":
            n_pps = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            pps = [create_pp_locative() for _ in range(n_pps)]
            subj = random.choice(nouns_inanimate_sg)

            if implausible:
                adj = random.choice(adjectives["person"])
            else:
                adj = random.choice(adjectives["inanimate"])

            verbs = ["is", "was"]
            random.shuffle(verbs)
            v1 = verbs[0]
            v2 = verbs[1]

            if n_pps == 0:
                s1 = " ".join(["the", subj, v2, adj, "."])
                s2 = " ".join(["the", subj, v1, v2, adj, "."])
            else:
                s1 = " ".join(["the", subj, " ".join(pps), v2, adj, "."])
                s2 = " ".join(["the", subj, v1, " ".join(pps), v2, adj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def recursion_poss_transitive(n_examples=1000, n_poss=None, implausible=False):
    
    examples = {}

    n_poss_input = n_poss

    for example_index in range(n_examples):

        if n_poss_input == "short":
            n_poss = random.choice([0,1,2])
        elif n_poss_input == "medium":
            n_poss = random.choice([3,4,5,6])
        elif n_poss_input == "long":
            n_poss = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            if implausible:
                rels = nouns_inanimate_sg
            else:
                rels = male_relation_sg + female_relation_sg + relation_sg
            subj = person_dp(name_allowed=True, implausible=implausible)
            io = random.choice(rels)
            poss_det = random.choice(["my", "your"])
            posses = [poss() for _ in range(n_poss)]
            
            v_clausal = inflect_verb_group(allv_clausal, number="sg", v_type="vs")
            v_transitive = inflect_verb_group(allv_person_person_not_clausal, number="sg", v_type="vs")

            adj = random.choice(adjectives["person"])

            if subj.split()[0] in det_sg + det_pl:
                subj = " ".join(subj.split()[1:])
                subj = "the " + subj

            if io.split()[0] in det_sg + det_pl:
                io = " ".join(io.split()[1:])
                io = "the " + io

            if n_poss == 0:
                s1 = " ".join([subj, v_clausal, "that", poss_det, io, "is", adj, "."])
                s2 = " ".join([subj, v_transitive, poss_det, io, "is", adj, "."])
            else:
                s1 = " ".join([subj, v_clausal, "that", poss_det, " ".join(posses), io, "is", adj, "."])
                s2 = " ".join([subj, v_transitive, poss_det, " ".join(posses), io, "is", adj, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples


def recursion_poss_ditransitive(n_examples=1000, n_poss=None, implausible=False):
    
    examples = {}

    n_poss_input = n_poss

    for example_index in range(n_examples):

        if n_poss_input == "short":
            n_poss = random.choice([0,1,2])
        elif n_poss_input == "medium":
            n_poss = random.choice([3,4,5,6])
        elif n_poss_input == "long":
            n_poss = random.choice([7,8,9,10])

        example_found = False
        while not example_found:
            if implausible:
                rels = nouns_inanimate_sg
            else:
                rels = male_relation_sg + female_relation_sg + relation_sg
            
            subj = person_dp(name_allowed=True, implausible=implausible)
            io = random.choice(rels)
            poss_det = random.choice(["my", "your"])
            posses = [poss() for _ in range(n_poss)]
           
            do = random.choice(nouns_inanimate_sg)
            
            v_ditransitive = inflect_verb_group(allv_ditransitive, number="sg", v_type="pst_sg")
            v_transitive = inflect_verb_group(allv_person_object_not_ditransitive, number="sg", v_type="pst_sg")

            if subj.split()[0] in det_sg + det_pl:
                subj = " ".join(subj.split()[1:])
                subj = "the " + subj

            if io.split()[0] in det_sg + det_pl:
                io = " ".join(io.split()[1:])
                io = "the " + io

            if n_poss == 0:
                s1 = " ".join([subj, v_ditransitive, poss_det, io, "the", do, "."])
                s2 = " ".join([subj, v_transitive, poss_det, io, "the", do, "."])
            else:
                s1 = " ".join([subj, v_ditransitive, poss_det, " ".join(posses), io, "the", do, "."])
                s2 = " ".join([subj, v_transitive, poss_det, " ".join(posses), io, "the", do, "."])

            example = (s1, s2)
            if example not in examples and freq(s1) and freq(s2):
                examples[example] = 1
                example_found = True

    return examples






###################################
# Priming
###################################

def priming_short(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            locative = create_pp_locative()
            clause = create_matrix_clause(v_category="person_person", aux_types=["inf_sg", "ving_sg", "ven_sg", "inf_pl", "ving_pl", "ven_pl"], implausible=implausible)

            position = random.choice([1,2,3])
            if position == 1:
                example_single = " ".join([locative, clause["subject"], clause["verb"], clause["object"], "."])
            elif position == 2:
                example_single = " ".join([clause["subject"], locative, clause["verb"], clause["object"], "."])
            else:
                example_single = " ".join([clause["subject"], clause["verb"], clause["object"], locative, "."])

            example_double = example_single + " " + example_single

            example = (example_single, example_double)


            if example not in examples and freq(example_single) and len(example_single.split()) == 10: 
                examples[example] = 1
                example_found = True

    return examples


def priming_long(n_examples=1000, implausible=False):
    
    examples = {}
    for example_index in range(n_examples):

        example_found = False
        while not example_found:
            
            locative1 = create_pp_locative()
            locative2 = create_pp_locative()
            clause1 = create_matrix_clause(v_category="person_person", aux_types=["inf_sg", "ving_sg", "ven_sg", "inf_pl", "ving_pl", "ven_pl"], implausible=implausible)
            clause2 = create_matrix_clause(v_category="person_person", aux_types=["inf_sg", "ving_sg", "ven_sg", "inf_pl", "ving_pl", "ven_pl"], implausible=implausible)

            position1 = random.choice([1,2,3])
            if position1 == 1:
                example_single1 = " ".join([locative1, clause1["subject"], clause1["verb"], clause1["object"]])
            elif position1 == 2:
                example_single1 = " ".join([clause1["subject"], locative1, clause1["verb"], clause1["object"]])
            else:
                example_single1 = " ".join([clause1["subject"], clause1["verb"], clause1["object"], locative1])

            position2 = random.choice([1,2,3])
            if position2 == 1:
                example_single2 = " ".join([locative2, clause2["subject"], clause2["verb"], clause2["object"]])
            elif position2 == 2:
                example_single2 = " ".join([clause2["subject"], locative2, clause2["verb"], clause2["object"]])
            else:
                example_single2 = " ".join([clause2["subject"], clause2["verb"], clause2["object"], locative2])

            example_single = example_single1 + " , and " + example_single2 + " ."

            example_double = example_single + " " + example_single

            example = (example_single, example_double)

            if example not in examples and freq(example_single) and len(example_single.split()) == 21: 
                examples[example] = 1
                example_found = True

    return examples











def print_examples(directory_name, category, examples, to_command_line=False):
    print(category)
    if to_command_line:
        for sentence_good, sentence_bad in examples:
            print(sentence_good)
            print(sentence_bad)
            print("")

    else:
        fo = open(directory_name + category + ".tsv", "w")
        for sentence_good, sentence_bad in examples:
            fo.write(sentence_good + "\t" + sentence_bad + "\n")


def print_group(directory_name, category, examples, to_command_line=False):
    print(category)
    if to_command_line:
        for example in examples:
            print("\t".join(example))

    else:
        fo = open(directory_name + category + ".tsv", "w")
        for example in examples:
            fo.write("\t".join(example) + "\n")



if args.category == "anaphor_gender_agreement":
    examples = anaphor_gender_agreement(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "anaphor_number_agreement":
    examples = anaphor_number_agreement(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "matrix_question_npi_licensor_present":
    examples = matrix_question_npi_licensor_present(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "npi_present":
    examples = npi_present(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "only_npi_licensor_present":
    examples = only_npi_licensor_present(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "sentential_negation_npi_licensor_present":
    examples = sentential_negation_npi_licensor_present(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "only_npi_scope":
    examples = only_npi_scope(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "sentential_negation_npi_scope":
    examples = sentential_negation_npi_scope(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "intransitive":
    examples = intransitive(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "transitive":
    examples = transitive(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "left_branch_island_simple_question":
    examples = left_branch_island_simple_question(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "coordinate_structure_constrain_object_extraction":
    examples = coordinate_structure_constrain_object_extraction(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_island":
    examples = wh_island(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "complex_np_island":
    examples = complex_np_island(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "adjunct_island":
    examples = adjunct_island(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_vs_that_with_gap_long_distance":
    examples = wh_vs_that_with_gap_long_distance(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_vs_that_with_gap":
    examples = wh_vs_that_with_gap(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_vs_that_no_gap_long_distance":
    examples = wh_vs_that_no_gap_long_distance(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_vs_that_no_gap":
    examples = wh_vs_that_no_gap(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_questions_subject_gap_long_distance":
    examples = wh_questions_subject_gap_long_distance(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_questions_object_gap_long_distance":
    examples = wh_questions_object_gap_long_distance(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_questions_subject_gap":
    examples = wh_questions_subject_gap(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "wh_questions_object_gap":
    examples = wh_questions_object_gap(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "principle_A_domain_3":
    examples = principle_A_domain_3(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "principle_A_domain_2":
    examples = principle_A_domain_2(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "principle_A_domain_1":
    examples = principle_A_domain_1(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "principle_A_c_command":
    examples = principle_A_c_command(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "subj_aux_vary_subj":
    examples = subj_aux_vary_subj(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "subj_aux_vary_verb":
    examples = subj_aux_vary_verb(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "subj_aux_vary_verb_rc":
    examples = subj_aux_vary_verb(n_examples=args.n_examples, implausible=args.implausible, distractor="rc", distractor_count=1)
elif args.category == "subj_aux_vary_verb_pp":
    examples = subj_aux_vary_verb(n_examples=args.n_examples, implausible=args.implausible, distractor="pp", distractor_count=1)
elif args.category == "subj_verb_vary_verb":
    examples = subj_verb_vary_verb(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "subj_aux_agreement_question_vary_subj":
    examples = subj_aux_agreement_question_vary_subj(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "subj_aux_agreement_question_vary_aux":
    examples = subj_aux_agreement_question_vary_aux(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "determiner_noun_agreement_1":
    examples = determiner_noun_agreement_1(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "determiner_noun_agreement_adj_1":
    examples = determiner_noun_agreement_1(n_examples=args.n_examples, implausible=args.implausible, adjs=1)
elif args.category == "determiner_noun_agreement_2":
    examples = determiner_noun_agreement_2(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "determiner_noun_agreement_adj_2":
    examples = determiner_noun_agreement_2(n_examples=args.n_examples, implausible=args.implausible, adjs=1)
elif args.category == "svo_vos":
    examples = svo_vos(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "green_ideas":
    examples = green_ideas(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "adv_order":
    examples = adv_order(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "pp_order":
    examples = pp_order(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "center_embedding_single_1":
    examples = center_embedding_single_1(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "center_embedding_single_2":
    examples = center_embedding_single_2(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "center_embedding_double_1":
    examples = center_embedding_double_1(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "center_embedding_double_2":
    examples = center_embedding_double_2(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "swapped_ditransitive_1":
    examples = swapped_ditransitive_1(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "swapped_ditransitive_2":
    examples = swapped_ditransitive_2(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "ellipsis":
    examples = ellipsis(n_examples=args.n_examples, implausible=args.implausible)
elif args.category.startswith("recursion_intensifier_adj"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_intensifier_adj(n_examples=args.n_examples, implausible=args.implausible, n_ints=number)
elif args.category.startswith("recursion_intensifier_adv"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_intensifier_adv(n_examples=args.n_examples, implausible=args.implausible, n_ints=number)
elif args.category.startswith("recursion_pp_verb"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_pp_verb(n_examples=args.n_examples, implausible=args.implausible, n_pps=number)
elif args.category.startswith("recursion_pp_is"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_pp_is(n_examples=args.n_examples, implausible=args.implausible, n_pps=number)
elif args.category.startswith("recursion_poss_transitive"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_poss_transitive(n_examples=args.n_examples, implausible=args.implausible, n_poss=number)
elif args.category.startswith("recursion_poss_ditransitive"):
    number = args.category.split("_")[-1]
    if number.isnumeric():
        number = int(number)
    examples = recursion_poss_ditransitive(n_examples=args.n_examples, implausible=args.implausible, n_poss=number)
elif args.category == "priming_short":
    examples = priming_short(n_examples=args.n_examples, implausible=args.implausible)
elif args.category == "priming_long":
    examples = priming_long(n_examples=args.n_examples, implausible=args.implausible)




if args.implausible:
    print_examples(args.directory_name, args.category + "_implausible", examples, to_command_line=args.to_command_line)
else:
    print_examples(args.directory_name, args.category, examples, to_command_line=args.to_command_line)



