import nltk
from nltk.corpus import treebank

# here we load in the sentences
sentence22 = treebank.parsed_sents('wsj_0003.mrg')[21]
sentence7 = treebank.parsed_sents('wsj_0003.mrg')[6]
sentence13 = treebank.parsed_sents('wsj_0004.mrg')[12]

# here we define a grammar
grammar = nltk.CFG.fromstring("""
S -> NP VP
S -> Aux NP VP
S -> VP
S -> IVP 
NP -> Pronoun
NP -> Proper-Noun
NP -> Det Nominal 
NP -> NP PP
Nominal -> Noun
Nominal -> Nominal Noun
Nominal -> Nominal PP 
IVP -> IVerb NP NP | IVerb  NP NP PP
VP -> V NP
VP -> Verb
VP -> Verb NP
VP -> Verb NP PP
VP -> Verb PP
VP -> VP PP
PP -> Preposition NP
Det -> 'the'
N -> 'students' | 'subject'
V -> 'like' | 'love'
NP -> Det N | 'NLP' | 'I'     
Det -> 'that' | 'this' | 'the' | 'a'
Noun -> 'book' | 'flight' | 'meal' | 'money' | 'meals' | 'seats'
Verb -> 'book' | 'include' | 'prefer' | 'Show' | 'List'
IVerb -> 'Show' | 'List'
Pronoun -> 'I' | 'she' | 'me'
Proper-Noun -> 'Houston' | 'NWA' | 'SF' | 'Denver'
Aux -> 'does'
Preposition -> 'from' | 'to' | 'on' | 'near' |'through'
""")

# here we let nltk construct a chart parser from our grammar
parser = nltk.ChartParser(grammar)

# input: a list of words
# returns all the parses of a sentence
def allParses(sentenceList):
	return parser.parse(sentenceList)

# input: a list of parse trees
# prints all the parse trees
def printParses(parses):
	for tree in parses:
		print(tree)

# input: a sentence as a string or as a list of words
# prints a sentence, then parses it and prints all the parse trees
def processSentence(sentence):
	sentenceList = sentence
	if isinstance(sentence,str):
		sentenceList = sentence.split(' ')
	print ('Original sentence: ' + ' '.join(sentenceList))
	printParses(allParses(sentenceList))

def mainScript():
	#processSentence('I like NLP')
	#processSentence('the students love the subject')
	#print(sentence22.draw())
	#a = "\n".join([rule.unicode_repr() for rule in sentence22.productions()])
	#print(a)
	#print(sentence7.draw())
	#b = "\n".join([rule.unicode_repr() for rule in sentence7.productions()])
	#print(b)
	#print(sentence13.draw())
	#c = "\n".join([rule.unicode_repr() for rule in sentence13.productions()])
	#print(c)
	meals = 'List me the seats on the flight to Denver'
	processSentence(meals)
mainScript()
