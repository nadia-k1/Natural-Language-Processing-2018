import nltk
import nltk.grammar
from nltk.grammar import read_grammar
from nltk import tree, CFG, nonterminals, induce_pcfg

from nltk.corpus import treebank
from nltk import treetransforms
from nltk.parse import pchart


#==============================================================================
# Grammar
#==============================================================================
rules = """
S -> NP VP
S -> Aux NP VP
S -> VP
S -> IVP 
NP -> NP PP
NP -> Pronoun
NP -> Proper-Noun
NP -> Det Nominal 
Nominal -> Noun 
Nominal -> Nominal Noun
Nominal -> Nominal PP 
IVP -> IVerb NP NP 
IVerb ->  NP NP PP
VP -> V NP
VP -> Verb
VP -> Verb NP
VP -> Verb NP PP
VP -> Verb PP
VP -> VP PP
verb -> VP
PP -> Preposition NP
Det -> 'the'
N -> 'students' | 'subject'
V -> 'like' | 'love'
NP -> Det N | 'NLP' | 'I'     
Det -> 'that' | 'this' | 'the' | 'a'
Noun -> 'book' | 'flight' | 'meal' | 'money' | 'meals'
Verb -> 'book' | 'include' | 'prefer' | 'Show'
IVerb -> 'Show'
Pronoun -> 'I' | 'she' | 'me'
Proper-Noun -> 'Houston' | 'NWA' | 'SF'
Aux -> 'does'
Preposition -> 'from' | 'to' | 'on' | 'near' |'through'
"""

grammar = nltk.CFG.fromstring(rules)

parser = nltk.ChartParser(grammar)

sentence = "show me the meals on the flight from Phoenix"

#==============================================================================
# Functions
#==============================================================================
def loadData(path):
	with open(path,'r') as f:
		data = f.read().split('\n')
	return data

def getTreeData(data):
	return map(lambda s: tree.Tree.fromstring(s), data)

def printParses(parses):
	for trees in parses:
		print(trees)

def processSentence(sentence):
	sentenceList = sentence
	if isinstance(sentence,str):
		sentenceList = sentence.split(' ')
	print ('Original sentence: ' + ' '.join(sentenceList))
	printParses(allParses(sentenceList))
    
def allParses(sentenceList):
	return parser.parse(sentenceList)
#==============================================================================
#  Main script
#==============================================================================
print("loading data..")
data = loadData('parseTrees.txt')
treeData = getTreeData(data)
print ("done loading data\n\n")

start = nltk.Nonterminal("S")
treeProduction = []
for items in treeData:
    rulesCFG = items.productions() 
    treeProduction += rulesCFG
#grammar = CFG(start, treeProduction)
#print(grammar)

grammar = nltk.induce_pcfg(start, treeProduction)
parser = nltk.InsideChartParser(grammar)
#parser = nltk.ViterbiParser(grammar)
for item in parser.parse((sentence).split()):
    print(item)

#print(grammar)
