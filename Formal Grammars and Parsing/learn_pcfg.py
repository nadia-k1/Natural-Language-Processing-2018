from __future__ import division
import nltk
from nltk import tree,CFG, nonterminals
from collections import defaultdict
import sys


def loadData(path):
	with open(path,'r') as f:
		data = f.read().split('\n')
	return data

def getTreeData(data):
	return map(lambda s: tree.Tree.fromstring(s), data)


freqs = defaultdict(int)
condCounts = defaultdict(int)

data = loadData('parseTrees.txt')
treeData = getTreeData(data)
print ("done loading data\n\n")

start = nltk.Nonterminal("S")
treeProduction = []
for items in treeData:
 for (x,y) in items.productions():
    freqs[(x,y)] += 1
    condCounts[x] += 1

for (x,y), freq in freqs.iteritems():
    p = freq / condCounts[x]
    print ("%s -> %s # %.4f" % (x,y,p))
