import networkx as nx
import random
import shlex
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import utils, analyze_dataset
import pickle
import arff
import os


mm = {}
with open('corpus_pubtator.txt', 'r') as f:
    raw_mm = f.readlines()


i_page = 0
for line in raw_mm:  
    if line == '\n':
        i_page = 0
        continue
    
    line = line.strip()
    
    if i_page == 0:
        if '|t|' in line:
            paperID, title = line.split('|t|')
            mm[paperID] = {'abs':'', 'tag':[]}
            mm[paperID]['abs'] = title
            
    elif i_page == 1:
        if '|a|' in line:
            paperID, abst = line.split('|a|')
            mm[paperID]['abs'] += ' ' + abst
    else:
        paperID, _, _, _, raw_tags, _ = line.split('\t')
        tags = raw_tags.split(',')
        
        for tag in tags:
            if tag not in mm[paperID]['tag'] and tag != 'UnknownType':
                mm[paperID]['tag'].append(tag)
            
        
            
    
    i_page +=1

    
with open('2017AA-full/2017AA/SRDEF', 'rb') as f:
    lines = f.readlines()
    
entities = []

for line in lines:
    cats = line.strip().split(b'|')
    if cats[0] == b'STY':
        entities.append([cats[1].decode('utf-8'), cats[3].decode('utf-8')])

entities = sorted(entities, key= lambda x: len(x[1]))

relations = []
hierarchies = {}
    
for entity, hierarchy in entities:
    if '.' not in hierarchy:
        if len(hierarchy) == 1:
            relations.append(['root', entity])
            hierarchies[hierarchy] = entity
        else:
            relations.append(['T051', entity])
            hierarchies[hierarchy] = entity
    else:
        parent = hierarchies['.'.join(hierarchy.split('.')[:-1])]
        hierarchies[hierarchy] = entity
        relations.append([parent, entity])


DG = nx.DiGraph()
       

DG.add_edges_from(relations)

for node in DG.nodes:
    DG.nodes[node]['data_count'] = 0

mapping = {}
all_abstracts = {}

for paperID, values in mm.items():
    mapping[paperID] = values['tag']
    for tag in values['tag']:
        DG.nodes[tag]['data_count']+=1
    all_abstracts[paperID] = values['abs']

utils.analyze_subgraph(DG,mapping, all_abstracts, 'root', max_hop = 100, min_data = 0, export = True)