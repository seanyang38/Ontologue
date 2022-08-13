import networkx as nx
import shlex
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import pickle
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
import os

def load_graph(PATH_TO_GRAPH):

	i = 0
	links = []
	relations = []
	with open(PATH_TO_GRAPH) as file:
	    for line in tqdm(file):
	        if i==0:
	            i+=1
	            continue
	            
	        if 'broader' in line:
	            s, p, o, _ = shlex.split(line.rstrip().replace("'", "\\'"))
	            s = s[1:-1].split('/')[-1][9:]
	            p = p[1:-1].split('/')[-1]
	            o = o[1:-1].split('/')[-1][9:]
	            
	            links.append([o, s])
	        

	DG = nx.DiGraph()
	DG.add_edges_from(links)

	for node in DG.nodes:
	    DG.nodes[node]['data_count'] = 0

	return DG

def load_labels(PATH_TO_LABELS):
	mapping = {}
	i=0
	with open(PATH_TO_LABELS) as file:
	    for line in tqdm(file):
	        if i==0:
	            i+=1
	            continue
	            
	        if 'subject' in line:
	            s, p, o, _ = shlex.split(line.rstrip().replace("'", "\\'"))
	            s = s[1:-1].split('/')[-1].split(':')[-1]
	            p = p[1:-1].split('/')[-1]
	            o = o[1:-1].split('/')[-1].split(':')[-1]
	            

	            if s in mapping:
	                mapping[s].append(o)
	            else:
	                mapping[s] = [o]
	                


	return mapping

def load_abstracts(PATH_TO_ABSTRACTS):

    sub_abstracts = {}

    i=0
    with open(PATH_TO_ABSTRACTS) as file:
        for line in tqdm(file):
            if i==0:
                i+=1
                continue

    #         print (shlex.split(line.rstrip()))

            s, p, o, _ = shlex.split(line.rstrip().replace("'", "\\'"))
            s = s[1:-1].split('/')[-1].split(':')[-1]
            
#             if s in subjects:
                
            p = p[1:-1].split('/')[-1]
            if 'comment' in p:
                o = o.replace('@en', '')
                sub_abstracts[s] = o


    return sub_abstracts

def dfs(DG, starting_node, max_hop = 10000, min_data = 0):
    avoids = ['_in_','_by_']
    queue = []
    seen = {}
    seen[starting_node] = 0
    queue += [(b, 0) for a, b in DG.out_edges(starting_node)]
            
    
    count = 0
    while len(queue)>0:
        
        node, depth = queue.pop(-1)
        includeAvoids = False
        for avoid in avoids:
            if avoid in node:
                includeAvoids = True


        # if node not in seen and not bool(re.search(r'\d', node)) and depth+1<=max_hop and DG.nodes[node]['data_count']>=min_data:
        if node not in seen and depth+1<=max_hop and DG.nodes[node]['data_count']>=min_data:
             
            
            count+=1
            seen[node] = 0
            queue += [(b, depth+1) for a, b in DG.out_edges(node)]
    
    return count, seen

def getAbstractsFromSubs(subjects, all_abstracts):
    mapping_sub_abstract = {}
    count = 0
    for sub in subjects:
        if sub in all_abstracts:
            count+=1
#             print (sub)
            mapping_sub_abstract[sub] = all_abstracts[sub]
#         else:
#             print (sub)

    return mapping_sub_abstract

def map_data(node_dict, mapping):
    data_count = 0
    subjects = {}
    for a, cats in mapping.items():
        hasCat = False
        for b in cats:
            if b in node_dict:
                if a not in subjects:
                    subjects[a] = []
                subjects[a].append(b)
                node_dict[b]+=1
                hasCat = True
        if hasCat:
            data_count+=1

       
    return data_count, subjects

def compute_stats(DG, mapping, all_abstracts):

	for n, cats in mapping.items():
	    if n in all_abstracts:
	        for cat in cats:
	            if cat in DG.nodes:
	                if 'data_count' in DG.nodes[cat]: 
	                    DG.nodes[cat]['data_count'] +=1

def analyze_subgraph(DG, mapping, all_abstracts, starting_node, max_hop = 10000, min_data = 0, getAbstract = False, export = False):
    
    node_count, node_dict = dfs(DG, starting_node, max_hop, min_data)
    data_count, subjects = map_data(node_dict, mapping)

    mapping_sub_abstract = getAbstractsFromSubs(subjects, all_abstracts)

#     if getAbstract:
#         subHasAbstracts, abstracts = getAbstracts(subjects)
#         print (set(subjects) - set(subHasAbstracts))
    
    cc = nx.weakly_connected_components(DG.subgraph([a for a, b in node_dict.items() if b != 0 or a == starting_node]))
    
    subgraph = DG.subgraph(list([c for c in sorted(cc, key=len, reverse=True)][0]))


    degree_sequence = sorted([d for n, d in subgraph.degree()], reverse=True)
    dmax = max(degree_sequence)
    node_dict = {n:d for n, d in node_dict.items() if n in subgraph}
    data_sequence = sorted([d for n, d in node_dict.items()], reverse=True)
    y_sequence = sorted([len(d) for n, d in subjects.items()], reverse=True)
    
    print ("Node Count:", len(subgraph))
    print ("Data Count:", len(mapping_sub_abstract))
    
    
    
    if export:
        if max_hop == 10000 and min_data == 0:
            folder_path = "%s/"%(starting_node)
        elif max_hop != 10000 and min_data != 0:
            folder_path = "%s_h%d_d%d/"%(starting_node, max_hop, min_data)
        elif max_hop != 10000:
            folder_path = "%s_h%d/"%(starting_node, max_hop)
        elif min_data != 0:
            folder_path = "%s_d%d/"%(starting_node, min_data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        nx.write_edgelist(subgraph, "%s/graph.edgelist"%folder_path, data=False)
        with open("%s/abstracts.txt"%folder_path, 'w') as f:
            for sub, abstract in mapping_sub_abstract.items():
                f.write("%s\t%s\n"%(sub, abstract))
        with open('%s/data.txt'%folder_path, 'w') as f:
            for sub, nodes in subjects.items(): 
                y = '@'.join(nodes)
                f.write("%s\t%s\n"%(sub, y))
            
            
#         nx.write_gpickle(subgraph, "%s/graph.gpickle"%folder_path)    
            
        final_data = {}

        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for sub, abstract in tqdm(mapping_sub_abstract.items()):
            final_data[sub] = {}
            final_data[sub]['abstract'] = abstract
            final_data[sub]['y'] = subjects[sub]
            final_data[sub]['bert_abstract'] = bert_model.encode(abstract.strip())
            
        
        
        with open('%s/data.pkl'%folder_path, 'wb') as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    
    norm = plt.Normalize()
    colors = plt.cm.Reds(norm([node_dict[n] for n in subgraph]))
    
    
        
    
    
    for i, n in enumerate(subgraph.nodes):
        subgraph.nodes[n]['color'] = matplotlib.colors.rgb2hex(colors[i])

    net = Network(notebook=False, height="750px", width="100%")
    net.from_nx(subgraph)
    if export:
        net.show('%s/graph.html'%folder_path)
    else:
        net.show('%s.html'%starting_node)
    
#     nx.draw(subgraph, node_color=[node_dict[n] for n in subgraph], node_size=100, cmap=plt.cm.Blues)
#     plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    axes[0].bar(*np.unique(data_sequence, return_counts=True))
    axes[0].set_xlabel('data per node')
    axes[0].set_xlim([0, 100])
    axes[0].set_ylabel('node counts')

    
    axes[1].bar(*np.unique(degree_sequence, return_counts=True))
    axes[1].set_xlabel('degree')
    axes[1].set_ylabel('node counts')
    axes[1].set_xlim([0, 30])
    
    axes[2].bar(*np.unique(y_sequence, return_counts=True))
    axes[2].set_xlabel('# of y')
    axes[2].set_ylabel('data counts')
    axes[2].set_xlim([0, 10])
    fig.tight_layout()

    plt.show()
    
    return subgraph