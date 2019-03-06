"""
run 2th
"""
import sys
sys.path.insert(0,'../')
import networkx as nx
from nltk.corpus import wordnet as wn

def get_graph(edge_list_f, directed=True):
	if  directed == True:
		G = nx.read_edgelist(edge_list_f, create_using=nx.DiGraph(), nodetype=str, comments='#')
	else:
		G = nx.read_edgelist(edge_list_f, create_using=nx.Graph(), nodetype=str, comments='#')
	return G


def identify_root_node(G):
	for node in G.nodes():
		if len(list(G.predecessors(node))) == 0:
			return node
	else:
		print('Warning the root node has not been found')
		return None


def generate_path(G, root_node, target_nodes, path_file, extra_path_file):
    
    with open(path_file, 'w') as f, open(extra_path_file, 'w') as ex_f:
        for node in target_nodes:
            path = nx.shortest_path(G, root_node, node)
            path[0] = '<SOS>' # replace root node with the start symbol
            if G.degree(node) == 1: 
                ex_f.write(' '.join(path + ['<EOS>']) + '\n')
                          
            path.pop()
            path = path + ['<EOS>']
            # find a path between the root and the target node
            f.write(' '.join( [node]+ path) + '\n') 
    return 0


def get_target_nodes(target_nodes_file):
    target_nodes = []
    with open(target_nodes_file, 'r') as f:
        for node in f:
            node = node.rstrip().split()
            target_nodes.append(node[0])
    return target_nodes


if __name__ == '__main__':
    edge_list_file = sys.argv[1]
    target_nodes_file = sys.argv[2]
    path_file = sys.argv[3]
    extra_path_file = sys.argv[4]
    G = get_graph(edge_list_file)
    root = identify_root_node(G)
    target_nodes = get_target_nodes(target_nodes_file)
    if root in target_nodes:
        target_nodes.remove(root)
    generate_path(G, root, target_nodes, path_file, extra_path_file)
	
