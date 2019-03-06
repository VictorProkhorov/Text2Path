"""
First to run
"""

import sys
import networkx as nx


def get_graph(edge_list_f, directed=True):
	if  directed == True:
		G = nx.read_edgelist(edge_list_f, create_using=nx.DiGraph(), nodetype=str, comments='#')
	else:
		G = nx.read_edgelist(edge_list_f, create_using=nx.Graph(), nodetype=str, comments='#')
	return G

def get_vocab_size(G):
    """
    The node with the largest degree defines the vocab size
    """
    nodes_degree = dict(G.degree(G.nodes()))
    max_degree_node = max(nodes_degree, key=lambda node:nodes_degree[node])
    print max_degree_node
    # degree function includes both parent node and children nodes
    # since a graph is a tree then assumption is that each
    # node has a single parent. Since we need to label only edges
    # connecting children nodes with the current node substract - 1 
    # to compensate for the parent of the node 
    return nodes_degree[max_degree_node] - 1 

def generate_vocab(vocab_size):
    """
    generates vocab of size of the max degree node e.g. 
    if vocab_size = 3 then generated vocabulary will be v1, v2, v3
    """
    vocab = []
    default_char = 'v'
    print 'vocab size:', vocab_size
    for i in xrange(1, vocab_size + 1):
        vocab.append(default_char + str(i))
    return vocab

def store_vocab(vocab, vocab_file):
    with open(vocab_file, 'w') as f:
        for symbol in vocab:
            f.write(symbol + '\n')
    return 0

if __name__ == "__main__":
    edge_list_file = sys.argv[1]
    vocab_file = sys.argv[2]
    G = get_graph(edge_list_file, directed=True)
    store_vocab(generate_vocab(get_vocab_size(G)), vocab_file)
