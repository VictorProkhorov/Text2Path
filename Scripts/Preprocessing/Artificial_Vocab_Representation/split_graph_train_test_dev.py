"""
run 3rd
This script splits graph into training test and development sets.
To do the split, first all the leaves from a tree graph are removed.
This leaves form a set that will be later divided into training and development.
All the remaining nodes in the graph are the training set.
"""
import sys
import math
import numpy as np
import networkx as nx

np.random.seed(9)

def get_graph(edge_list_f, directed=True):
	if  directed == True:
		G = nx.read_edgelist(edge_list_f, create_using=nx.DiGraph(), nodetype=str, comments='#')
	else:
		G = nx.read_edgelist(edge_list_f, create_using=nx.Graph(), nodetype=str, comments='#')
	return G

def get_all_leaves(G):
    leaves = []
    node_degrees = dict(G.degree(G.nodes()))
    for node in node_degrees:
        if node_degrees[node] == 1:
            leaves.append(node)
    return leaves

def leaves_to_remove(leaves, sample_size):
    return np.random.choice(leaves, sample_size,replace=False)

def remove_leaves_from_graph(G, leaves):
    for leaf in leaves:
        G.remove_node(leaf)
    return G

def store_leaf_hyper_pairs(G, leaves, leaf_hyp_file):
    with open(leaf_hyp_file, 'w') as f:
        for leaf in leaves:
            f.write(leaf + ' ' + list(G.predecessors(leaf))[0]+ '\n')
    return 0


def store_train_labels(G, nodes_file):
    with open(nodes_file, 'w') as f:
        for node in G.nodes():
            f.write(node + '\n')
    return 0

def test_dev_node_split(leaves, dev_set_size):
    dev_set = set(np.random.choice(leaves, dev_set_size, replace=False))
    test_set = set(leaves).difference(dev_set)
    print 'test set size', len(test_set)
    print 'dev set size', len(dev_set)
    return test_set, dev_set

def identify_root_node(G):
	for node in G.nodes():
		if len(list(G.predecessors(node))) == 0:
			return node

	print('Warning the root node has not been found')
	return None

def remove_vocab_leaves_from_sampling(leaves, vocab_nodes):
    assert isinstance(vocab_nodes, set)
    return [leaf for leaf in leaves if len(set(G.predecessors(leaf)).intersection(vocab_nodes)) == 0]

def get_vocab_nodes(G, vocab_size):
    return set([node for node in G.nodes() if G.degree(node) == vocab_size])
    
def get_neigbours_of_vocab_nodes(vocab_nodes, G):
    neighbours = list()
    for node in vocab_nodes:
        neighbours.append(list(G.neighbors(node)))
    # flatten
    return set([node for nodes in neighbours for node in nodes])


def get_vocab_size(G):
    """
    The node with the largest degree defines the vocab size
    
    """
    nodes_degree = dict(G.degree(G.nodes()))
    max_degree_node = max(nodes_degree, key=lambda node:nodes_degree[node])
    print max_degree_node
    return nodes_degree[max_degree_node]

if __name__ == '__main__':
    gold_graph_file = sys.argv[1]
    train_graph_file = sys.argv[2]
    leaf_hyp_dev_file = sys.argv[3]
    leaf_hyp_test_file = sys.argv[4]
    training_nodes_file = sys.argv[5]

    G = get_graph(gold_graph_file, directed=True)
    vocab_size = get_vocab_size(G)
    vocab_nodes = get_vocab_nodes(G, vocab_size)
    print ('Vocab Nodes', vocab_nodes)
    neigbours_of_vocab_nodes = get_neigbours_of_vocab_nodes(vocab_nodes, G)

    print ('neigbours of vocab Nodes', neigbours_of_vocab_nodes)
    all_leaves = get_all_leaves(G)
    print('number of leaves before vocab leaves removal:', len(all_leaves))
    all_leaves_ = remove_vocab_leaves_from_sampling(all_leaves, vocab_nodes)
    print('number of leaves after vocab leaves removal:', len(all_leaves_))
    leaves_difference = set(all_leaves).difference(set(all_leaves_))
    print('leaves defference', leaves_difference)
    print ('all removed leaves are neigbours of vocab nodes', neigbours_of_vocab_nodes.issuperset(leaves_difference) )
    print len(all_leaves_)
    sample_size = int(math.ceil(len(all_leaves_) * 0.1))
    dev_set_size = int(math.ceil(sample_size * 0.1))
    rm_leaves = leaves_to_remove(all_leaves_, sample_size)
    print '***Statistics before sampling***'
    print('Gold graph is connected', nx.is_weakly_connected(G))
    print ('root gold graph:', identify_root_node(G))
    print ('Sample size', sample_size)
    print ('dev set size', dev_set_size)
    print('number of leaves to remove: ', len(rm_leaves))
    print('number of nodes in original graph: ', len(G.nodes()))
    print('expected number of nodes after removal: ', len(G.nodes()) - sample_size)
    print('total number of edges: ', len(G.edges()))
   
    
   
    
    test_set, dev_set = test_dev_node_split(rm_leaves, dev_set_size)
    store_leaf_hyper_pairs(G, test_set, leaf_hyp_test_file)
    store_leaf_hyper_pairs(G, dev_set, leaf_hyp_dev_file)
   
    G_train = remove_leaves_from_graph(G, rm_leaves)
    store_train_labels(G_train, training_nodes_file)
    nx.write_edgelist(G_train, train_graph_file, data=False)
    print '***Statistics after sampling***'
    print ('root train graph:', identify_root_node(G_train))
    print('Sampled graph is connected', nx.is_weakly_connected(G_train))
    print('number of nodes in reduced graph: ', len(G_train.nodes()))
    print('number of edges in the reduced graph: ', len(G_train.edges()))
  
    