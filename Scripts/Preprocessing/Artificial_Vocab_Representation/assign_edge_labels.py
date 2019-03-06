"""
Run 2nd
"""
import sys
import networkx as nx


def get_graph(edge_list_f, directed=True):
	if  directed == True:
		G = nx.read_edgelist(edge_list_f, create_using=nx.DiGraph(), nodetype=str, comments='#')
	else:
		G = nx.read_edgelist(edge_list_f, create_using=nx.Graph(), nodetype=str, comments='#')
	return G


def load_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r') as f:
        for letter in f:
            letter = letter.rstrip()
            vocab.append(letter)
    return vocab


def label_edges(G, vocab):
    labels = dict()
    for node in G.nodes():
        neighbours = list(G.neighbors(node))
        for idx, neigbour in enumerate(neighbours):
            labels[(node, neigbour)] = vocab[idx]
    
    nx.set_edge_attributes(G, labels, 'labels')



if __name__ == "__main__":
    edge_list_file = sys.argv[1]
    vocab_file = sys.argv[2]
    edge_list_label_file = sys.argv[3]
    G = get_graph(edge_list_file, directed=True)
    vocab = load_vocab(vocab_file)
    label_edges(G, vocab)

    nx.write_edgelist(G, edge_list_label_file)


