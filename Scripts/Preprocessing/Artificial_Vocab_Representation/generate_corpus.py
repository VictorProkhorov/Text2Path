"""
run 6th
"""
import sys
sys.path.insert(0,'../')


def get_node_def(definitions_file):
	definitions = dict()
	with open(definitions_file, 'r') as f:
		for definition in f:
			definition = definition.rstrip().split()
			label = definition[0]
			#name = label.split('.')[0] # ignore POS
			definitions[label] = ' '.join(definition[1:])
	return definitions

def get_node_path(paths_file):
    paths = dict()
    with open(paths_file, 'r') as f:
        for path in f:
            path = path.rstrip().split()
            label = path[0]
            paths[label] = ' '.join(path[1:])
	return paths

def make_data_set(dataset_file, paths, definitions):
	with open(dataset_file, 'w') as f:
		for target_node in paths:
			path = paths[target_node]
			definition = definitions[target_node]
			f.write(target_node + '\t' + definition + '\t' + path + '\n')
	return None

if __name__ == '__main__':
    paths_file = sys.argv[1]
    definitions_file = sys.argv[2]
    dataset_file = sys.argv[3]
    
    paths = get_node_path(paths_file)
    definitions = get_node_def(definitions_file)
    make_data_set(dataset_file, paths, definitions)
