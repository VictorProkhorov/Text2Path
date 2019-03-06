"""
Run 4th
Splits the definitions into training test and development sets.
The split is based on the trainng test and development nodes obtained during the graph split
"""


import sys


#IO files
synset_definitions_file = sys.argv[1]
synset_definitions_dir = sys.argv[2]

target_synset_file = sys.argv[3]
target_synset_dir = sys.argv[4]

synset_definitions_filtered_file = sys.argv[5]
synset_definitions_filtered_dir = sys.argv[6]

is_wordnet = int(sys.argv[7])

def get_target_sysnets(file_, dir_):
	target_synsets = set()
	with open(dir_ + file_, 'r') as target_synset_f:
		for node_label in target_synset_f:
			node_label = node_label.rstrip().split()[0]
			target_synsets.add(node_label)
	return target_synsets

target_synsets = get_target_sysnets(target_synset_file, target_synset_dir)


with open(synset_definitions_dir + synset_definitions_file, 'r') as synset_def_f, \
	open(synset_definitions_filtered_dir + synset_definitions_filtered_file, 'w') as synset_filtered_f:
	for synset in synset_def_f:
		synset = synset.rstrip().split()
		if synset[0] in target_synsets:
			if is_wordnet == 1:
				head_word = synset[0].split('.')[0]
				head_word = ' '.join(head_word.split('_'))
				synset_filtered_f.write(' '.join([synset[0]]+[head_word]+synset[1:]) + '\n')
			else:
				synset_filtered_f.write(' '.join(synset) + '\n')




