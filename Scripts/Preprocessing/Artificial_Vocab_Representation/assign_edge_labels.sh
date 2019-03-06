EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_animal_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_animal_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE


EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_plant_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_plant_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE

EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_entity_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_entity_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE

EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/HPO/hpo_edge_list_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_edge_list_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE

EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/DO/do_edge_list_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_edge_list_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE

EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/GO/go_edge_list_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_edge_list_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE

EDGE_LIST_FILE='../../../Data/Preprocessed_Graphs/PATO/pato_edge_list_acycle.txt'
VOCAB_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_artificial_vocab.txt'
EDGE_LIST_LABEL_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_edge_list_acycle_labeled.txt'

python assign_edge_labels.py $EDGE_LIST_FILE $VOCAB_FILE $EDGE_LIST_LABEL_FILE