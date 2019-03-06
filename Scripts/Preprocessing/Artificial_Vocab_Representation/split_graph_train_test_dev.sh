ECHO 'ANIMAL'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_animal_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/wordnet_edge_list_animal_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE


ECHO 'PLANT'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_plant_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/wordnet_edge_list_plant_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE


ECHO 'ENTITY'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/WordNet/wordnet_edge_list_entity_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/wordnet_edge_list_entity_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE


ECHO 'HPO'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/HPO/hpo_edge_list_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_edge_list_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE

ECHO 'DO'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/DO/do_edge_list_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_edge_list_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE



ECHO 'GO'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/GO/go_edge_list_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_edge_list_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE


ECHO 'PATO'
GOLD_GRAPH_FILE='../../../Data/Preprocessed_Graphs/PATO/pato_edge_list_acycle.txt'
TRAIN_GRAPH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_edge_list_train.txt'
LEAF_HYP_DEV_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_leaf_hyp_dev.txt'
LEAF_HYP_TEST_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_leaf_hyp_test.txt'
TRAINING_NODES_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_training_nodes.txt'
python split_graph_train_test_dev.py $GOLD_GRAPH_FILE $TRAIN_GRAPH_FILE $LEAF_HYP_DEV_FILE $LEAF_HYP_TEST_FILE $TRAINING_NODES_FILE




