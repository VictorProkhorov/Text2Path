
ECHO 'ANIMAL TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_animal_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/path_animal_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/extra_path_animal_train.txt'

python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'ANIMAL TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_animal_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/path_animal_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/extra_path_animal_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'ANIMAL DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_animal_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/animal_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/path_animal_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/extra_path_animal_dev.txt'

python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE



ECHO 'PLANT TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_plant_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/path_plant_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/extra_path_plant_train.txt'

python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'PLANT TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_plant_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/path_plant_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/extra_path_plant_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'PLANT DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_plant_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/plant_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/path_plant_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/extra_path_plant_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE



ECHO 'ENTITY TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_entity_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/path_entity_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/extra_path_entity_train.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE



ECHO 'ENTITY TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_entity_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/path_entity_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/extra_path_entity_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'ENTITY DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/wordnet_edge_list_entity_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/entity_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/path_entity_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/extra_path_entity_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE




ECHO 'HPO TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/path_hpo_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/extra_path_hpo_train.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'HPO TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/path_hpo_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/extra_path_hpo_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'HPO DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Graph/hpo_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/path_hpo_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/extra_path_hpo_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'DO TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/path_do_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/extra_path_do_train.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'DO TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/path_do_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/extra_path_do_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'DO DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/DO/Graph/do_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/path_do_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/DO/Definitions/extra_path_do_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE



ECHO 'GO TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/path_go_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/extra_path_go_train.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'GO TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/path_go_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/extra_path_go_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'GO DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/GO/Graph/go_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/path_go_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/GO/Definitions/extra_path_go_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'PATO TRAINING'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_training_nodes.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/path_pato_train.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/extra_path_pato_train.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'PATO TEST'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_leaf_hyp_test.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/path_pato_test.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/extra_path_pato_test.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE


ECHO 'PATO DEV'
EDGE_LIST_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_edge_list_acycle_labeled.txt'
TARGET_NODES_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Graph/pato_leaf_hyp_dev.txt'
PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/path_pato_dev.txt'
EXTRA_PATH_FILE='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/extra_path_pato_dev.txt'
python generate_target_paths.py $EDGE_LIST_FILE $TARGET_NODES_FILE $PATH_FILE $EXTRA_PATH_FILE