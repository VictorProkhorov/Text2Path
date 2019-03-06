ECHO 'ANIMAL'
### TRAIN SET ###
SYNS_DEF_FILE='wordnet_node_def_animal_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='animal_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/'

SYNS_DEF_FILT_FILE='animal_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='wordnet_node_def_animal_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='animal_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/'

SYNS_DEF_FILT_FILE='animal_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='wordnet_node_def_animal_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='animal_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Animal/'

SYNS_DEF_FILT_FILE='animal_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


ECHO 'PLANT'
### TRAIN SET ###
SYNS_DEF_FILE='wordnet_node_def_plant_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='plant_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/'

SYNS_DEF_FILT_FILE='plant_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='wordnet_node_def_plant_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='plant_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/'

SYNS_DEF_FILT_FILE='plant_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='wordnet_node_def_plant_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='plant_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Plant/'

SYNS_DEF_FILT_FILE='plant_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


ECHO 'ENTITY'
### TRAIN SET ###
SYNS_DEF_FILE='wordnet_node_def_entity_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='entity_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/'

SYNS_DEF_FILT_FILE='entity_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='wordnet_node_def_entity_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='entity_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/'

SYNS_DEF_FILT_FILE='entity_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='wordnet_node_def_entity_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/WordNet/'

TARGET_SYN_FILE='entity_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Graph/Entity/'

SYNS_DEF_FILT_FILE='entity_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/'
IS_WORDNET='1'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET




ECHO 'HPO'
### TRAIN SET ###
SYNS_DEF_FILE='hpo_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/HPO/'

TARGET_SYN_FILE='hpo_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Graph/'

SYNS_DEF_FILT_FILE='hpo_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='hpo_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/HPO/'

TARGET_SYN_FILE='hpo_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Graph/'

SYNS_DEF_FILT_FILE='hpo_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='hpo_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/HPO/'

TARGET_SYN_FILE='hpo_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Graph/'

SYNS_DEF_FILT_FILE='hpo_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/HPO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET



ECHO 'DO'
### TRAIN SET ###
SYNS_DEF_FILE='do_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/DO/'

TARGET_SYN_FILE='do_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/DO/Graph/'

SYNS_DEF_FILT_FILE='do_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/DO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='do_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/DO/'

TARGET_SYN_FILE='do_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/DO/Graph/'

SYNS_DEF_FILT_FILE='do_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/DO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='do_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/DO/'

TARGET_SYN_FILE='do_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/DO/Graph/'

SYNS_DEF_FILT_FILE='do_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/DO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

ECHO 'GO'
### TRAIN SET ###
SYNS_DEF_FILE='go_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/GO/'

TARGET_SYN_FILE='go_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/GO/Graph/'

SYNS_DEF_FILT_FILE='go_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/GO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='go_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/GO/'

TARGET_SYN_FILE='go_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/GO/Graph/'

SYNS_DEF_FILT_FILE='go_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/GO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='go_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/GO/'

TARGET_SYN_FILE='go_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/GO/Graph/'

SYNS_DEF_FILT_FILE='go_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/GO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET



ECHO 'PATO'
### TRAIN SET ###
SYNS_DEF_FILE='pato_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/PATO/'

TARGET_SYN_FILE='pato_training_nodes.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Graph/'

SYNS_DEF_FILT_FILE='pato_train_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET

### TEST SET ###
SYNS_DEF_FILE='pato_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/PATO/'

TARGET_SYN_FILE='pato_leaf_hyp_test.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Graph/'

SYNS_DEF_FILT_FILE='pato_test_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET


### DEV SET ###
SYNS_DEF_FILE='pato_node_def_prep.txt'
SYNS_DEF_DIR='../../../Data/Preprocessed_Graphs/PATO/'

TARGET_SYN_FILE='pato_leaf_hyp_dev.txt'
TARGET_SYN_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Graph/'

SYNS_DEF_FILT_FILE='pato_dev_node_definitions.txt'
SYNS_DEF_FILE_DIR='../../../Data/Artificial_Vocab_Representation/PATO/Definitions/'
IS_WORDNET='0'
python split_def_train_test_dev.py $SYNS_DEF_FILE $SYNS_DEF_DIR\
                                $TARGET_SYN_FILE $TARGET_SYN_DIR\
                                $SYNS_DEF_FILT_FILE $SYNS_DEF_FILE_DIR $IS_WORDNET
