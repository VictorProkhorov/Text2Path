ECHO 'GRAPH PATO; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/PATO/Definitions/pato_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/PATO/Definitions/extra_path_pato_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/PATO/Definitions/pato_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_pato_pretrained_word_embed'
LSTM_DIM=128
IS_LOGGING=True

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM --is_logging $IS_LOGGING





ECHO 'GRAPH WORDNET ANIMAL; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/animal_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/extra_path_animal_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Animal/animal_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_animal_pretrained_word_embed'
LSTM_DIM=128

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM




ECHO 'GRAPH WORDNET PLANT; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/plant_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/extra_path_plant_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Plant/plant_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_plant_pretrained_word_embed'
LSTM_DIM=128

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM




ECHO 'GRAPH HDO; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/DO/Definitions/do_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/DO/Definitions/extra_path_do_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/DO/Definitions/do_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_do_pretrained_word_embed'
LSTM_DIM=128

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM




ECHO 'GRAPH HPO; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/HPO/Definitions/hpo_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/HPO/Definitions/extra_path_hpo_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/HPO/Definitions/hpo_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_hpo_pretrained_word_embed'
LSTM_DIM=128

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM



ECHO 'GRAPH GO; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/GO/Definitions/go_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/GO/Definitions/extra_path_go_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/GO/Definitions/go_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_go_pretrained_word_embed'
LSTM_DIM=256

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM



ECHO 'GRAPH WORDNET ENTITY; Model: text2edges*'
TRAIN_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/entity_train_corpus.txt'
AUGMENT_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/extra_path_entity_train.txt'
TEST_DATA='../../Data/Artificial_Vocab_Representation/WordNet/Definitions/Entity/entity_test_corpus.txt'
CHECKPOINT='./Trained_Models/training_checkpoints_graph_entity_pretrained_word_embed'
LSTM_DIM=256

python text_to_path_model.py --train_data $TRAIN_DATA --augment_data $AUGMENT_DATA --test_data $TEST_DATA --checkpoint $CHECKPOINT --lstm_dims $LSTM_DIM



