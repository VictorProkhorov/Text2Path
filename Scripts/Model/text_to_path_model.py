# many parts of the code were taken from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb

from __future__ import absolute_import, division, print_function
import os
import time
import numpy as np
import re
import random
import unicodedata
np.random.seed(1)
import tensorflow as tf
tf.enable_eager_execution()
tf.reset_default_graph()
tf.set_random_seed(1)
from keras.constraints import unit_norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.contrib.eager as tfe
from keras.layers import Bidirectional, concatenate
from keras import initializers
import keras.backend as K
from sklearn.decomposition import PCA
import argparse
import logging


def get_embeddings(path, data_w2index, embed_size):
    with open(path, 'r') as f:
        done = 0
        covered = dict()
        W = np.zeros((len(data_w2index), embed_size))
        #print("W:",W.shape)
        for line in f:
            comps = line.rstrip().split(' ')
            if comps[0] in data_w2index:
                done += 1
                covered[data_w2index[comps[0]]] = comps[0]
                for i in range(embed_size):
                    W[data_w2index[comps[0]], i] = comps[i + 1]

        missed = dict()
        for w in data_w2index:
            k = data_w2index[w]
            if k not in covered:
                missed[k] = w

        #print("Missed",len(missed),"words of", len(data_w2index))
        return W
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    #w = unicode_to_ascii(w.strip())
        
    return w


def create_dataset(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # starts with 1 to remove the name of the synset
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[1:]]  for l in lines if len(l) > 0]
    
    return word_pairs

def create_dataset_augment(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # starts with 1 to remove the name of the synset
    paths = [[l]  for l in lines if len(l) > 0]
    return paths


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
    
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            # + 1 to take into account <pad>
            self.word2idx[word] = index + 1
    
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
    
    def augment_vocab(self, new_phrases):
        vocab = set()
        for phrase in new_phrases:
            vocab.update(phrase[0].split(' '))
        new_vocab = set(vocab).difference(self.vocab)
        new_vocab = sorted(new_vocab)
        self.vocab = self.vocab + new_vocab
        initial_index = sorted(self.idx2word.keys())[-1]
        for word in new_vocab:
            initial_index += 1
            self.word2idx[word] = initial_index 
            self.idx2word[initial_index] = word 

def max_length(tensor):
    return max(len(t) for t in tensor)

def mean_length(tensor):
    return int(sum(len(t) for t in tensor) / len(tensor))


def load_dataset(path, augmetation_data_file=None, is_shift_decoder_output=False):
    # creating cleaned input, output pairs
    pairs = create_dataset(path)
    if augmetation_data_file is not None:
        extra_path_seqs = create_dataset_augment(augmetation_data_file)
        
    # index language using the class defined above    
    text_lang = LanguageIndex(txt for txt, path in pairs)
    graph_lang = LanguageIndex(path for txt, path in pairs)
    graph_lang.augment_vocab(extra_path_seqs)
    logging.debug('Number of symbols in graph: %d', len(graph_lang.vocab))
   
    # text definitions of concepts
    text_tensor = [[text_lang.word2idx[word] for word in txt.split(' ')] for txt, path in pairs]
    

    graph_tensor = [[graph_lang.word2idx[node] for node in path.split(' ')] for txt, path in pairs]
    
    graph_tensor_extra = [[graph_lang.word2idx[node] for node in path[0].split(' ')] for path in extra_path_seqs]
 
    max_length_text = mean_length(text_tensor)
    logging.debug('mean_length_text: %d', max_length_text)
    max_length_path = max_length(graph_tensor) + 1
    logging.debug('max_length_path: %d', max_length_path)
   
    
    # Padding the input and output tensor to the maximum length
    encoder_tensor = tf.keras.preprocessing.sequence.pad_sequences(text_tensor, 
                                                                 maxlen=max_length_text,
                                                                 padding='post',truncating='post')
    
    decoder_tensor = tf.keras.preprocessing.sequence.pad_sequences(graph_tensor, 
                                                                  maxlen=max_length_path, 
                                                                  padding='post',truncating='post')

    decoder_tensor_extra =  tf.keras.preprocessing.sequence.pad_sequences(graph_tensor_extra, 
                                                                  maxlen=max_length_path, 
                                                                  padding='post',truncating='post')
    return encoder_tensor, decoder_tensor, decoder_tensor_extra, text_lang, graph_lang, max_length_text, max_length_path




def rnn_dec(units): 
    return tf.keras.layers.LSTM(units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=4), recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=1), dropout=0.5, return_state=True, implementation=2)


def rnn_enc(units, go_backwards=False):
    return tf.keras.layers.LSTM(units, return_sequences=True,go_backwards=go_backwards, return_state=True, dropout=0.5,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=3), recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=2), implementation=2)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, W, is_word_pretraining=False):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        if is_word_pretraining == True:
            logging.debug('Using Pretrained Word Embeddings')
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True,embeddings_initializer=tf.keras.initializers.Constant(W), trainable=True)
        else:
            logging.debug('Using Random Word Embeddings')
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True,embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
        self.rnn_f = rnn_enc(self.enc_units, go_backwards=False)
        self.rnn_b = rnn_enc(self.enc_units, go_backwards=True)
        self.hidden_dense = tf.keras.layers.Dense(enc_units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1))
             
    def call(self, x):
        x = self.embedding(x)
        output_f, state_h_f, state_c_f = self.rnn_f(x)
        output_b, state_h_b, state_c_b = self.rnn_b(x)        
        state_h =  tf.concat([state_h_f, state_h_b], axis=-1)
        state_c =  tf.concat([state_c_f, state_c_b], axis=-1)
        output = tf.concat([output_f, tf.reverse(output_b, axis=[1])], axis=-1)
        state_h = self.hidden_dense(state_h)
        state_c = self.hidden_dense(state_c)
    
        return output, [state_h, state_c]
    
    def initialize_hidden_state(self, batch_size):
        return [tf.zeros((batch_size, self.enc_units)), tf.zeros((batch_size, self.enc_units))]

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2))
        self.rnn = rnn_dec(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=2))
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=3), activation='tanh')
        self.attention = BahdanauAttention(dec_units)
        
    def call(self, x, hidden_state, enc_output, is_attention=True):
        
        x = self.embedding(x)
      
        attention_weights = []
        
        out, h, c = self.rnn(x, initial_state = hidden_state)
        prev_states = [h, c]
       
        if is_attention:
            context_vector, attention_weights = self.attention(enc_output, hidden_state[0])
            h = tf.concat([h, context_vector], axis=-1)
            h = self.W1(h)
        out = self.fc(h)
       
        return out, prev_states, attention_weights
        
    def initialize_hidden_state(self, batch_size):
        return [tf.zeros((batch_size, self.dec_units)), tf.zeros((batch_size, self.dec_units))]

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units , kernel_initializer=tf.keras.initializers.glorot_uniform(seed=9))
    self.W2 = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=8))
    self.V = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=10))
  
  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
    
    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    
    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
    
    # attention_weights shape == (batch_size, 64, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)
    
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    
    return context_vector, attention_weights




def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)



def data_iter(data_1, data_2):
    #assert len(data_1) > len(data_2)
    while True:
        yield next(data_1), next(data_2, ())

def paired_sentenses(batch_size, inp, targ):
    loss = 0
    enc_output, enc_hidden = encoder(inp)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<SOS>']] * batch_size, 1) 
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, is_attention=True)
        loss += loss_function(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)
    return loss

def dummy_sentences(batch_size, inp):
    loss = 0
    dec_input = tf.expand_dims([targ_lang.word2idx['<SOS>']] * batch_size, 1) 
    hidden = decoder.initialize_hidden_state(batch_size)
    features = tf.expand_dims(tf.zeros((batch_size, 2*units)), 1)
    for t in range(1, inp.shape[1]):   
        predictions, hidden, _ = decoder(dec_input, hidden, features, is_attention=False)
        loss += loss_function(inp[:, t], predictions)*0.25
        dec_input = tf.expand_dims(inp[:, t], 1)
    return loss

def train(EPOCHS, encoder, decoder, optimizer, dataset_paired, dataset_aux, checkpoint):  
    with tf.device("/cpu:0"):
        for epoch in range(1, EPOCHS + 1):
            data = data_iter(enumerate(dataset_paired), enumerate(dataset_aux))
            start = time.time()
            total_loss = 0
            for batch_id, batch in enumerate(data):
                batch_paired = batch[0][1]
                inp = batch_paired[0]
                targ = batch_paired[1]
                batch_size = inp.shape[0]
                with tf.GradientTape() as tape:
                    loss = paired_sentenses(batch_size, inp, targ)
                
                variables = encoder.variables + decoder.variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
                
                batch_loss = (loss / (int(targ.shape[1]) - 1))
                total_loss += batch_loss
                     
                if batch_id % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch ,
                                                         batch_id,
                                                         batch_loss.numpy()))
                batch_aux = batch[1]
                # Explained in the paper section 2.1 last paragraph
                if len(batch_aux) > 0:           
                    inp = batch_aux[1]
                    batch_size = inp.shape[0]
                    with tf.GradientTape() as tape:
                        loss = dummy_sentences(batch_size, inp)
                    
                    variables =  decoder.variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
                
                
            print('Epoch {} Loss {:.4f}'.format(epoch,
                                        total_loss/N_BATCH))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            if epoch % 10 == 0: 
                save = checkpoint.save(file_prefix=checkpoint_prefix)
                print('save', save)
            
                

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] if i in inp_lang.word2idx else inp_lang.word2idx['<pad>'] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post', truncating='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = []

    enc_out, enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<SOS>']], 0)

    for _ in range(max_length_targ):
       
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, is_attention=True)
        
        _, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=1, sorted=False)
        predicted_id = topi[0][0].numpy()

        result.append(targ_lang.idx2word[predicted_id])

        if targ_lang.idx2word[predicted_id] == '<EOS>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot





def get_pairs_of_lang_input(lines):
    return [[s for s in l.split('\t')] for l in lines if len(l) > 0]

def read_test_data(data_file_name):
    with open(data_file_name, 'r') as f:
        lines = f.read().split('\n')
    return get_pairs_of_lang_input(lines)

def fill_batch(batch, max_length):
    batch_size = batch.shape[0]
    missing = BATCH_SIZE - batch_size
    padding = np.zeros([missing, max_length], dtype='int32')
    return tf.concat([batch, tf.convert_to_tensor(padding)], axis=0)

def f1(gold_edges, predicted_edges):
    def intersection():
        correct_predictions = 0.0
        for i in range(min(len(gold_edges), len(predicted_edges))):
            if gold_edges[i] == predicted_edges[i]:
                correct_predictions += 1.0
            else: # if make mistake return current score
                return correct_predictions
        return correct_predictions
    if len(gold_edges) == 0 and len(predicted_edges) == 0:
        return 1.
    if len(predicted_edges) > 0:
        correct_predictions = intersection()
        if int(correct_predictions) == 0: return 0 
        precision = correct_predictions / len(predicted_edges)
        recall = correct_predictions / len(gold_edges)
        return 2 * ((precision * recall) / (precision + recall))
    else: 
        return 0

def calculate_f1(gold_edges, predicted_edges):
    if '<EOS>' in gold_edges:
            gold_edges.remove('<EOS>')
    if '<EOS>' in predicted_edges:
            predicted_edges.remove('<EOS>')
    return f1(gold_edges, predicted_edges)


greedy_f1_scores = []
is_train = False
if __name__ == '__main__':
    descr = "Tensorflow (Eager) implementation for text-to-path model. In all experiments Tensorflow (CPU) 1.11.0 and python 2.7 were used."
    epil  = "See: Generating Knowledge Graph Paths from Textual Definitionsusing Sequence-to-Sequence Models [V. Prokhorov, M.T. Pilehvar, N. Collier (NAACL 2019)]"

    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--train_data', required=True,
                         help='File with training (text and path pairs) data')
    parser.add_argument('--augment_data', required=True,
                         help='File with path (only) data')
    parser.add_argument('--test_data', required=True,
                         help='File with test (text and path pairs) data')
    parser.add_argument('--checkpoint', required=True,
                         help='Directory where a trained model is stored')
    
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Size of data batches during training')

    parser.add_argument('--lstm_dims', type=int, default=128,
                        help='Number of embedding dimensions')
    
    parser.add_argument('--embed_dims', type=int, default=64,
                        help='Number of embedding dimensions')
    parser.add_argument('--is_pretrain_embed', type=bool, default=False,
                        help='Determines whether to use pretrained word embeddings or not')
    parser.add_argument('--is_logging', type=bool, default=False,
                        help='logging')
    args = parser.parse_args()
    if args.is_logging:
        logging.basicConfig(level=logging.DEBUG)

    # Loading Training Data #
    training_data_file_name=args.train_data
    data_augmentation_file_name = args.augment_data 
    input_tensor, target_tensor, decoder_tensor_extra, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(training_data_file_name, augmetation_data_file=data_augmentation_file_name)
    
    # Initialising Hyperparameters #
    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = args.batchsize 
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = args.embed_dims 
    units = args.lstm_dims 
    vocab_inp_size = len(inp_lang.word2idx)
    logging.debug('vocab_inp_size: %d', vocab_inp_size)
    vocab_tar_size = len(targ_lang.word2idx)
    pretrained_embeddings_file = '../../../Data/Auxiliary_Data/Corpora/numberbatch-en.txt'
    W = get_embeddings(pretrained_embeddings_file, inp_lang.word2idx, 300)
    pca = PCA(n_components=embedding_dim)
    W = pca.fit_transform(W)
    logging.debug('W reduced: (%d, %d)', W.shape[0], W.shape[1])
    EPOCHS = args.epochs

    # Create Model #
    encoder = Encoder(vocab_inp_size, embedding_dim, units, W, is_word_pretraining=args.is_pretrain_embed)
    decoder = Decoder(vocab_tar_size, embedding_dim, units)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

    # Set store directory #
    checkpoint_dir = args.checkpoint
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder,
                                optimizer_step=tf.train.get_or_create_global_step())

    # Create Training Dataset #
    dataset_paired = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset_paired = dataset_paired.batch(BATCH_SIZE, drop_remainder=False)
    
    dataset_aux = tf.data.Dataset.from_tensor_slices((decoder_tensor_extra)).shuffle(len(decoder_tensor_extra))
    dataset_aux = dataset_aux.batch(BATCH_SIZE, drop_remainder=False)
    
    # Choose to train a model from scratch or to load an existing
    if is_train == True:
        train(EPOCHS, encoder, decoder, dataset_paired, optimizer, dataset_aux, checkpoint)
    else:
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        logging.debug('load path %s', load_path)
        load = checkpoint.restore(load_path)
        
        
    
    # Evaluation #
    test_data_file_name=args.test_data
    dev_pairs = read_test_data(test_data_file_name)
    
    is_print_path = False # set to True to observe predicted path
    for dev_pair in dev_pairs:
        current_word = dev_pair[0]
        text_input = dev_pair[1]
        gold_path = [node for node in dev_pair[2].split(' ')][1:]
       
        decoded_path, sentence, attention_plot = evaluate(text_input, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        if is_print_path:
            print('Current Word:', current_word)
            print('Text Input:', text_input)
            print('Gold Path:', gold_path)
            print('Decoded Path:', decoded_path)
            print(''.join(['-'for _ in range(10)]))
        greedy_f1_score = calculate_f1(gold_path, decoded_path)
        greedy_f1_scores.append(greedy_f1_score)
    print('###F1###:', sum(greedy_f1_scores) / len(greedy_f1_scores))



    

    
