"""
"""

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import time
from datetime import datetime
import argparse

import hgtk
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K            
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import (
	Model,
	model_from_json
)
from tensorflow.keras.layers import (
	Input, 
	LSTM, 
	GRU, 
	Dense, 
	Embedding,
	Bidirectional, 
	RepeatVector, 
	Concatenate, 
	Activation, 
	Dot, 
	Lambda
)

import warnings
warnings.filterwarnings('ignore')

from utils import *
from config import *
from model import Seq2seqAtt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(':: gpu', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    print(':: memory growth:' , tf.config.experimental.get_memory_growth(gpu))

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train", action="store_true", help="Train Mode")
parser.add_argument("--test", action="store_true", help="Train Mode")
args = parser.parse_args()

YEARMONTHDAY = str(datetime.fromtimestamp(time.time())).split()[0]
CUR_PATH = os.path.dirname(os.path.abspath( __file__ ))
#CUR_PATH = os.getcwd()
BACK_PATH = '/'.join(CUR_PATH.split('/')[:-3]) # back 2 times
PRETRAINED_MODEL_PATH = CUR_PATH + "/resources/2019-05-14_model.h5"


def log(*s): # multiple args
    if DEBUG_MODE:
        print(s)

def load_data(path_trans): # dataset only for this projecte (specifc form)
    log('> Loading')
    data = []
    for filename in os.listdir(path_trans):
        full_path = path_trans + '/' + filename
        each_file = open(full_path, 'r', encoding='utf-8')
        for x in each_file:
            if '#' == list(x)[0]:
                continue
            data.append(x.strip())
    return data

def eng_preprop(in_str):
    in_str = in_str.lower()
    in_str = in_str.replace(' ', '_')
    in_str = in_str.replace('-', '_')
    return in_str

def preprocessing(data):
    log('> Preprocessing')
    def kor_preprop(in_str):
        in_str = in_str.replace(' ', '')
        in_str_decompose = hgtk.text.decompose(in_str)
        in_str_filter = [x for x in list(in_str_decompose) if x != DEFAULT_COMPOSE_CODE]
        in_str_join = ''.join(in_str_filter)
        return in_str_join
    for i, _ in enumerate(data):
        source_eng = data[i].split('\t')[0]
        target_kor = data[i].split('\t')[-1]
        data[i] = eng_preprop(source_eng) + '\t' + kor_preprop(target_kor)
    return data

def input_formatting(data):
    log('> Input Formatting')
    input_texts = [] # sentence in original language
    target_texts = [] # sentence in target language
    target_texts_inputs = [] # sentence in target language offset by 1
    """
    < korean-go.txt >
    ... ... ...
    gahnite     가나이트
    garnetting  가네팅
    GANEFO      가네포
    garnett     가넷
    ... ... ...
    """
    #t = 0
    #for line in open(os.getcwd() + '/spa.txt'):
    for line in data:
        # only keep a limited number of samples
        #t += 1
        #if t > NUM_SAMPLES:
        #    break
        # input and target are separated by tab
        if '\t' not in line:
            continue
        # split up the input and translation
        input_text, translation = line.rstrip().split('\t')

        # make the target input and output
        # recall we'll be using teacher forcing
        target_text = ' '.join(list(translation)) + ' <eos>'
        target_text_input = '<sos> ' + ' '.join(list(translation))

        input_texts.append(' '.join(list(input_text)))
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)

    log(">> Number of Data:", len(input_texts))
    params['LEN_INPUT_TEXTS'] = len(input_texts)
    return (input_texts, target_texts_inputs, target_texts)

def tokenizing(input_texts, target_texts_inputs, target_texts):
    log('> Tokenizing')
    ## tokenize the inputs
    #tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_inputs = Tokenizer(num_words=params['MAX_NUM_WORDS'], filters='') # MAX_NUM_WORDS = None
    tokenizer_inputs.fit_on_texts(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
    # get the word to index mapping for input language
    word2idx_inputs = tokenizer_inputs.word_index
    params['LEN_WORD2IDX_INPUTS'] = len(word2idx_inputs)
    #print('Found %s unique input tokens.' % len(word2idx_inputs))
    # determine maximum length input sequence
    params['MAX_LEN_INPUT'] = max(len(s) for s in input_sequences)
    # save 'tokenizer_inputs' for decoding
    save_pkl(tokenizer_inputs, CUR_PATH + '/resources/tokenizer_inputs.pkl')
    log('>> Tokenizer_inputs is saved!')

    ## tokenize the outputs
    # tokenize the outputs
    # don't filter out special characters
    # otherwise <sos> and <eos> won't appear
    tokenizer_outputs = Tokenizer(num_words=params['MAX_NUM_WORDS'], filters='') # MAX_NUM_WORDS = None
    tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
    # get the word to index mapping for output language
    word2idx_outputs = tokenizer_outputs.word_index
    params['LEN_WORD2IDX_OUTPUTS'] = len(word2idx_outputs)
    #print('Found %s unique output tokens.' % len(word2idx_outputs))
    # store number of output words for later
    # remember to add 1 since indexing starts at 1 (index 0 = unknown)
    #num_words_output = len(word2idx_outputs) + 1
    # determine maximum length output sequence
    params['MAX_LEN_TARGET'] = max(len(s) for s in target_sequences) 
    # save 'tokenizer_inputs' for decoding
    save_pkl(tokenizer_outputs, CUR_PATH + '/resources/tokenizer_outputs.pkl')
    log('>> Tokenizer_outputs is saved!')

    return (input_sequences, target_sequences_inputs, target_sequences, word2idx_inputs, word2idx_outputs)

def padding(input_sequences, target_sequences_inputs, target_sequences):
    log('> Padding')
    # pad the sequences
    encoder_inputs = pad_sequences(input_sequences, maxlen=params['MAX_LEN_INPUT'])
    log(">> encoder_data.shape:", encoder_inputs.shape)
    #print("encoder_data[0]:", encoder_inputs[0])

    decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=params['MAX_LEN_TARGET'], padding='post')
    #print("decoder_data[0]:", decoder_inputs[0])
    log(">> decoder_data.shape:", decoder_inputs.shape)

    decoder_targets = pad_sequences(target_sequences, maxlen=params['MAX_LEN_TARGET'], padding='post')

    return (encoder_inputs, decoder_inputs, decoder_targets)





class Transliterator(object):

    def __init__(self):
        ## Basic process for model
        # 아래 과정을 통해 입출력 길이를 파악해야 해야만, 네트워크 파라미터 크기를 결정할 수 있음. (필수적)
        data = load_data(CUR_PATH + '/data') # dataset only for transliteration
        data = preprocessing(data)
        input_texts, target_texts_inputs, target_texts = input_formatting(data)
        input_sequences, target_sequences_inputs, target_sequences, word2idx_inputs, word2idx_outputs = tokenizing(input_texts, target_texts_inputs, target_texts)
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = padding(input_sequences, target_sequences_inputs, target_sequences)        
        
        self.seq2seq_att = Seq2seqAtt(params)
        self.seq2seq_att.build_model()

        ## Variables
        self.tokenizer_inputs = load_pkl(CUR_PATH + '/resources/tokenizer_inputs.pkl')
        self.tokenizer_outputs = load_pkl(CUR_PATH + '/resources/tokenizer_outputs.pkl')
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def _softmax_over_time(self, x):
        # make sure we do softmax over the time axis
        # expected shape is N x T x D
        assert(K.ndim(x) > 2)
        e = K.exp(x - K.max(x, axis=1, keepdims=True)) # axis=1에 주목.
        s = K.sum(e, axis=1, keepdims=True)
        return e / s

    def _stack_and_transpose(self, x): # 다시 원래의 shape로 만들기 위해.
        # 'outputs' is now a list of length Ty
        # each element is of shape (batch size, output vocab size)
        # therefore if we simply stack all the outputs into 1 tensor
        # it would be of shape T x N x D
        # we would like it to be of shape N x T x D
        # x is a list of length T, each element is a batch_size x output_vocab_size tensor
        x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
        x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
        return x


    def train(self):

        print('> Train Model Start...')
        self.model = self.seq2seq_att.e2d_model 

        decoder_targets_one_hot = np.zeros(
            (
                params['LEN_INPUT_TEXTS'],
                params['MAX_LEN_TARGET'],
                params['LEN_WORD2IDX_OUTPUTS'] + 1
            ),
            dtype='float32'
        )

        # assign the values
        for i, d in enumerate(self.decoder_targets):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1

        # train the model
        z = np.zeros((params['LEN_INPUT_TEXTS'], params['LATENT_DIM_DECODER'])) # initial [s, c]
        r = self.model.fit(
                [self.encoder_inputs, self.decoder_inputs, z, z], decoder_targets_one_hot,
                batch_size=params['BATCH_SIZE'],
                epochs=params['EPOCHS'],
                validation_split=0.15,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10)
                ] # early stopping
            )


        self.model.save_weights(CUR_PATH + '/resources/' + YEARMONTHDAY + "_model.h5")
        log(">> Saved model's weight")        
        plt.savefig(CUR_PATH + '/resources/' + 'loss_plot.png')
        plt.savefig(CUR_PATH + '/resources/' +'acc_plot.png')        

        #log('> Desgin Model for Prediction')
        #self.encoder_model = self.seq2seq_att.encoder_model 
        #self.decoder_model = self.seq2seq_att.decoder_model 

    def use_pretrained_model(self): 

        self.model = self.seq2seq_att.e2d_model
        self.model.load_weights(PRETRAINED_MODEL_PATH)

        self.encoder_model = self.seq2seq_att.encoder_model
        self.decoder_model = self.seq2seq_att.decoder_model

    def compose_hangul(self, in_str):
        # https://zetawiki.com/wiki/...
        kor_vowel_list = "ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ".split()
        temp_list = [DEFAULT_COMPOSE_CODE]
        temp_input_list = in_str[::-1].split()
        for i, x in enumerate(temp_input_list):
            #print(i, x)
            if i >= 2:
                if temp_input_list[i-2] in kor_vowel_list:
                    temp_list.append(DEFAULT_COMPOSE_CODE)
                temp_list.append(temp_input_list[i])
            else:
                temp_list.append(temp_input_list[i])
        #print(temp_list)
        out_str = hgtk.text.compose(temp_list[::-1])
        return out_str

    def decode_sequence(self, input_seq):
        # preprocessing & tokenizing & padding for input_seq
        input_seq = eng_preprop(input_seq)
        input_seq = ' '.join(list(input_seq))
        input_seq = self.tokenizer_inputs.texts_to_sequences([input_seq]) # it is array!
        input_seq = pad_sequences(input_seq, maxlen=params['MAX_LEN_INPUT'])

        # Encode the input as state vectors.
        enc_out = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        
        # Populate the first character of target sequence with the start character.
        # NOTE: tokenizer lower-cases all words
        target_seq[0, 0] = self.tokenizer_outputs.word_index['<sos>'] # word2idx_outputs

        # if we get this we break
        eos = self.tokenizer_outputs.word_index['<eos>'] # word2idx_outputs

        # [s, c] will be updated in each loop iteration
        s = np.zeros((1, params['LATENT_DIM_DECODER']))
        c = np.zeros((1, params['LATENT_DIM_DECODER']))

        # Create the translation
        output_sentence = []
        output_prob_dist = []
        for _ in range(params['MAX_LEN_TARGET']):
            o, s, c = self.decoder_model.predict([target_seq, enc_out, s, c])

            output_prob_dist.append(max(o.flatten()))

            # Get next word
            idx = np.argmax(o.flatten())

            # End sentence of EOS
            if eos == idx:
                break

            word = ''
            if idx > 0:
                word = {v:k for k, v in self.tokenizer_outputs.word_index.items()}[idx] # idx2word_trans 
                output_sentence.append(word)

            # Update the decoder input
            # which is just the word just generated
            target_seq[0, 0] = idx

        return (self.compose_hangul(' '.join(output_sentence)), np.average(output_prob_dist))





if __name__ == "__main__":

    if args.train:
        model = Transliterator()
        model.train() # train

    elif args.test:
        model = Transliterator()
        model.use_pretrained_model() # use pre-trained model
        a = model.decode_sequence('attention') # input: attention
        print(a)