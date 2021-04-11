"""
"""

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import copy
import argparse

import hgtk
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K            
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

from utils import *
from config import *
from preprocess import *
from model import Seq2seqAtt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(':: gpu', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
    print(':: memory growth:' , tf.config.experimental.get_memory_growth(gpu))

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train", action="store_true", help="Train Mode")
parser.add_argument("--test", action="store_true", help="Test Mode")
parser.add_argument("--test_simple", action="store_true", help="Simple Test Mode")
args = parser.parse_args()

base_path = os.path.dirname(os.path.abspath( __file__ ))
rsrc_path = base_path + '/resources'
#pretrained_model_path = rsrc_path + "/2019-05-14_model.h5"
#pretrained_model_path = rsrc_path + '/' + SAVE_NAME + "__model.h5"
pretrained_model_path = rsrc_path + '/20210411__0_95__0_05__16__model.h5'
data_path = base_path + '/data'
train_data_path = data_path + '/train.txt'
test_data_path = data_path + '/test.txt'



class Transliterator(object):

    def __init__(self):
        self._load_data()
        self._process_data()

    def _load_data(self):
        if not os.path.isfile(train_data_path):
            raw = load_data_listdir(data_path + '/raw')
            train, test = split_data(raw, ratio=params['TRAIN_RATIO'])
            save_data(train, train_data_path)
            save_data(test, test_data_path)
        else:
            train = load_data(train_data_path)
            test = load_data(test_data_path)
        log(">> total number of data:", len(train) + len(test))
        log(">> number of train data:", len(train))
        log(">> number of test data:", len(test))
        self.train = train
        self.test = test

    def _process_data(self):
        ## Basic process for model
        # 아래 과정을 통해 입출력 길이를 파악해야 해야만, 네트워크 파라미터 크기를 결정할 수 있음. (필수적)
        train = preprocessing(self.train)
        input_texts, target_texts_inputs, target_texts = input_formatting(train)
        tkn_info = tokenizing(input_texts, target_texts_inputs, target_texts, rsrc_path)
        input_sequences = tkn_info[0]
        target_sequences_inputs = tkn_info[1]
        target_sequences = tkn_info[2]
        word2idx_inputs = tkn_info[3]
        word2idx_outputs = tkn_info[4]
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = padding(input_sequences, target_sequences_inputs, target_sequences)        
        
        self.seq2seq_att = Seq2seqAtt(params)
        self.seq2seq_att.build_model()

        ## Variables
        self.tokenizer_inputs = load_pkl(rsrc_path + '/' + 'tokenizer_inputs.pkl')
        self.tokenizer_outputs = load_pkl(rsrc_path + '/' + 'tokenizer_outputs.pkl')
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def run_train(self):

        log('> Train Model Start...')
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
                validation_split=params['VALID_RATIO'],
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=params['PATIENCE'])
                ] # early stopping
            )

        log(">> Save model's weight") 
        self.model.save_weights(rsrc_path + '/' + SAVE_NAME + "__model.h5")
        plt.savefig(rsrc_path + '/' + SAVE_NAME + '__loss_plot.png')
        plt.savefig(rsrc_path + '/' + SAVE_NAME + '__acc_plot.png')        

        #log('> Desgin Model for Prediction')
        #self.encoder_model = self.seq2seq_att.encoder_model 
        #self.decoder_model = self.seq2seq_att.decoder_model 

    def use_pretrained_model(self): 

        self.model = self.seq2seq_att.e2d_model
        self.model.load_weights(pretrained_model_path)

        self.encoder_model = self.seq2seq_att.encoder_model
        self.decoder_model = self.seq2seq_att.decoder_model

    def _compose_hangul(self, in_str):
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

        return (self._compose_hangul(' '.join(output_sentence)), np.average(output_prob_dist))


def accuracy(model, test):
    cnt = 0
    full_hit_cnt = 0
    sylb_hit_cnt = 0
    jamo_hit_cnt = 0
    raw_test = copy.deepcopy(test)
    raw_test = [x.split('\t') for x in raw_test]
    test = preprocessing(test)
    test = [x.split('\t') for x in test]
    print(test[:5])

    for i, (eng, kor) in enumerate(test):
        cnt += 1
        if cnt % 1000 == 0:
            print('cur cnt: ', cnt)
        #print(eng, kor)

        pred_kor = ""

        if WORD_SPLIT:
            for x in eng.split('_'):
                y, _ = model.decode_sequence(x)
                pred_kor += y
        else:
            pred_kor, _ = model.decode_sequence(eng)


        true_kor = raw_test[i][-1]
        #print(pred_kor)
        #print(raw_test[i][-1])

        # full
        if pred_kor == true_kor:
            full_hit_cnt += 1

        # sylb
        min_sylb_length = min(len(list(pred_kor)), len(list(true_kor))) # 짧은 길이 선택
        temp_sylb_cnt = 0
        for i in range(0, min_sylb_length):
            if pred_kor[i] == true_kor[i]:
                temp_sylb_cnt += 1
        avg_temp_sylb_cnt = temp_sylb_cnt / min_sylb_length
        sylb_hit_cnt += avg_temp_sylb_cnt

        # jamo
        pred_kor_jamo_list = list(hgtk.text.decompose(pred_kor))
        true_kor_jamo_list = list(hgtk.text.decompose(true_kor))
        min_jamo_length = min(len(list(pred_kor_jamo_list)), len(list(true_kor_jamo_list))) # 짧은 길이 선택
        temp_jamo_cnt = 0
        for i in range(0, min_jamo_length):
            if pred_kor_jamo_list[i] == true_kor_jamo_list[i]:
                temp_jamo_cnt += 1
        avg_temp_jamo_cnt = temp_jamo_cnt / min_jamo_length
        jamo_hit_cnt += avg_temp_jamo_cnt

        #break
    print('total cnt: ', cnt)
    print('full hit cnt: ', full_hit_cnt, round((full_hit_cnt / cnt) * 100, 2))
    print('sylb hit cnt: ', sylb_hit_cnt, round((sylb_hit_cnt / cnt) * 100, 2))
    print('jamo hit cnt: ', jamo_hit_cnt, round((jamo_hit_cnt / cnt) * 100, 2))



if __name__ == "__main__":

    model = Transliterator()

    if args.train:
        model.run_train() # train

    elif args.test_simple:
        model.use_pretrained_model() # use pre-trained model
        test_list = ['attention', 'tokenizer', 'transliterator', 'suddenly', 'mecab', 'adidas', 'nike']
        for x in test_list:
            print(model.decode_sequence(x)) # input: attention

    elif args.test:
        model.use_pretrained_model()
        accuracy(model, 
                 model.test
        )


# train ratio = 1.0
# dev ratio = 0.15

# total cnt:  5670
# full hit cnt:  3279 57.83
# sylb hit cnt:  4550.605770618269 80.26
# jamo hit cnt:  4970.415985451947 87.66


# train ratio = 0.9
# dev ratio = 0.1

# total cnt:  5670
# full hit cnt:  3038 53.58
# sylb hit cnt:  4265.99119491619 75.24
# jamo hit cnt:  4756.531747772234 83.89


# train ratio = 0.95
# dev ratio = 0.05
# batch size = 32

# total cnt:  2835
# full hit cnt:  1510 53.26
# sylb hit cnt:  2133.804478854481 75.27
# jamo hit cnt:  2372.392873615778 83.68


# train ratio = 0.95
# dev ratio = 0.05
# patience = 15
# batch size = 16

# total cnt:  2835
# full hit cnt:  1522 53.69
# sylb hit cnt:  2160.215223665229 76.2
# jamo hit cnt:  2386.930456557307 84.2