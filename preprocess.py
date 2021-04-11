
import hgtk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import *
from config import *


def eng_preprop(in_str):
    in_str = in_str.lower()
    in_str = in_str.replace(' ', '_')
    in_str = in_str.replace('-', '_')
    return in_str

def kor_preprop(in_str):
    in_str = in_str.replace(' ', '')
    in_str_decompose = hgtk.text.decompose(in_str)
    in_str_filter = [x for x in list(in_str_decompose) if x != DEFAULT_COMPOSE_CODE]
    in_str_join = ''.join(in_str_filter)
    return in_str_join

def preprocessing(data):
    log('> Preprocessing')
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

    params['LEN_INPUT_TEXTS'] = len(input_texts)
    return (input_texts, target_texts_inputs, target_texts)

def tokenizing(input_texts, target_texts_inputs, target_texts, rsrc_path):
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
    save_pkl(tokenizer_inputs, rsrc_path + '/'+ 'tokenizer_inputs.pkl')
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
    save_pkl(tokenizer_outputs, rsrc_path + '/' + 'tokenizer_outputs.pkl')
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
