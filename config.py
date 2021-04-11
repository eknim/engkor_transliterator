
import time
import datetime

#YEARMONTHDAY = str(datetime.fromtimestamp(time.time())).split()[0]
YEARMONTHDAY = datetime.datetime.now().strftime("%Y%m%d")

DEBUG_MODE = True
WORD_SPLIT = False
DEFAULT_COMPOSE_CODE = 'ᴥ'

def log(*s): # multiple args
    if DEBUG_MODE:
        print(s)

params = {
    'PATIENCE' : 20,
    'TRAIN_RATIO' : 0.95,
    'VALID_RATIO' : 0.05,
    'BATCH_SIZE' : 16, #64
    'EPOCHS' : 300, #100, # Max Epochs
    'LATENT_DIM' : 300, #256
    'LATENT_DIM_DECODER' : 300, #256 # idea: make it different to ensure things all fit together properly!
    'EMBEDDING_DIM' : 150, #100
    #'MAX_SEQUENCE_LENGTH' = 100,
    'MAX_NUM_WORDS' : None, #20000

    # below values are saved after training process
    # for using pretrained model
    'LEN_INPUT_TEXTS' : None, # 56699, # = NUM_SAMPLES
    'MAX_LEN_INPUT' : None, # 52,
    'MAX_LEN_TARGET' : None, # 47,
    'LEN_WORD2IDX_INPUTS' : None, # 33,
    'LEN_WORD2IDX_OUTPUTS' : None, # 43,
}

SAVE_NAME_list = [
    YEARMONTHDAY,
    str(params['TRAIN_RATIO']),
    str(params['VALID_RATIO']),
    str(params['BATCH_SIZE']),
]
SAVE_NAME = '__'.join(SAVE_NAME_list)
SAVE_NAME = SAVE_NAME.replace('.', '_')