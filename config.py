

DEBUG_MODE = True
DEFAULT_COMPOSE_CODE = 'ᴥ'

params = {
    'BATCH_SIZE' : 32, #64
    'EPOCHS' : 100, #100,
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

