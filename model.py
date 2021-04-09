
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



class Seq2seqAtt(object):

	def __init__(self, args):
		self.args = args
		pass

	def build_model(self):

		###################
		### Encoder
		###################
		# definition
		encoder_inputs = Input(shape=(self.args['MAX_LEN_INPUT'],))
		encoder_embed = Embedding(self.args['LEN_WORD2IDX_INPUTS'] + 1,
									self.args['EMBEDDING_DIM'],
									#weights=[embedding_matrix],
									input_length=self.args['MAX_LEN_INPUT'],
									#trainable=True
						)
		encoder_bilstm = Bidirectional(
							LSTM(self.args['LATENT_DIM'],
								return_sequences=True,
								# dropout=0.5 # dropout not available on gpu
						))

		# pipeline
		encoder_x = encoder_embed(encoder_inputs)
		encoder_outputs = encoder_bilstm(encoder_x)

		###################
		### Decoder
		###################

		# definition
		decoder_inputs = Input(shape=(self.args['MAX_LEN_TARGET'],)) # teacher forcing input
		decoder_embed = Embedding(self.args['LEN_WORD2IDX_OUTPUTS'] + 1, 
										self.args['EMBEDDING_DIM']
							)

		# pipeline
		decoder_x = decoder_embed(decoder_inputs)


		def _softmax_over_time(x):
			# make sure we do softmax over the time axis
			# expected shape is N x T x D
			assert(K.ndim(x) > 2)
			e = K.exp(x - K.max(x, axis=1, keepdims=True)) # axis=1에 주목.
			s = K.sum(e, axis=1, keepdims=True)
			return e / s

		# ATTENTION
		# Attention layers need to be global (전역 변수) because they will be repeated Ty times at the decoder
		attn_repeat_layer = RepeatVector(self.args['MAX_LEN_INPUT'])
		attn_concat_layer = Concatenate(axis=-1)
		attn_dense1 = Dense(10, activation='tanh')
		attn_dense2 = Dense(1, activation=_softmax_over_time)
		attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]

		# define the rest of the decoder (after attention)
		decoder_lstm = LSTM(self.args['LATENT_DIM_DECODER'], return_state=True)
		decoder_dense = Dense(self.args['LEN_WORD2IDX_OUTPUTS'] + 1, activation='softmax')

		initial_s = Input(shape=(self.args['LATENT_DIM_DECODER'],), name='s0')
		initial_c = Input(shape=(self.args['LATENT_DIM_DECODER'],), name='c0')
		context_last_word_concat_layer = Concatenate(axis=2) # for teacher forcing

		# Unlike previous seq2seq, we cannot get the output all in one step
		# Instead we need to do Ty steps And in each of those steps, we need to consider all Tx h's

		# s, c will be re-assigned in each iteration of the loop
		s = initial_s
		c = initial_c

		def _one_step_attention(h, st_1):
			# h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
			# st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

			# copy s(t-1) Tx times
			# now shape = (Tx, LATENT_DIM_DECODER)
			st_1 = attn_repeat_layer(st_1)

			# Concatenate all h(t)'s with s(t-1)
			# Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
			x = attn_concat_layer([h, st_1])

			# Neural net first layer
			x = attn_dense1(x)

			# Neural net second layer with special softmax over time
			alphas = attn_dense2(x)

			# "Dot" the alphas and the h's
			# Remember a.dot(b) = sum over a[t] * b[t]
			context = attn_dot([alphas, h])

			return context


		# collect outputs in a list at first
		outputs = []
		# 원래 LSTM은 내부적으로 아래와 같은 for문을 진행하지만, 여기서 우리는 Context를 계산하기 위해서 manual하게 for문을 구성함.
		for t in range(self.args['MAX_LEN_TARGET']): # Ty times

			######################################################
			## `one_step_attention` function !
			# get the context using attention
			context = _one_step_attention(encoder_outputs, s)
  
			# we need a different layer for each time step
			selector = Lambda(lambda x: x[:, t:t+1]) # 해당 time 벡터만 추출. 우리는 layer-wise로 코딩해야 되기 때문에 lambda를 사용.
			xt = selector(decoder_x)
	
			# combine 
			decoder_lstm_input = context_last_word_concat_layer([context, xt])

			# pass the combined [context, last word] into the LSTM
			# along with [s, c]
			# get the new [s, c] and output
			o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

			# final dense layer to get next word prediction
			decoder_outputs = decoder_dense(o)
			outputs.append(decoder_outputs)

		def _stack_and_transpose(x): # 다시 원래의 shape로 만들기 위해.
			# 'outputs' is now a list of length Ty
			# each element is of shape (batch size, output vocab size)
			# therefore if we simply stack all the outputs into 1 tensor
			# it would be of shape T x N x D
			# we would like it to be of shape N x T x D
			# x is a list of length T, each element is a batch_size x output_vocab_size tensor
			x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
			x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
			return x

		# pipeline
		stacker = Lambda(_stack_and_transpose)
		decoder_outputs = stacker(outputs)

		#########
		### Encoder&Decoder Model
		self.e2d_model = Model(
			inputs=[
				encoder_inputs,
				decoder_inputs,
				initial_s, 
				initial_c,
			],
			outputs=decoder_outputs)

		# compile the model
		self.e2d_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


		########################### For Prediction ###########################

		###################
		### t1 Encoder
		###################

		self.encoder_model = Model(encoder_inputs, encoder_outputs)

		###################
		### t1 Decoder
		###################

		# next we define a T=1 decoder model
		encoder_outputs_as_input = Input(shape=(self.args['MAX_LEN_INPUT'], self.args['LATENT_DIM'] * 2,))
		decoder_inputs_single = Input(shape=(1,))
		decoder_inputs_single_x = decoder_embed(decoder_inputs_single)

		# no need to loop over attention steps this time because there is only one step
		context = _one_step_attention(encoder_outputs_as_input, initial_s)

		# combine context with last word
		decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

		# lstm and final dense
		o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
		decoder_outputs = decoder_dense(o)

		# note: we don't really need the final stack and tranpose
		# because there's only 1 output
		# it is already of size N x D
		# no need to make it 1 x N x D --> N x 1 x D
		# time dimension이 1이기 때문에 자동으로 없어짐: 따라서, stack_and_transpose함수가 필요없음.

		# create the model object
		self.decoder_model = Model(
			inputs=[
				decoder_inputs_single,
				encoder_outputs_as_input,
				initial_s, 
				initial_c
			],
			outputs=[decoder_outputs, s, c]
		)




# class Seq2seqAtt():
#     """ 추후에 encoder ,decoder 구분해볼것.
#     """

#     def __init__(self, args):

#         self.args = args
#         self.model_nm_map = dict()

#         # encoder 와 decoder 라는 뼈대를 만듦.
#         self._build_encoder()
#         self._build_decoder()
#         self._build_t1_decoder()

#     def _build_encoder(self):	

#         # definition
#         self.encoder_inputs = Input(shape=(self.args['MAX_LEN_INPUT'],))
#         encoder_embed = Embedding(self.args['LEN_WORD2IDX_INPUTS'] + 1,
#                                        self.args['EMBEDDING_DIM'],
#                                        #weights=[embedding_matrix],
#                                        input_length=self.args['MAX_LEN_INPUT'],
#                                        #trainable=True
#                              )
#         encoder_bilstm = Bidirectional(
#                         LSTM(self.args['LATENT_DIM'],
#                              return_sequences=True,
#                              # dropout=0.5 # dropout not available on gpu
#                         ))

#         # pipeline
#         encoder_x = encoder_embed(self.encoder_inputs)
#         self.encoder_outputs = encoder_bilstm(encoder_x)

#     def _build_decoder(self):

#         # definition
#         self.decoder_inputs = Input(shape=(self.args['MAX_LEN_TARGET'],)) # teacher forcing input
#         self.decoder_embed = Embedding(self.args['LEN_WORD2IDX_OUTPUTS'] + 1, 
#                                        self.args['EMBEDDING_DIM']
#                              )

#         # pipeline
#         decoder_x = self.decoder_embed(self.decoder_inputs)


#         def _softmax_over_time(x):
#             # make sure we do softmax over the time axis
#             # expected shape is N x T x D
#             assert(K.ndim(x) > 2)
#             e = K.exp(x - K.max(x, axis=1, keepdims=True)) # axis=1에 주목.
#             s = K.sum(e, axis=1, keepdims=True)
#             return e / s

#         # ATTENTION
#         # Attention layers need to be global (전역 변수) because they will be repeated Ty times at the decoder
#         attn_repeat_layer = RepeatVector(self.args['MAX_LEN_INPUT'])
#         attn_concat_layer = Concatenate(axis=-1)
#         attn_dense1 = Dense(10, activation='tanh')
#         attn_dense2 = Dense(1, activation=_softmax_over_time)
#         attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]


#         # define the rest of the decoder (after attention)
#         self.decoder_lstm = LSTM(self.args['LATENT_DIM_DECODER'], 
#                             return_state=True
#                        )
#         self.decoder_dense = Dense(self.args['LEN_WORD2IDX_OUTPUTS'] + 1, 
#                               activation='softmax'
#                         )

#         self.initial_s = Input(shape=(self.args['LATENT_DIM_DECODER'],), name='s0')
#         self.initial_c = Input(shape=(self.args['LATENT_DIM_DECODER'],), name='c0')
#         self.context_last_word_concat_layer = Concatenate(axis=2) # for teacher forcing

#         # Unlike previous seq2seq, we cannot get the output all in one step
#         # Instead we need to do Ty steps And in each of those steps, we need to consider all Tx h's

#         # s, c will be re-assigned in each iteration of the loop
#         s = self.initial_s
#         c = self.initial_c

#         def _one_step_attention(h, st_1):
#             # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
#             # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)
 
#             # copy s(t-1) Tx times
#             # now shape = (Tx, LATENT_DIM_DECODER)
#             st_1 = attn_repeat_layer(st_1)

#             # Concatenate all h(t)'s with s(t-1)
#             # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
#             x = attn_concat_layer([h, st_1])

#             # Neural net first layer
#             x = attn_dense1(x)

#             # Neural net second layer with special softmax over time
#             alphas = attn_dense2(x)

#             # "Dot" the alphas and the h's
#             # Remember a.dot(b) = sum over a[t] * b[t]
#             context = attn_dot([alphas, h])

#             return context
#         self._one_step_attention = _one_step_attention


#         # collect outputs in a list at first
#         outputs = []
#         # 원래 LSTM은 내부적으로 아래와 같은 for문을 진행하지만, 여기서 우리는 Context를 계산하기 위해서 manual하게 for문을 구성함.
#         for t in range(self.args['MAX_LEN_TARGET']): # Ty times

#             ######################################################
#             ## `one_step_attention` function !
#             # get the context using attention
#             context = _one_step_attention(self.encoder_outputs, s)
  
#             # we need a different layer for each time step
#             selector = Lambda(lambda x: x[:, t:t+1]) # 해당 time 벡터만 추출. 우리는 layer-wise로 코딩해야 되기 때문에 lambda를 사용.
#             xt = selector(decoder_x)
            
#             # combine 
#             decoder_lstm_input = self.context_last_word_concat_layer([context, xt])

#             # pass the combined [context, last word] into the LSTM
#             # along with [s, c]
#             # get the new [s, c] and output
#             o, s, c = self.decoder_lstm(decoder_lstm_input, initial_state=[s, c])

#             # final dense layer to get next word prediction
#             decoder_outputs = self.decoder_dense(o)
#             outputs.append(decoder_outputs)

#         def _stack_and_transpose(x): # 다시 원래의 shape로 만들기 위해.
#             # 'outputs' is now a list of length Ty
#             # each element is of shape (batch size, output vocab size)
#             # therefore if we simply stack all the outputs into 1 tensor
#             # it would be of shape T x N x D
#             # we would like it to be of shape N x T x D
#             # x is a list of length T, each element is a batch_size x output_vocab_size tensor
#             x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
#             x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
#             return x

#         # pipeline
#         stacker = Lambda(_stack_and_transpose)
#         self.decoder_outputs = stacker(outputs)



#     def get_model(self, model_nm):
#         """
#             앞에서 정의된 블록들을 합쳐서
#             하나의 모델을 정의한다.
#         """
#         if model_nm == 'encoder2decoder':
#             # build model
#             self.model = Model(inputs=[
#                                     self.encoder_inputs,
#                                     self.decoder_inputs,
#                                     self.initial_s, 
#                                     self.initial_c,
#                                 ],
#                                 outputs=self.decoder_outputs)
#             # compile the model
#             self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#             return self.model
#         elif model_nm == 'encoder':
#             self.model = Model(self.encoder_inputs, 
#                                self.encoder_outputs)
#             return self.model
#         elif model_nm == 't1_decoder':
#             pass


