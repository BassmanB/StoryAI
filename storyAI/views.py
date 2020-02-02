from __future__ import absolute_import, division, print_function, unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
import os, shutil
import tensorflow as tf
import keras
from keras.models import load_model
import keras.models
import os, numpy, sys, time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def home(request):
	return render(request, 'home.html')

def generate(request):
	return render(request, 'generate.html')

class AiModel(View):

	def author_answer(request):

		def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
		    model = tf.keras.Sequential([
		    tf.keras.layers.Embedding(vocab_size, embedding_dim,
		                              batch_input_shape=[batch_size, None]),
		    tf.keras.layers.GRU(rnn_units,
		                        return_sequences=True,
		                        stateful=True,
		                        recurrent_initializer='glorot_uniform'),
		    tf.keras.layers.Dense(vocab_size),
		    tf.keras.layers.Dense(vocab_size)
		  	])
		    return model

		def build_model1(vocab_size, embedding_dim, rnn_units, batch_size):
		    model = tf.keras.Sequential([
		    tf.keras.layers.Embedding(vocab_size, embedding_dim,
		                              batch_input_shape=[batch_size, None]),
		    tf.keras.layers.GRU(rnn_units,
		                        return_sequences=True,
		                        stateful=True,
		                        recurrent_initializer='glorot_uniform'),
		    tf.keras.layers.Dense(vocab_size)])
		    return model
		    
		def generate_text(model, start_string, num_generate, char2idx, idx2char):
		    
		    input_eval = [char2idx[s] for s in start_string]
		    input_eval = tf.expand_dims(input_eval, 0)

		    text_generated = []
		    temperature = 1.0


		    model.reset_states()
		    for i in range(num_generate):
		        predictions = model(input_eval)
		       
		        predictions = tf.squeeze(predictions, 0)
		        predictions = predictions / temperature
		        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

		        input_eval = tf.expand_dims([predicted_id], 0)

		        text_generated.append(idx2char[predicted_id])

		    return (start_string + '\n' + ''.join(text_generated))
		    """
		def generate_text(model, start_string):
		    # Evaluation step (generating text using the learned model)

		    # Number of characters to generate
		    num_generate = 1000

		    # Converting our start string to numbers (vectorizing)
		    input_eval = [char2idx[s] for s in start_string]
		    input_eval = tf.expand_dims(input_eval, 0)

		  # Empty string to store our results
		    text_generated = []

		  # Low temperatures results in more predictable text.
		  # Higher temperatures results in more surprising text.
		  # Experiment to find the best setting.
		    temperature = 1.0

		  # Here batch size == 1
		    model.reset_states()
		    for i in range(num_generate):
		        predictions = model(input_eval)
		        # remove the batch dimension
		        predictions = tf.squeeze(predictions, 0)

		        # using a categorical distribution to predict the word returned by the model
		        predictions = predictions / temperature
		        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

		        # We pass the predicted word as the next input to the model
		        # along with the previous hidden state
		        input_eval = tf.expand_dims([predicted_id], 0)

		        text_generated.append(idx2char[predicted_id])

		    return (start_string + ''.join(text_generated))
		    """

		text = request.POST.get('text', None)
		writer = request.POST.get('writer', None)
		tf.compat.v1.enable_eager_execution()

		if(writer=='lem' or writer == 'adam'):
			sciezka = 'static/checkpoints/lem1/ckpt_29'
		else:
			sciezka = 'static/checkpoints/anton/ckpt_10'	
			vocab_size = 71
			embedding_dim = 256
			rnn_units = 1024
			model = build_model1(vocab_size, embedding_dim, rnn_units, batch_size=1)

			vocab = ['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', 
				 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 
		         ';', '>', '?', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f',
		         'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
		         'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '«', '»', 'ç', 'é', 
		         'ó', 'ą', 'ć', 'ę', 'ł', 'ń', 'ś', 'ź', 'ż', 
		         '–', '—', '”', '„', '…']

		if(writer=='lem'):
		
			vocab = ['\n', ' ', '!', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '\xa0', 'ó', 'ą', 'ć', 'ę', 'ł', 'ń', 'ś', 'ź', 'ż', '—', '’', '”', '„', '…', '\ufeff']

			vocab_size = 68
			embedding_dim = 256
			rnn_units = 1048
			model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

		if(writer == 'adam'):
			vocab = ['\n', ' ', '!', '"', '%', "'", '(', ')', '*', ',', '-', '.',
			 ':', ';', '?', '[', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
			  'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 
			  'z', '}', '\x80', '\x81', '\x82', '\x84', '\x85', '\x86', '\x87', '\x8c', 
			  '\x8f', '\x93', '\x94', '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9f', '\xa0', 
			'â', 'ä', 'é', 'ó', 'ă', 'ć', 'ď', 'ę', 'ĺ', 'ł', 'ń', 'ś', 
			'ş', 'š', 'ť', 'ź', 'ż']
			sciezka = 'static/checkpoints/adam/ckpt_30'
			vocab_size = len(vocab)
			embedding_dim = 256
			rnn_units = 1048
			model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

	

		
		
		char2idx = {u:i for i, u in enumerate(vocab)}
		idx2char = np.array(vocab)
		
		#text = "taki był człowiek pewnego razu, który szedł i potem zrobił"
		
		model.load_weights(sciezka)
		#text = u"przygodę swoją rozpocząłęm od wyruszenia z rynku z tobołkiem, oraz pamiątkowym zdjęciem brata w kieszeni"
		data = generate_text(model, text[-100:], 600, char2idx, idx2char)
		print(data)

		return HttpResponse(data, content_type='application/json')

