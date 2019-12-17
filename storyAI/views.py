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
		    tf.keras.layers.Dense(vocab_size)
		  	])
		    return model

		def generate_text(model, start_string, num_generate = 200):
		    
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

		text = request.POST.get('text', None)
		tf.compat.v1.enable_eager_execution()
		vocab_size = 71
		embedding_dim = 256
		rnn_units = 1024
		
		vocab = ['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 
		         ';', '>', '?', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
		         'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '«', '»', 'ç', 'é', 'ó', 'ą', 'ć', 'ę', 'ł', 'ń', 'ś', 'ź', 'ż', 
		         '–', '—', '”', '„', '…']
		char2idx = {u:i for i, u in enumerate(vocab)}
		idx2char = np.array(vocab)
		sciezka = 'static/ai_models/ckpt_10'
		#text = "taki był człowiek pewnego razu, który szedł"
		model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
		model.load_weights(sciezka)
		data = generate_text(model, text[-100:], 200)

		return HttpResponse(data, content_type='application/json')

