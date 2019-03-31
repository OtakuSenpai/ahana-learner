#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import socket, os, time, base64
import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
import os
import time

# duplicate and shift each sequence to form the input and target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# build a training model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

# download the dictionary
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# open the dictionary as a file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# find the unique characters in the text
vocab = sorted(set(text))

# find the number of character occurences
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# save them as an array
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# use batch method to convert the individual characters to the sequences of desired size 
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

# Print the first example's input and target values
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

print('\n')

# For the input at time step 0, the model receives the index for "F" and 
# trys to predict the index for "i" as the next character.

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

print('\n')

# Batch size
BATCH_SIZE = 64

steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 128

# Number of RNN units
rnn_units = 256

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')

model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

# Run the model to see that it behaves as expected.
for input_example_batch, target_example_batch in dataset.take(1): 
  example_batch_predictions = model(input_example_batch)

# Now train the model

# This function attaches a optimiser and a loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())

# Configure the training procedure using the tf.keras.Model.compile method. 
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=15

# Since we have already done the training
#history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

# generate texts
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
      predictions = model.predict_on_batch(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions,0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

null = u"\u0000"
channel = "#bot-test"

def str_2_list(string,strsplit):
    li = list(string.split(strsplit))
    return li 

def check(word, list):
    retValue = False
    if word in list:
       retValue = True
    else:
       retValue = False
    return retValue  

class PyBot: 
    connect = False 
    nick = ""
    user = ""
    server = ""
    password = ""
    port = 6667
    sock = socket.socket()
    
    def __init__(self,Nick,User,Server,Password):
        self.nick = Nick
        self.user = User
        self.server = Server
        self.password = Password 
         
    def Connect(self):
        self.sock.connect((self.server,self.port))
        self.sock.send(b"CAP REQ :sasl\r\n")
        time.sleep(0.5)
        self.sock.send(("NICK " + self.nick + " \r\n").encode())
        self.sock.send(("USER " + self.nick + " hostname servername :" + self.user + " \r\n").encode())
        self.sock.send(b"AUTHENTICATE PLAIN\n")
        authMessage = self.nick + null + self.nick + null + self.password  
        authMessage = base64.b64encode(authMessage.encode())
        msgs = []
        
        while True:
            msg = self.sock.recv(2048).decode()
            print(msg)
            strlist = str_2_list(msg.strip("\r\n")," ") 
            msgs = msg.split(' ')
            if msg.find("AUTHENTICATE +") != -1:
                self.sock.send(("AUTHENTICATE " + authMessage.decode() + "\n").encode())
            if msg.find(":SASL authentication successful") != -1:
                self.sock.send(b"CAP END\n")
                print("SASL authentication successful\n")
                self.sock.send(("JOIN " + channel + "\r\n").encode()) 
                self.connect = True
 
            if msg.find("PING :") != -1:
                self.sock.send(b"PONG :pingis\n")   
               
            if self.connect and (":" + self.nick) in strlist and (len(strlist) - strlist.index(":" + self.nick)) > 0: 
                print("Str len ", len(strlist))
                print("strlist.index(self.nick) ",strlist.index(":" + self.nick))
                print("Connect ", self.connect)
                print("Inside wordlist")
                word = strlist[strlist.index(":" + self.nick) + 1]   
                print("Word is ", word)
                print("Strlist is ", strlist)
                gettext = generate_text(model,word)
                wordlist = str_2_list(gettext,"\n")
                print("\nGenerated text is ", wordlist[0]) 
                self.sock.send(("PRIVMSG " + channel + " :" + wordlist[5] + "\r\n").encode())       
            else:
                continue           
                     
                      
                     
mybot = PyBot("BamBaka","AghoraBot","chat.freenode.net","ffff")
mybot.Connect()




