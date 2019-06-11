import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import DropoutWrapper, LSTMCell 
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.rnn import LSTMBlockCell

# This class is not used with dropconnect, only with variational autoencoder (We avoid to reimplement a lstm)
# This class is not used with dropconnect, only with variational autoencoder (We avoid to reimplement a lstm)
class VA_LSTM():
    
    def __init__(self, units, inputSize, peephole=False, name="var_lstm"):
        self.name = name
        self.units = units
        self.peephole = peephole
        self.inputSize = inputSize
        self.minval = -1/ np.sqrt(self.units)
        self.maxval = -self.minval
        self.trainable = True
        self.build()

        
    def build(self):
        self.lstm_cell = LSTMBlockCell(self.units, 
                                  #use_peepholes=self.peephole, 
                                  use_peephole = True)
                                  #initializer=tf.initializers.random_uniform(minval=self.minval,
                                  #                                           maxval=self.maxval))

        self.va_lstm_cell = DropoutWrapper(self.lstm_cell, 
                                          variational_recurrent=True, input_keep_prob=0.7,
                                          output_keep_prob=0.7, state_keep_prob=0.7, dtype=tf.float32,
                                          input_size=self.inputSize)
    
        tf.nn.dynamic_rnn(self.va_lstm_cell, 
                          tf.random_normal((1,1,self.inputSize)), 
                          dtype=tf.float32)
        
        self._trainable_weights = self.lstm_cell.variables
                                      
    def forward(self, input_, seq_lens):                      
        outpus, state =  tf.nn.dynamic_rnn(self.va_lstm_cell, 
                                            input_, sequence_length = seq_lens, 
                                            dtype=tf.float32)
                                            
        #self._trainable_weights = self.lstm_cell.variables


        return outpus, state
        
    def get_trainable_weights(self):
        return self._trainable_weights if self.trainable else []