import tensorflow as tf

from ..layers import DenseDropBN

class TransferModel:
    
    def __init__(self, input_dim, output_dim, training=True):
        self.hidden_dim = 1024
        self.dropout_rate = 0.2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training = training
        self.build()
        
    def build(self):
        self.dense1 = DenseDropBN((self.input_dim, self.hidden_dim), self.dropout_rate)
        self.dense2 = DenseDropBN((self.hidden_dim, self.output_dim), self.dropout_rate)
        
        self.layers = [self.dense1, self.dense2]
        
    def forward(self, input_):
        pooled = tf.layers.MaxPooling1D(2,2)(input_)
        flattened = tf.layers.Flatten()(pooled)
        h_output = self.dense1.forward(flattened, training=self.training)
        result = self.dense2.forward(h_output, training=self.training, activation=False)
        return result