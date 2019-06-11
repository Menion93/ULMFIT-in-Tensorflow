import tensorflow as tf

class Dense:
    
    def __init__(self, input_dim, output_dim, bias=True, dropout=0.5, trainable=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dropout = dropout
        self.trainable = trainable
        self.build()
        
    def build(self):
        self.W = tf.Variable(tf.random_normal((self.input_dim, self.output_dim)))
        if self.bias:
            self.b  = tf.Variable(tf.random_uniform((self.output_dim,)))
            self._trainable_weights = [self.W, self.b]
        else:
            self._trainable_weights = [self.W]
   
    #@tf.contrib.eager.defun
    def forward(self, input_, training=False):

        if training:
            W = tf.nn.dropout(self.W, self.dropout)
        else:
            W = self.W
            
        if self.bias:
            return tf.tensordot(input_, W, axes=[[-1], [0]]) + self.b
        else:
            return tf.tensordot(input_, W, axes=[[-1], [0]])
        
    def get_trainable_weights(self):
        return self._trainable_weights if self.trainable else []