import tensorflow as tf

class DenseDropBN:
    
    def __init__(self, shape, dropout):
        self.trainable = True
        self.shape = shape
        self.dropout_rate = dropout
        self.build()
        
    def build(self):
        self.dense = tf.Variable(tf.random_normal((self.shape)))
        self._trainable_weights = self.dense
        
    def forward(self, input_, training=True, activation=True):
        dropout_dense = tf.layers.dropout(self.dense, self.dropout_rate)
        batch_norm_activ = tf.layers.batch_normalization(tf.matmul(input_, dropout_dense), 
                                                         training=training)
        if activation:
            return tf.nn.relu(batch_norm_activ)
            
        return batch_norm_activ
    
    def get_trainable_weights(self):
        return [self._trainable_weights] if self.trainable else []