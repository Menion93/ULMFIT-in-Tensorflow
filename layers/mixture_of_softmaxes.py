import tensorflow as tf

class MixtureOfSoftmaxes():
    
    def __init__(self, k, h_size, embeddings):
        self.k = k
        self.h_size = h_size
        self.embed_size = embeddings.shape[1]
        self.embeddings = embeddings
        self.build()
    
    def build(self):
        self.Whk = tf.Variable(tf.random_normal((self.k, self.h_size, self.embed_size)))
        self.Wpk = tf.Variable(tf.random_normal((self.h_size, self.k)))
        self._trainable_weights = [self.Whk, self.Wpk, self.embeddings]
    
    def compute_k_softmaxes(self, k_hct, embeddings):
        return tf.map_fn(lambda hct : tf.nn.softmax(tf.matmul(hct, 
                                                              tf.transpose(embeddings))),
                         k_hct)
    
    def forward(self, ht, embeddings):
        # Compute the pi weights
        pi_k = tf.nn.softmax(tf.matmul(ht, self.Wpk))

        # Make the size of the hidden outputs as (b_size, K, 1, hidden_dim)
        ht = tf.expand_dims(ht, axis=1)
        ht = tf.expand_dims(ht, axis=1)
        ht = tf.tile(ht, [1,self.k,1,1])
        
        # Compute MoS over a batch. This has shape (b_size, k, voc_dim)
        batch_of_sm = tf.squeeze(
                        tf.map_fn(
                            lambda ht_b: self.compute_k_softmaxes(tf.nn.tanh(tf.matmul(ht_b, self.Whk)),
                                                                 embeddings),
                            ht))
            
        # Prepare the prior to be broadcasted, shape is (b_size,k,1)
        # broadcasted to (b_size, k, voc_dim)
        # output after reduce is (b_size, voc_dim)
        pi_k = tf.expand_dims(pi_k, axis=-1)
        output = tf.reduce_sum(batch_of_sm * pi_k, axis=1)
                
        return output
        
    def get_trainable_weights(self):
        return self._trainable_weights
        