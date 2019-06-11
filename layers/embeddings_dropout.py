import tensorflow as tf

class EmbeddingsDropout():
    
    def __init__(self, vocab_size, embed_size, p_do = 0.5, name='embeddings'):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.p_do = p_do
        self.build()
        self.trainable = True
        
    def build(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                                 name='embedding')
        self._trainable_weights = self.embeddings

    def forward(self, input_):
        keep_prob = 1 - self.p_do
        self.do_embeddings = tf.nn.dropout(self.embeddings, 
                                           keep_prob=keep_prob, 
                                           noise_shape=(self.vocab_size, 1))
        

        return tf.nn.embedding_lookup(self.do_embeddings, input_)
    
    def get_trainable_weights(self):
        return [self._trainable_weights] if self.trainable else []
    
    def apply_gradients(self, gradient):
        self.embeddings.assign_sub(gradient)