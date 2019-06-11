import tensorflow as tf

class TightEmbeddings():
    
    def __init__(self, embed_do_layer, name='t_embeddings'):
        self.name = name
        self.embed_do_layer = embed_do_layer
        self._trainable_weights = self.embed_do_layer.embeddings
        self.trainable = True
        
    def forward(self, input_):
        return tf.matmul(input_, tf.transpose(self._trainable_weights))
    
    def get_trainable_weights(self):
        return [self._trainable_weights] if self.trainable else []