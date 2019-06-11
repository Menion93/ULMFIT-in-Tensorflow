import tensorflow as tf
from ..layers import EmbeddingsDropout, VA_LSTM, MixtureOfSoftmaxes

tf.enable_eager_execution()
class LanguageModelMoS():
    
    def __init__(self, voc_size, n_experts):        
        # Deep NN cells dimension and init
        self.embed_dim = 400
        self.lstm_h_units = 1150
        self.embedding_init_range = (-1,1)
        self.embed_drop = 0.4
        self.vocabulary_size = voc_size
        self.k = n_experts
        
        self.build()
        
    def build(self):
        self.embeddings = EmbeddingsDropout(self.vocabulary_size, self.embed_dim, p_do=self.embed_drop)
        self.lstm_layer1 = VA_LSTM(self.embed_dim, self.embed_dim)
        self.lstm_layer2 = VA_LSTM(self.lstm_h_units, self.embed_dim)
        self.lstm_layer3 = VA_LSTM(self.lstm_h_units, self.lstm_h_units)
        self.lstm_layer4 = VA_LSTM(self.embed_dim, self.lstm_h_units)
        self.mos = MixtureOfSoftmaxes(self.k, self.embed_dim, self.embeddings.embeddings)
        
        self.layers = [self.embeddings, 
                       self.lstm_layer1, 
                       self.lstm_layer2, 
                       self.lstm_layer3, 
                       self.lstm_layer4,
                       self.mos]

        self.ckp = tf.train.Checkpoint(**dict([(str(i),var) for i, var in enumerate(self.get_trainable_weights())]))

        
    def forward(self, batch, seq_len):
        embeds = self.embeddings.forward(batch)
        o_layer1, _ = self.lstm_layer1.forward(embeds, seq_len)
        o_layer2, _ = self.lstm_layer2.forward(o_layer1, seq_len)
        o_layer3, _ = self.lstm_layer3.forward(o_layer2, seq_len)
        outputs , o_layer4 = self.lstm_layer4.forward(o_layer3, seq_len)
        y_ = self.mos.forward(o_layer4[1], self.embeddings.embeddings)
        return y_, outputs
        
    def get_trainable_weights(self):
        return [weight 
                for layer in self.layers 
                for weight in layer.get_trainable_weights()]
    
    def backward(self, variables, gradients):
        self.embeddings.apply_gradients(self.tight_policy(gradients[0], gradients[-1]))
        for var, grad in list(zip(variables, gradients))[1:-1]:
            var.assign_sub(grad)
            
    def tight_policy(self, g1, g2):
        return g1 + g2             

    def save_model(self, ckpt="./checkpoint_mos/"):
        self.ckp.save(ckpt)

    def restore_model(self, ckpt="./checkpoint_mos/"):
        self.ckp.restore(tf.train.latest_checkpoint(ckpt))
