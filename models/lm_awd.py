import tensorflow as tf
from ..layers import EmbeddingsDropout, VA_LSTM, TightEmbeddings

tf.enable_eager_execution()

class LanguageModelAWD():
    
    def __init__(self, voc_size):   
        # Deep NN cells dimension and init
        self.embed_dim = 400
        self.lstm_h_units = 1150
        self.embedding_init_range = (-1,1)
        self.embed_drop = 0.4
        self.vocabulary_size = voc_size
        self.num_last_states = 150
        self.K = 5
        
        self.build()
        
    def build(self):
        self.embeddings = EmbeddingsDropout(self.vocabulary_size, self.embed_dim, p_do=self.embed_drop)
        self.lstm_layer1 = VA_LSTM(self.embed_dim, self.embed_dim)
        self.lstm_layer2 = VA_LSTM(self.lstm_h_units, self.embed_dim)
        self.lstm_layer3 = VA_LSTM(self.lstm_h_units, self.lstm_h_units)
        self.lstm_layer4 = VA_LSTM(self.embed_dim, self.lstm_h_units)

        self.tight_embed = TightEmbeddings(self.embeddings)
        
        self.layers = [self.embeddings, 
                       self.lstm_layer1, 
                       self.lstm_layer2, 
                       self.lstm_layer3, 
                       self.lstm_layer4,
                       self.tight_embed]

        self.ckp = tf.train.Checkpoint(**dict([(str(i),var) for i, var in enumerate(self.get_trainable_weights())]))
        
    def forward(self, batch, seq_len):
        embeds = self.embeddings.forward(batch)
        o_layer1, _ = self.lstm_layer1.forward(embeds, seq_len)
        o_layer2, _ = self.lstm_layer2.forward(o_layer1, seq_len)
        o_layer3, _ = self.lstm_layer3.forward(o_layer2, seq_len)
        outputs , o_layer4 = self.lstm_layer4.forward(o_layer3, seq_len)
        y_ = self.tight_embed.forward(o_layer4[1])
        return y_, outputs

    def forward_last(self, batch, seq_len):
        embeds = self.embeddings.forward(batch)
        o_layer1, _ = self.lstm_layer1.forward(embeds, seq_len)
        o_layer2, _ = self.lstm_layer2.forward(o_layer1, seq_len)
        o_layer3, _ = self.lstm_layer3.forward(o_layer2, seq_len)
        last_states , _ = self.lstm_layer4.forward(o_layer3, seq_len)
        return last_states[:, - self.num_last_states:, :]
        
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

    def save_model(self, ckpt="./checkpoint_awd/"):
        self.ckp.save(ckpt)

    def restore_model(self, ckpt="./checkpoint_awd/"):
        self.ckp.restore(tf.train.latest_checkpoint(ckpt))
