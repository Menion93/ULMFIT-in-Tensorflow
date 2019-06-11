import tensorflow as tf

class ScaledDotProductAttention:
    
    def __init__(self, dim):
        self.scale_factor = tf.sqrt(float(dim))

    def forward(self, Q, K, V):
        K = tf.transpose(K, perm=[0,2,1])
        softmax_scaled_weights = tf.nn.softmax(tf.matmul(Q,K) / self.scale_factor)
        return tf.matmul(softmax_scaled_weights, V)
    
class MultiHeadAttention:
    
    def __init__(self, heads, hidden_inp, trainable=True):
        assert(hidden_inp % heads == 0)
        
        self.trainable = trainable
        self.heads = heads
        self.h_in = hidden_inp
        self.dk = hidden_inp // heads
        self.t_shape = (self.h_in, self.dk * self.heads)
        self.scaled_dpa = ScaledDotProductAttention(self.dk)
        self.build()
    
    def build(self):
        self.Wq = tf.Variable(tf.random_normal(self.t_shape))
        self.Wk = tf.Variable(tf.random_normal(self.t_shape))
        self.Wv = tf.Variable(tf.random_normal(self.t_shape))
        self.Wo = tf.Variable(tf.random_normal((self.h_in, self.dk * self.heads)))
        self._trainable_weights = [self.Wq, self.Wk, self.Wv, self.Wo]
        
    #@tf.contrib.eager.defun
    def forward(self, Q, K, V):
        
        # input dims [batch, ts, dk*heads]
        q = tf.tensordot(Q, self.Wq, axes=[[-1], [0]])
        k = tf.tensordot(K, self.Wk, axes=[[-1], [0]])
        v = tf.tensordot(V, self.Wv, axes=[[-1], [0]])
        
        def reshape1(x):
            s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], self.heads, self.dk])
            x = tf.transpose(x, [2, 0, 1, 3])  
            x = tf.reshape(x, [-1, s[1], self.dk])  # [n_head * batch_size, len_q, dk]
            return x
        
        # Reshape to do the scaled dot product attention to [batch*heads, ts, dk]
        q = reshape1(q)
        k = reshape1(k)
        v = reshape1(v)
        
        dp_att_out = self.scaled_dpa.forward(q,k,v)
        
        # Reshape back to [batch, ts, dk*heads]
        def reshape2(x):
            s = tf.shape(x)   # [n_head * batch_size, len_v, d_k]
            x = tf.reshape(x, [self.heads, -1, s[1], s[2]]) 
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], self.heads*self.dk])  # [batch_size, len_v, n_head * d_k]
            return x
        
        dp_att_out = reshape2(dp_att_out)
        
        return tf.tensordot(dp_att_out, self.Wo, axes=[[-1], [0]])

    
    def get_trainable_weights(self):
        return self._trainable_weights if self.trainable else []