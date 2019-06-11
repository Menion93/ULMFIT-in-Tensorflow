import tensorflow as tf
from tensorflow.losses import softmax_cross_entropy, sparse_softmax_cross_entropy, sigmoid_cross_entropy
from ..logging import log_scalar

def AR(alpha, ht):
    return alpha*tf.reduce_sum(
                    tf.map_fn(
                        tf.nn.l2_loss, tf.reduce_sum(ht, axis=2)))

def TAR(beta, ht):
    return beta*AR(1, ht[1:]-ht[:-1])

def perplexity(entropy):
    return tf.pow(2,entropy)
        
def lm_loss(model, X, y, seq_len, alpha=2, beta=1):
    y_, last_lstm_states = model.forward(X, seq_len)
    softmax_ce = softmax_cross_entropy(y, logits=y_)
    
    correct_mask = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
    
    loss =  softmax_ce +\
            AR(alpha, last_lstm_states) +  \
            TAR(beta, last_lstm_states)
    print('-'*70)
    print("Loss: {0:5f} | Perplexity: {1:5f} | Accuracy: {2:5f}".format(loss, perplexity(softmax_ce), accuracy))
    print('-'*70)
    return loss

def lm_loss_sparse(model, X, y, seq_len, logging=False, iteration=0, alpha=2, beta=1, log_iterations=10):
    y_, last_lstm_states = model.forward(X, seq_len)
    softmax_ce = sparse_softmax_cross_entropy(labels=y, logits=y_)
    
    loss =  softmax_ce +\
            AR(alpha, last_lstm_states) +  \
            TAR(beta, last_lstm_states)

    perplexity_ = perplexity(softmax_ce)
    
    if logging and iteration % log_iterations == 0:
        log_scalar("train_perplexity", perplexity_)
        log_scalar("train_loss", loss)

    print('-'*70)
    print("Loss: {0:5f} | Perplexity: {1:5f}".format(loss, perplexity_))
    print('-'*70)
    return loss

def cross_entropy_w_softmax(model, X, y, seq_len=None, logging=False, iteration=0, log_iterations=10):
    y_ =  model.forward(X, seq_len) if seq_len else model.forward(X)

    loss = softmax_cross_entropy(y, logits=y_)

    correct_mask = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    if logging and iteration % log_iterations == 0:
        log_scalar("train_loss", loss)
        log_scalar("accuracy", accuracy)

    print('-'*50)
    print("Loss: {0:5f} | Accuracy: {1:5f}".format(loss, accuracy))
    print('-'*50)
    return loss

def m_sigmoid_cross_entropy(model, X, y, seq_len=None, logging=False, iteration=0, log_iterations=10):
    y_ =  model.forward(X, seq_len) if seq_len else model.forward(X)

    loss = sigmoid_cross_entropy(y, y_)

    correct_mask = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    if logging and iteration % log_iterations == 0:
        log_scalar("train_loss", loss)
        log_scalar("accuracy", accuracy)

    print('-'*50)
    print("Loss: {0:5f} | Accuracy: {1:5f}".format(loss, accuracy))
    print('-'*50)
    return loss


