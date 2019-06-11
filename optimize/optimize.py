import tensorflow as tf

def minimize(optimizer, model, loss, X, y, seq_len=None, lr=0.001, logging=False, it=0, log_every=10):
    optimizer.minimize(lambda: loss(model, X, y, seq_len, logging=logging, iteration=it, log_iterations=log_every), 
                                        var_list=model.get_trainable_weights())

def minimize_w_clipping(optimizer, model, loss, X, y, seq_len, lr=0.001, logging=False, it=0, log_every=10, clip_norm=0.25, vars=None):
    if vars:
        grad_and_var = optimizer.compute_gradients(lambda: loss(model, X, y, seq_len, logging=logging, iteration=it, log_iterations=log_every), 
                                            var_list=vars)
    else:
        grad_and_var = optimizer.compute_gradients(lambda: loss(model, X, y, seq_len, logging=logging, iteration=it, log_iterations=log_every), 
                                    var_list=model.get_trainable_weights())

    grad_and_var = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grad_and_var]

    optimizer.apply_gradients(grad_and_var)

