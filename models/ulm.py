import tensorflow as tf
from math import ceil
import numpy as np

from ..data.iterator import get_batch_iterator, get_bptt_batch_iterator
from ..logging import log_scalar

class UniversalLMClassifier:
    
    def __init__(self, language_model, transfer_model):
        self.lm = language_model
        self.tf_model = transfer_model
        self.build()
        
    def build(self):
        self.layers = self.lm.layers[:-1] + self.tf_model.layers
        self.ckp = tf.train.Checkpoint(**dict([(str(i),var) for i, var in enumerate(self.get_trainable_weights())]))

    def forward(self, X, seq_len):
        output = self.lm.forward_last(X, seq_len)
        return self.tf_model.forward(output)
    
    def slanted_t_lr(self, t, T, cut_frac, ratio, n_max):
        p = 0
        cut = T * cut_frac
        if t < cut:
            p = t/cut
        else:
            p = 1 - (t - cut) / (cut * (1 / (cut_frac - 1))) 
        lr = n_max * (1 + p*(ratio - 1)) / ratio
        return lr
    
    def discr_finetuning(self, gradients, layers):
        var_count = len(gradients)
        layer_index = 0
        for level_layer, layer in enumerate(layers[::-1]):
            trainable_weights = layer.get_trainable_weights()
            for _ in trainable_weights:
                if level_layer > 0:
                    gradients[var_count - 1 - layer_index] /= (level_layer*2.6)
                layer_index += 1
    
    def finetune_lm(self,
                    train_tokens, 
                    val_tokens, 
                    epochs, 
                    loss,
                    tensorboard=False,
                    log_dir="./finetune_log/",
                    ckpt_dir="./finetune_ckpts",
                    batch_size=32, 
                    val_bs=32,
                    cut_frac=0.1, 
                    ratio=32, 
                    n_max=0.01, 
                    bptt=10):
        
        if tensorboard:
            summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
            summary_writer.set_as_default()
            global_step = tf.train.get_or_create_global_step()
        
        iteration = 0
        T_estimate = epochs * int(len(train_tokens) / (bptt*1.5*batch_size))
        current_val_score = self.compute_token_score(val_tokens, val_bs, bptt)
        
        for _ in range(epochs):
            for x_train, y_train, seq_len in get_bptt_batch_iterator(train_tokens, batch_size, bptt):
                
                if tensorboard:
                    global_step.assign_add(1)

                with tf.GradientTape() as tape:
                    loss_ = loss(self.lm, 
                                x_train, 
                                y_train, 
                                seq_len, 
                                logging=tensorboard,
                                iteration=iteration)
                # Get trainable weights
                trainable_weights = self.lm.get_trainable_weights()
                gradients = tape.gradient(loss_, trainable_weights)
                # Apply Slanted Triangular learning rate
                gradients = [grad * self.slanted_t_lr(iteration, T_estimate, cut_frac, ratio, n_max)
                             for grad in gradients]
                # Apply discriminative finetuning
                self.discr_finetuning(gradients, self.lm.layers)
                # Apply gradient clipping
                gradients = [tf.clip_by_norm(grad, clip_norm=0.25) for grad in gradients]
                # Update weights
                self.lm.backward(trainable_weights, gradients)
            
            val_score = self.compute_token_score(val_tokens, val_bs, bptt)

            if tensorboard:
                log_scalar('val_perplex', val_score)

            print("Validation score is {0}".format(val_score))
            
            if val_score < current_val_score:
                self.lm.save_model(ckpt=ckpt_dir)

    
    def train(self,
              x_train, 
              y_train, 
              x_val, 
              y_val, 
              loss, 
              epochs,
              score_fun,
              tensorboard=False,
              log_dir="./ulm_log/",
              ckpt_dir="./ulm_ckpt/",
              pad_value=0,
              batch_size=32, 
              val_bs=32,
              cut_frac=0.1, 
              ratio=32, 
              n_max=0.01):
        
        if tensorboard:
            summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
            summary_writer.set_as_default()
            global_step = tf.train.get_or_create_global_step()
            
        iteration = 0
        n_classes = y_train.shape[1]
        T = epochs * ceil(x_train.shape[0] / batch_size)
        current_val_score = self.compute_score(x_val, y_val, n_classes, val_bs, score_fun, pad_value)
        for epoch in range(epochs):
            for x, y, seq_len in get_batch_iterator(x_train, y_train, n_classes, batch_size, pad_value=pad_value):
                
                if tensorboard:
                    global_step.assign_add(1)

                with tf.GradientTape() as tape:
                    loss_ = loss(self, 
                                x, 
                                y, 
                                seq_len, 
                                logging=tensorboard,
                                iteration=iteration)
                # Get unfrozen weights
                trainable_weights = [weight for layer in self.layers[-epoch-1:] for weight in layer.get_trainable_weights()]
                gradients = tape.gradient(loss_, trainable_weights)
                # Apply Slanted Triangular learning rate
                gradients = [grad * self.slanted_t_lr(iteration, T, cut_frac, ratio, n_max)
                             for grad in gradients]
                # Apply discriminative finetuning on the unfrozer layers
                self.discr_finetuning(gradients, self.layers[-epoch-1:])
                # Apply gradient clipping
                gradients = [tf.clip_by_norm(grad, clip_norm=0.25) for grad in gradients]
                # Update weights
                self.backward(trainable_weights, gradients)
            
            val_score = self.compute_score(x_val, y_val, n_classes, val_bs, score_fun, pad_value)

            if tensorboard:
                log_scalar('val_score', val_score)

            print("Validation score is {0}".format(val_score))
            
            if val_score > current_val_score:
                self.save_model(ckpt=ckpt_dir)

    
    def backward(self, weights, gradients):
        for weight, grad in zip(weights, gradients):
            weight.assign_sub(grad)
            
    def get_trainable_weights(self):
        return [weight for layer in self.layers for weight in layer.get_trainable_weights()]
    
    def save_model(self, ckpt="./ulm_ckpt/"):
        self.ckp.save(ckpt)

    def restore_model(self, ckpt="./ulm_ckpt/"):
        self.ckp.restore(tf.train.latest_checkpoint(ckpt))

    def batch_score(self, model, x, y, seq_len):
        y_, _ =  model.forward(x, seq_len)
        return tf.pow(2,tf.losses.sparse_softmax_cross_entropy(y, logits=y_))

    def compute_token_score(self, tokens, bs, bptt):
        scores = []
        for x, y, seq_len in get_bptt_batch_iterator(tokens, bs, bptt):
            scores.append(self.batch_score(self.lm, x, y, seq_len))
        return np.mean(scores)

    def compute_score(self, x_val, y_true, n_classes, bs, score_fun, pad_value):
        scores = []
        
        for x, y, seq_len in get_batch_iterator(x_val, y_true, n_classes, bs, pad_value=pad_value):
            scores.append(score_fun(self.forward(x, seq_len), y))
        return np.mean(scores)