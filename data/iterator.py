import numpy as np
from math import floor
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from .fb_embeddings import EmbeddingsLoader

def old_get_next_batch(data, batch_size, bptt_val, vocab):
    i = 0
    ohe = OneHotEncoder(sparse=False, n_values=len(vocab))
    
    while i < len(data)-1-1:
        bptt = bptt_val if np.random.random() < 0.95 else bptt_val / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, bptt_val + 10) + 1
        
        if  i + seq_len*batch_size >= len(data) and len(data) - i >= seq_len:
            miss_toks = len(data) - i
            batch_size = floor(miss_toks / seq_len)
            
        if len(data) - i < seq_len:
            break
        
        upp = i + seq_len*batch_size 
        batch = np.array(data[i:upp]).reshape(batch_size, seq_len)
        i = upp
        yield batch[:,:-1],ohe.fit_transform(np.array(batch[:,-1])[:,None]), [seq_len]*batch_size

def get_bptt_batch_iterator(data, batch_size, bptt_val):
    i = 0
    
    while i < len(data)-1-1:
        bptt = bptt_val if np.random.random() < 0.95 else bptt_val / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, bptt_val + 10) + 1
        
        if  i + seq_len*batch_size >= len(data) and len(data) - i >= seq_len:
            miss_toks = len(data) - i
            batch_size = floor(miss_toks / seq_len)
            
        if len(data) - i < seq_len:
            break
        
        upp = i + seq_len*batch_size 
        batch = np.array(data[i:upp]).reshape(batch_size, seq_len)
        i = upp
        yield batch[:,:-1],np.array(batch[:,-1]), [seq_len]*batch_size

def get_batch_iterator(X, y, num_classes, batch_size, pad_value=0):
    data = np.hstack([X, y])
    np.random.shuffle(data)
    
    i = 0
    
    while i*batch_size < data.shape[0]:
        i += 1
        batch_x = data[(i-1)*batch_size:i*batch_size, 0:-num_classes]
        batch_y = data[(i-1)*batch_size:i*batch_size, -num_classes:]
        seq_len = [len(point[0]) for point in batch_x]
        batch_x = pad_sequences(batch_x.reshape((batch_y.shape[0])).tolist(), value=pad_value, padding='post')
        yield batch_x, np.asarray(batch_y, dtype=np.int), seq_len

def get_embedded_iterator(X, y, num_classes, batch_size, vocabulary, max_ts):
    embeddings = EmbeddingsLoader()
    embeddings.load_embeddings(vocabulary)

    data = np.hstack([X, y])
    np.random.shuffle(data)
    
    i = 0
    
    while i*batch_size < data.shape[0]:
        i += 1
        batch_x = data[(i-1)*batch_size:i*batch_size, 0:-num_classes]
        batch_y = data[(i-1)*batch_size:i*batch_size, -num_classes:]
        padded_x_batch = []

        for point in batch_x:
            embedded = [embeddings.embed(token) for token in point[0]]
            pad_len = max_ts - len(embedded)
            embedded += [embeddings.embed('</s>') for _ in range(pad_len)]
            embedded = embedded[:max_ts]
            padded_x_batch.append(embedded)

        padded_x_batch = np.array(padded_x_batch).astype(np.float32)

        yield padded_x_batch, np.asarray(batch_y, dtype=np.int)