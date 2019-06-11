# Create the dataset
import string 
from sklearn.preprocessing import LabelBinarizer
import numpy as np



def get_dummy_dataset(num_instances=32, timesteps=10):
    
    start_token = 26
    end_token = 27
    input_size = num_instances

    def get_next_char(char):
        next_id = char2id[char] + 1
        next_id = 0 if next_id > 25 else next_id
        return id2char[next_id]

    def get_next_chars_from_nparray(array):
        return [get_next_char(char) for char in array]

    chars = list(string.ascii_lowercase)

    id2char = {start_token:"\t", end_token:"\n"}
    char2id = dict([(token,key) for key, token in id2char.items()])

    for i,char in enumerate(chars):
        id2char[i] = char
        char2id[char] = i

    enc_inputs = np.random.choice(chars, (input_size, timesteps))
    dec_inputs = np.array([get_next_chars_from_nparray(example) for example in enc_inputs])

    lb = LabelBinarizer()

    my_x = np.array([char2id[c] for arr in enc_inputs for c in arr]).reshape((input_size,timesteps))
    lb.fit([c for c in string.ascii_letters[0:26]])
    my_y = lb.transform(dec_inputs[:,-1])
    seq_len = [timesteps]*input_size
    return my_x, my_y, seq_len