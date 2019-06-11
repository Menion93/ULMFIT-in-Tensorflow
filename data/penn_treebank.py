from .preprocess import get_tokens_vocab_from_corpus, get_ids

p_data_path = "C:\\Users\\andre\\Documents\\Work\\lm_dataset\\penn\\"
p_train_path = "ptb.train.txt"
p_valid_path = "ptb.valid.txt"
p_test_path = "ptb.test.txt"

def load_data(max_voc):
    with open(p_data_path + p_train_path, 'r', encoding='utf8') as f:
        p_data_train = f.read()
    
    with open(p_data_path + p_valid_path, 'r', encoding='utf8') as f:
        p_data_val = f.read()
        
    with open(p_data_path + p_test_path, 'r', encoding='utf8') as f:
        p_data_test = f.read()

    tr_tokens, vocab = get_tokens_vocab_from_corpus(p_data_train, max_voc)
    val_tokens, _ = get_tokens_vocab_from_corpus(p_data_val, max_voc)
    test_tokens, _ = get_tokens_vocab_from_corpus(p_data_test, max_voc)

    tr_tokens = get_ids(tr_tokens, vocab)
    val_tokens =  get_ids(val_tokens, vocab)
    test_tokens =  get_ids(test_tokens, vocab)

    return tr_tokens, val_tokens, test_tokens, vocab