from collections import Counter

from .preprocess import simple_tokenizer, get_ids

w_data_path = "C:\\Users\\andre\\Documents\\Work\\lm_dataset\\wikitext-103\\"
w_train_path = "wiki.train.tokens"
w_valid_path = "wiki.valid.tokens"
w_test_path = "wiki.test.tokens"

def load_data(max_voc):
    with open(w_data_path + w_train_path, 'r', encoding='utf8') as f:
        w_data_train = f.read()
    
    with open(w_data_path + w_valid_path, 'r', encoding='utf8') as f:
        w_data_val = f.read()
        
    with open(w_data_path + w_test_path, 'r', encoding='utf8') as f:
        w_data_test = f.read()

    tr_tokens = simple_tokenizer(w_data_train)
    wiki_counter = Counter(tr_tokens)
    vocab = dict([(token,i) for i, (token, _) in enumerate(wiki_counter.most_common()[:100000])])

    val_tokens = simple_tokenizer(w_data_val)
    test_tokens = simple_tokenizer(w_data_test)

    tr_tokens = get_ids(tr_tokens, vocab)
    val_tokens =  get_ids(val_tokens, vocab)
    test_tokens =  get_ids(test_tokens, vocab)

    return tr_tokens, val_tokens, test_tokens, vocab