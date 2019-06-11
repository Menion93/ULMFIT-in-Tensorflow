from collections import Counter
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

def get_tokens_vocab_from_corpus(corpus, max_voc_size):
    
    tokens = tokenizer.tokenize(corpus)
    counter = Counter(tokens)
    vocab = dict([(token,i) for i, (token, _) in enumerate(counter.most_common()[:max_voc_size])])
    return tokens, vocab

def get_ids(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def simple_tokenizer(corpus):
    tokens=corpus.split(" ")
    tokens = [token for token in tokens if token != "" and token != " " and token != "\n"]
    return tokens