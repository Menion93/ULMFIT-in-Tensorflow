from pymongo import MongoClient

class EmbeddingsLoader:

    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client.embeddings
        self.collections = {
            'en': self.db.en,
            'it': self.db.it
        }

    def load_embeddings(self, vocabulary):
        self.embeddings = self.get_f_embedd(vocabulary)

    def embed(self, token):
        return self.embeddings['en'].get(token, 
                                         self.embeddings['it'].get(token,  
                                                                   self.embeddings['en'].get('</s>')))

    def clear(self):
        if self.embeddings:
            del self.embeddings

    def find_word(self, w):
        entry = self.collections['en'].find_one({"word": w})
        if entry: 
            return entry
        else: 
            entry = self.collections['it'].find_one({"word": w})
            if entry: 
                return entry
            else:
                return self.collections['en'].find_one({"word": '</s>'})

    def get_embedding_from_db_result(self, db_result):
        return list(db_result['vector'].values())

    def find_multiple(self, words, collection):
        # building list
        word_list = []
        for w in words:
            word_list.append({"word": w})
        
        return collection.find({"$or": word_list})

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    def get_f_embedd(self, vocabulary):
        vocabulary = list(vocabulary.keys())
        vocabulary.append('</s>')
        vocabulary_chunks = list(self.chunks(vocabulary, 100))

        we = {
            'en': {},
            'it': {}
        }

        total = len(vocabulary_chunks)
        count = 0
        for chunk in vocabulary_chunks:
            founds = self.find_multiple(chunk, self.collections['en'])
            for found in founds:
                we['en'][found['word']] = list(found['vector'].values())
            count += 1
            print(str(count) + "/" +str(total), end="\r")
            
        total = len(vocabulary_chunks)
        count = 0
        for chunk in vocabulary_chunks:
            founds = self.find_multiple(chunk, self.collections['it'])
            for found in founds:
                we['it'][found['word']] = list(found['vector'].values())
            count += 1
            print(str(count) + "/" +str(total), end="\r")
        return we