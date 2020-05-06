import torch
import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # Work out how cleanly we can divide the dataset into bsz parts.
        num_batches = ids.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        ids = ids[:num_batches*batch_size]
        # Evenly divide the data across the bsz batches.
        return ids.view(batch_size, -1)

    def batchify(self, data, batch_size, args):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't fit (remainders, same as data[:num_batches*batch_size]). 
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches. (explanation to contiguous() https://stackoverflow.com/a/52229694/9004294)
        data = data.view(batch_size, -1).t().contiguous()

        if args.cuda:
            data = data.cuda()
        return data