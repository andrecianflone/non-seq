import os
import json
import random
from io import open
import torch
import nltk
from nltk import word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class NaivePsychCorpus():
    def __init__(self, path, test_percent):
        self.dictionary = Dictionary()
        train, valid, test = self.read_data(path, test_percent)
        self.train, self.valid, self.test =\
                  self.tokenize(train), self.tokenize(valid), self.tokenize(test)

    def read_data(self, path, test_percent):
        with open(path, 'r') as f:
            data = json.load(f)

        # train, valid_test = self.split_dict(data, test_percent*2)
        # valid, test = self.split_dict(valid_test, 0.5)
        train, valid, test = self.split_dict_label(data)
        return train, valid, test

        # test_valid_size = int((len(keys)*args.test_percent)*2)
        # keys[-int((len(keys)*args.test_percent)*2):]
        # test_idx = test_valid[len(test_valid)//2:]
        # valid_idx = test_valid[:len(test_valid)//2]
        # train_idx = keys[:-int((len(keys)*args.test_percent)*2)]

    def split_dict_label(self, d, shuffle=False):
        """
        Looks for "train/dev/test" label and splits accordingly
        """
        train = {}
        valid = {}
        test = {}
        for idkey, story in d.items():
            if story["partition"] == 'train':
                train[idkey] = story
            elif story["partition"] == 'dev':
                valid[idkey] = story
            elif story["partition"] == 'test':
                test[idkey] = story
        return train, valid, test

    def split_dict_percent(self, d, percent, shuffle=True):
        """
        Return two dictionaries with `percent` being size of smaller dict
        """
        keys = list(d.keys())
        if shuffle:
            random.shuffle(keys)
        n = int(len(keys)*percent)
        d1_keys = keys[:n]
        d2_keys = keys[-n:]
        d1 = {}
        d2 = {}
        for key, value in d.items():
            if key in d1_keys:
                d1[key] = value
            else:
                d2[key] = value
        return d1, d2

    def tokenize(self, d):
        """
        Tokenizes dictionary of text.
        Returns Torch LongTensor with all word ids
        """
        data = []
        # Add words to the dictionary
        tokens = 0
        # loop stories
        for idkey, story in d.items():
            # Indicate beginning of a story.
            sequence = ['<bos>']
            # loops lines in story
            for line, sentence in story["lines"].items():
                words = word_tokenize(sentence["text"])
                sequence.extend(words)
                sequence.append('<eol>') # end of line
            # Indicate end of story
            sequence.append('<eos>')
            tokens += len(sequence)
            data.append(sequence)
            for word in sequence:
                self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in data:
            for word in line:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids

def get_data(args):
    # Load data
    print("Loading Data...")
    if args.dataset == 'naive':
        args.data = '../data/story_commonsense/json_version/annotations.json'
        if os.path.isfile(args.prepared_data):
            print("Found data pickle, loading from {}".format(args.prepared_data))
            with open(args.prepared_data, 'rb') as p:
                d = pickle.load(p)
                corpus = d["corpus"]
                train_data = d["train_data"]
                val_data = d["val_data"]
                test_data = d["test_data"]
        else:
            corpus = data.NaivePsychCorpus(args.data, args.test_percent)
            train_data = utils.batchify(corpus.train, args.batch_size)
            val_data = utils.batchify(corpus.valid, args.eval_batch_size)
            test_data = utils.batchify(corpus.test, args.eval_batch_size)
            with open(args.prepared_data, 'wb') as p:
                d = {}
                d["corpus"] = corpus
                d["train_data"] = train_data
                d["val_data"] = val_data
                d["test_data"] = test_data
                pickle.dump(d, p, protocol=pickle.HIGHEST_PROTOCOL)
                print("Saved prepared data for future fast load to: {}".format(\
                                                            args.prepared_data))
    elif args.dataset=='wiki-2':
        args.data='data/wikitext-2'
        corpus = Corpus(data)
        train_data = batchify(corpus.train, args.batch_size)
        val_data = batchify(corpus.valid, args.eval_batch_size)
        test_data = batchify(corpus.test, args.eval_batch_size)

    elif args.dataset=='wiki-100':
        args.data='data/wikitext-103'
        corpus = Corpus(data)
        train_data = batchify(corpus.train, args.batch_size)
        val_data = batchify(corpus.valid, args.eval_batch_size)
        test_data = batchify(corpus.test, args.eval_batch_size)

    return train_data, val_data, test_data, corpus

def get_batch(args, source, i):
    """
    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more
    efficient batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

