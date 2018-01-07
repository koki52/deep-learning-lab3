import numpy as np


class Dataset:

    def __init__(self):
        self.char2id = None
        self.id2char = None
        self.x = None
        self.char_count = None
        self.sorted_chars = None

    def preprocess(self, input_file):
        with open(input_file, 'r') as f:
            data = f.read()

        # count and sort most frequent characters
        self.char_count = {}
        for char in data:
            if self.char_count[char] is None:
                self.char_count[char] = 0
            self.char_count[char] += 1
        self.sorted_chars = sorted(self.char_count, key=lambda x: self.char_count[x], reverse=True)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        encoded = ""
        for char in sequence:
            encoded += self.char2id(char)

        return encoded

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        decoded = ""
        for num in encoded_sequence:
            decoded += self.id2char[num]

        return decoded
