import numpy as np


class Dataset:

    def __init__(self):
        self.char2id = None
        self.id2char = None
        self.x = None
        self.char_count = None
        self.sorted_chars = None
        self.num_batches = None
        self.batch_size = None
        self.sequence_length = None
        self.batches = []
        self.batch_pointer = 0

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

    def create_minibatches(self):
        self.num_batches = int(
            len(self.x) / (self.batch_size * self.sequence_length))  # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################

        pointer = 0
        for _ in range(self.num_batches):
            x = []
            y = []
            for __ in range(self.batch_size):
                x.append(self.x[pointer:pointer+self.sequence_length])
                y.append(self.x[pointer+1:pointer+self.sequence_length+1])
                pointer += self.sequence_length
            self.batches.append(Batch(x, y))
        pass

    def next_minibatch(self):
        current_batch = self.batches[self.batch_pointer]
        batch_x, batch_y = current_batch.x, current_batch.y
        # handling batch pointer & reset
        self.batch_pointer += 1
        if self.batch_pointer == len(self.batches):
            self.batch_pointer = 0
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        new_epoch = self.batch_pointer == 0
        return new_epoch, batch_x, batch_y


class Batch:
    def __init__(self, x, y):
        self.x = x
        self.y = y