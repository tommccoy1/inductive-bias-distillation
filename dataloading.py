
import os
import config

os.environ['TRANSFORMERS_CACHE'] = config.TRANSFORMERS_PATH
import sys

import math
import random
import numpy as np
from collections import Counter
import logging 

import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast

from utils import *

  
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




################################################################################################
# Abstract base classes for datasets and data splits
# For your specific use case, you should create new classes that inherit from these
# and override the attributes and methods listed in the abstract class
################################################################################################

# Abstract base class for a dataset
class Dataset:

    def __init__(self):
        
        # Should include 3 data splits - train, valid, and test
        self.train = None
        self.valid = None
        self.test = None

    def prepare_input(self, batch):
        
        # Keys in the batch that can be converted into tensors
        tensorizable_keys = ["input_ids", "labels", "attention_mask", "test_input_ids", "test_labels", "test_attention_mask"]

        prepared_batch = {}

        # Convert to tensors and put on the proper device (CPU or GPU)
        for key in batch:
            if key in tensorizable_keys:
                prepared_batch[key] = torch.LongTensor(batch[key]).to(device)
            else:
                prepared_batch[key] = batch[key]

        # Deal with labels for padding tokens
        if "labels" in prepared_batch:
            labels = prepared_batch["labels"].clone()
        else:
            labels = prepared_batch["input_ids"].clone()

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        prepared_batch["labels"] = labels

        # If it's a meta-batch, deal with labels for padding in the test set
        if "test_input_ids" in prepared_batch:
            if "test_labels" in prepared_batch:
                test_labels = prepared_batch["test_labels"].clone()
            else:
                test_labels = prepared_batch["test_input_ids"].clone()

            if self.tokenizer.pad_token_id is not None:
                test_labels[test_labels == self.tokenizer.pad_token_id] = -100

            prepared_batch["test_labels"] = test_labels

            train_batches, test_batches = meta_mini_batches_from_batch(prepared_batch, prepared_batch["train_batch_size"], prepared_batch["eval_batch_size"], self.tokenizer.pad_token_id)

            prepared_batch["train_batches"] = train_batches
            prepared_batch["test_batches"] = test_batches


        return prepared_batch



# Abstract base class for a data split
class DataSplit:

    def __init__(self):

        # There are no specific attributes that need to be included
        pass

    def __iter__(self):
        return self

    def __len__(self):
        # Should return the length of the data split (i.e., the number of batches in it)
        raise NotImplementedError

    def __next__(self):
        # Should return the next batch (or raise StopIteration if you've
        # reached the end of the data split)
        raise NotImplementedError

    def reset(self):
        # Should reset the data split so that it can be iterated over again
        raise NotImplementedError




################################################################################################
# Data classes for language modeling with packed sequences loaded from a file
# "packed": We pack as many tokens as possible into one batch. This means that a batch (and the
# individual sequences within a batch) might start partway through a line/sentence and end 
# partway through a line/sentence
################################################################################################



# Dataset for language modeling (next-word prediction) extracted from files of text
# Contains training set, validation set, and test set
# - directory: Directory containing corpus. Should contain train.txt, valid.txt, and test.txt
# - add_eos: Whether to dd an EOS token at the end of every line in the corpus
# - batch_size: Number of sequences per batch
# - context_size: Max number of tokens that the model will consider per sequence
# - batches_per_buffer: Number of batches stored in memory at once
# - loop: Whether to loop over the corpus
# - stream: Whether to use streaming (instead of storing the whole dataset in memory)
# - shuffle: Whether to shuffle the batches that are in memory, as opposed to processing them in order
# - stride: Number of tokens at the start of every sequence that do not have a loss computed (for training
#   of evaluation); instead they serve only as context for later tokens, ensuring that every token has at
#   least 'stride' tokens of context
# - valid_stride, test_stride: Stride to use in special extra-stride versions of the validation and test sets
class LMDatasetFromFiles(Dataset):

    def __init__(self, directory=None, add_eos=False, batch_size=None, context_size=None, batches_per_buffer=None, loop=False, stream=True, shuffle=True, stride=0, valid_stride=None, test_stride=None):

        super(LMDatasetFromFiles, self).__init__()

        # Create a tokenizer based on the training set
        self.tokenizer = self.tokenizer_from_file(directory=directory, add_eos=add_eos)

        # Load the training, validation, and test data
        self.train = LMDataSplitFromFile(filename=directory + "train.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, loop=loop, stream=stream, tokenizer=self.tokenizer, shuffle=shuffle, stride=stride, add_eos=add_eos, prepare_input=self.prepare_input)
        self.valid = LMDataSplitFromFile(filename=directory + "valid.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, loop=loop, stream=stream, tokenizer=self.tokenizer, shuffle=shuffle, stride=stride, add_eos=add_eos, prepare_input=self.prepare_input)
        self.test = LMDataSplitFromFile(filename=directory + "test.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, loop=loop, stream=stream, tokenizer=self.tokenizer, shuffle=shuffle, stride=stride, add_eos=add_eos, prepare_input=self.prepare_input)

        
        # Create versions of the validation and test sets that
        # have a larger stride
        if valid_stride is None:
            valid_stride = context_size//2

        if test_stride is None:
            test_stride = context_size-1

        self.stride_valid = LMDataSplitFromFile(filename=directory + "valid.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, loop=loop, stream=stream, tokenizer=self.tokenizer, shuffle=shuffle, stride=valid_stride, add_eos=add_eos, prepare_input=self.prepare_input)
        self.stride_test = LMDataSplitFromFile(filename=directory + "test.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, loop=loop, stream=stream, tokenizer=self.tokenizer, shuffle=shuffle, stride=test_stride, add_eos=add_eos, prepare_input=self.prepare_input)


    def tokenizer_from_file(self, directory=None, add_eos=False):
        # train a word-level tokenizer on the training set

        # iterator that yields the lines in the training set
        fi_train = open(directory + "train.txt", "r")
        def get_training_corpus(add_eos=add_eos):
            for line in fi_train:
                # Get rid of <unk> to preserve its special tokenization
                # Also get rid of double spaces this might create
                line = line.replace("<unk>", "")
                line = line.replace("  ", " ")

                if add_eos:
                    yield [line.strip() + " <eos>"]
                else:
                    yield [line]

        # Splits only on whitespace - assumes data are pre-processed (e.g., punctuation separated from words)
        whitespace_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        whitespace_tokenizer.pre_tokenizer = WhitespaceSplit()

        # Vocab size truncated at 1 million; if you need larger, this should be changed
        tokenizer_trainer = WordLevelTrainer(vocab_size=1000000, special_tokens=["<unk>", "<pad>"])

        # Train the tokenizer
        whitespace_tokenizer.train_from_iterator(get_training_corpus(), tokenizer_trainer)

        fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=whitespace_tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                )

        return fast_tokenizer



# Designed to give you one batch at a time from the file
# at filename. It has streaming enabled to save memory: at a given
# time, only load some lines from the file
class LMDataSplitFromFile(DataSplit):

    def __init__(self, filename=None, batch_size=None, context_size=None, batches_per_buffer=None, loop=False, stream=True, shuffle=True, tokenizer=None, stride=0, add_eos=False, prepare_input=None):

        super(LMDataSplitFromFile, self).__init__()
        
        # Buffer of text drawn from the corpus
        self.current_tokens = []

        # Buffer of batches
        self.current_batches = []
        
        # Where in self.current_tokens and self.current_batches
        # we currently are
        self.token_pointer = 0
        self.batch_pointer = 0

        self.tokenizer = tokenizer
 
        # Whether we loop infinitely over the data (as opposed to
        # stopping iteration at the end of the file)
        self.loop = loop

        # Whether to read data in a streaming way (as opposed to reading 
        # in the whole file at once)
        self.stream = stream

        # Whether to shuffle the batches with the buffer
        self.shuffle = shuffle

        # Number of un-loss-computed tokens to have at the start of each sequence
        self.stride = stride

        # Stride that is present in self.current_tokens; starts out at 0
        # because there are no previous tokens to use for the stride
        self.stride_on_current_tokens = 0

        # Whether to add an <eos> token at the end of every line
        self.add_eos = add_eos
       
        self.filename = filename
        self.fi = open(filename, "r")
        self.at_end_of_file = False

        self.context_size = context_size
        self.batch_size = batch_size
        self.batches_per_buffer = batches_per_buffer
        self.tokens_per_batch = self.context_size * self.batch_size

        # Number of tokens we need to have on hand to fill the buffer
        if self.stream:
            self.length_text_in_buffer = self.tokens_per_batch * self.batches_per_buffer
        
        else:
            # So that we read lines until the file ends
            self.length_text_in_buffer = math.inf
            self.batches_per_buffer = math.inf

        self.length = None

        self.prepare_input = prepare_input

        # Populate the buffer
        self.reload_buffer()


    def __len__(self):
        # Return the length of the dataset
        # The first time, we have to compute it. After that,
        # it's stored.

        if self.length is not None:
            return self.length
        else:
            self.reset()
            n_batches = 0
            for _ in self:
                n_batches += 1
            self.length = n_batches
            self.reset()

            return n_batches

    def __next__(self):

        # We've used up all the batches in the buffer
        if self.batch_pointer == len(self.current_batches):
            self.batch_pointer = 0

            # We've reached the end of the file
            if self.at_end_of_file:

                # We reset the dataset at the start of the file
                # so we can continue looping indefinitely
                if self.loop:
                    self.reset()

                # Indicate that we have reached the end of the file
                else:
                    raise StopIteration

            # Not at the end of the file, so we simply reload the buffer in order
            # to have enough text to make a batch
            else:
                self.current_batches = []
                self.reload_buffer()

        if len(self.current_batches) == 0:
            raise StopIteration

        # Return the next batch in the buffer
        next_batch = self.current_batches[self.batch_pointer]
        self.batch_pointer += 1

        return self.prepare_input(next_batch)
        
    
    def reset(self, offset=None):

        if offset is not None or self.stream:
            # Need to reopen the file and start from the top
            self.fi.close()

            self.fi = open(self.filename, "r")
            self.at_end_of_file = False

            self.current_tokens = []
            self.stride_on_current_tokens = 0

            if offset is not None:
                # Add pad tokens at the start to implement the offset
                pad_tokens = [self.tokenizer.pad_token_id for _ in range(offset)]
                self.current_tokens = self.current_tokens + pad_tokens

            self.current_batches = []
            self.token_pointer = 0
            self.batch_pointer = 0

            self.reload_buffer()

            # If not streaming, reloading the buffer would have used up
            # the whole file
            if not self.stream:
                self.fi.close()

        else:
            # No need to reopen the file: instead, just return
            # the pointer to the start of current_tokens

            if self.shuffle:
                random.shuffle(self.current_batches)

            self.token_pointer = 0
            self.batch_pointer = 0

    def current_tokens_to_batches(self, pad_to_complete_batches=False):
        
        # Make batches out of the current buffer of tokens


        if pad_to_complete_batches:
            # Make the current text long enough to evenly split into batches
            if len(self.current_tokens) % self.tokens_per_batch == 0:
                pass
            else:
                desired_length = self.tokens_per_batch * ((len(self.current_tokens) // self.tokens_per_batch) + 1)
                pad_token_length = desired_length - len(self.current_tokens)
                pad_tokens = [self.tokenizer.pad_token_id for _ in range(pad_token_length)]
                self.current_tokens = self.current_tokens + pad_tokens

        # Split the text into batches
        while self.token_pointer + self.tokens_per_batch <= len(self.current_tokens):
            batch = []
            labels = []
            for i in range(self.batch_size):
                seq = self.current_tokens[self.token_pointer:self.token_pointer + self.context_size]

                seq_labels = seq[:]
                for j in range(self.stride_on_current_tokens):
                    # No label for stride tokens (so that no loss is computed)
                    seq_labels[j] = self.tokenizer.pad_token_id
                
                # Don't include if it's all padding (arises from having everything past
                # the stride be padding)
                # Ignore first one, since it's never evaluated on
                if not all(x == self.tokenizer.pad_token_id for x in seq_labels[1:]): 
                    batch.append(seq)
                    labels.append(seq_labels)

                # Subtracting stride ensures that we reuse the end of the current sequence at the start of the next one
                self.token_pointer = self.token_pointer + self.context_size - self.stride
                self.stride_on_current_tokens = self.stride

            if len(batch) != 0:
                self.current_batches.append({"input_ids" : batch, "labels" : labels})

        self.current_tokens = self.current_tokens[self.token_pointer:]
        self.token_pointer = 0


    def reload_buffer(self):

        # First, read enough text from the file to
        # fill up the buffer
        while len(self.current_tokens) < self.length_text_in_buffer:
                
            # Read in 1000 lines at a time; saves a lot of time to 
            # reduce the calls to the tokenizer (tokenizing one
            # long string instead of many short strings)
            lines = ""
            for _ in range(1000):
                line = self.fi.readline()
                lines = lines + line
                if self.add_eos:
                    lines = lines.strip() + " <eos> "

                if line == "":
                    self.at_end_of_file = True
                    break

            if lines == "":
                break

            tokens = self.tokenizer.encode(lines)
            
            self.current_tokens = self.current_tokens + tokens

            if self.at_end_of_file:
                break

        # Then, parcel that text into batches (saving the remainder that
        # was left over for use in the next set of batches)
        self.current_tokens_to_batches(pad_to_complete_batches=self.at_end_of_file)

        if self.shuffle:
            random.shuffle(self.current_batches)




################################################################################################
# Data classes for meta-language modeling: Meta-learning where the task is language
# modeling (next-word prediction) and each episode is a different language
# Here, the data are not packed: each input to the model is exactly one sentence
################################################################################################


# A dataset of datasets
# That is, each dataset is one corpus - one episode for meta-training
class MetaLMDataset(Dataset):
    # create_dataset: Function that takes in a random seed and returns a dataset.
    #     The dataset that is returned should be a dict with the following key, value pairs:
    #     - ["train", a list of training strings, with space-delimited tokens within each string]
    #     - ["test", a list of test strings, with space-delimited tokens within each string]
    # meta_train_size, meta_valid_size, meta_test_size: Number of tasks
    #     to include in each meta data split. You can leave meta_train_size
    #     as None to have it generate indefinitely.
    # integer_vocab_size: If all the words in the vocab are positive integers,
    #     gives the max integer in the vocab.
    def __init__(self, create_dataset=None, meta_train_size=None, meta_valid_size=None, meta_test_size=None, integer_vocab_size=None, context_size=None):

        super(MetaLMDataset, self).__init__()

        self.create_dataset = create_dataset

        self.meta_train_size = meta_train_size
        self.meta_valid_size = meta_valid_size
        self.meta_test_size = meta_test_size

        self.integer_vocab_size = integer_vocab_size
        self.context_size = context_size

        # Create a tokenizer (self.tokenizer) and a 
        # tokenization function (self.tokenize)
        self.create_tokenizer()


        # Define each dataset split
        self.train = MetaLMDataSplit(length=self.meta_train_size, tokenize=self.tokenize, create_dataset=self.create_dataset, initial_index=meta_valid_size+meta_test_size, context_size=self.context_size, prepare_input=self.prepare_input)
        self.valid = MetaLMDataSplit(length=self.meta_valid_size, tokenize=self.tokenize, create_dataset=self.create_dataset, initial_index=0, context_size=self.context_size, remember_languages=True, prepare_input=self.prepare_input)
        self.test = MetaLMDataSplit(length=self.meta_test_size, tokenize=self.tokenize, create_dataset=self.create_dataset, initial_index=self.meta_valid_size, context_size=self.context_size, prepare_input=self.prepare_input)


    def create_tokenizer(self):
        tok = Tokenizer(WordLevel(unk_token="<unk>"))

        # Splits only on whitespace - assumes data are pre-processed
        tok.pre_tokenizer = WhitespaceSplit()

        # Adds bos and eos token to every sequence
        # 2 and 3 because of position in list in next line
        tok.post_processor = TemplateProcessing(single="<bos> $0 <ENDTOKEN>", special_tokens=[("<bos>", 2), ("<ENDTOKEN>", 3)],)

        tok_trainer = WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<ENDTOKEN>"])

        if self.integer_vocab_size is not None:
            # Just give it the vocab that we provided
            tok.train_from_iterator([str(i) for i in range(self.integer_vocab_size)], tok_trainer)

        elif self.meta_train_size is not None:
            # Iterate over the training set and induce the vocab from that
            tok.train_from_iterator([self.create_dataset(i)["train"] for i in range(self.meta_valid_size + self.meta_test_size, self.meta_valid_size + self.meta_test_size + self.meta_train_size)], tok_trainer)
        else:
            raise ValueError('To create corpus, you need to specify either integer_vocab_size or train_size')

        # Don't tell it about the EOS token because sometimes that causes the tokenizer
        # to truncate the output where we don't want it to be truncated
        wrapped_tok = PreTrainedTokenizerFast(tokenizer_object=tok,
                                              unk_token="<unk>",
                                              pad_token="<pad>",
                                              bos_token="<bos>",
                                             )

        self.tokenizer = wrapped_tok


        # Also create a tokenization function using this tokenizer
        def tokenize(dataset):
            train_outputs = self.tokenizer(dataset["train"],
                                           padding="longest",
                                           truncation=True,
                                           max_length=self.context_size,
                                           return_overflowing_tokens=False,
                                          )

            test_outputs = self.tokenizer(dataset["test"],
                                          padding="longest",
                                          truncation=True,
                                          max_length=self.context_size,
                                          return_overflowing_tokens=False,
                                         )

            return_dict = {"input_ids" : train_outputs["input_ids"], "attention_mask" : train_outputs["attention_mask"], "test_input_ids" : test_outputs["input_ids"], "test_attention_mask" : test_outputs["attention_mask"]}

            for key in dataset:
                if key != "train" and key != "test":
                    return_dict[key] = dataset[key]

            return return_dict

        self.tokenize = tokenize



# Based on a dataset iterator, rather than a dataset file
# The meta training set, meta validation set, and meta test set are
# all MetaDataSplits
# - length: Number of episodes
# - tokenize: Function for tokenizing a dataset (takes in dataset, returns tokenized dataset)
# - initial_index: index of the first episode (so, it runs from initial_index to initial_index + length)
# - context_size: context size of the model, so we can truncate text to that
# - remember_languages: When withholding certain languages, remember all languages we have
#   seen so that we know if they should be withheld or not
class MetaLMDataSplit(DataSplit):

    def __init__(self, length=None, tokenize=None, create_dataset=None, initial_index=None, context_size=None, remember_languages=False, prepare_input=None):

        super(MetaLMDataSplit, self).__init__()

        self.length = length
        self.tokenize = tokenize
        self.create_dataset = create_dataset

        self.initial_index = initial_index
        self.current_index = initial_index

        self.context_size = context_size

        self.remember_languages = remember_languages
        if self.remember_languages:
            self.remembered_languages = {}

        self.prepare_input = prepare_input

    def __len__(self):
        return self.length

    def __next__(self):

        if self.current_index == self.initial_index + self.length:
            # We've used the whole dataset
            raise StopIteration

        if self.remember_languages:
            to_return = self.tokenize(self.create_dataset(self.current_index, remembered_languages=self.remembered_languages))
        else:
            to_return = self.tokenize(self.create_dataset(self.current_index))

        to_return = self.prepare_input(to_return)

        self.current_index += 1
        return to_return

    def reset(self, offset=None):
        self.current_index = self.initial_index



if __name__ == "__main__":
    from dataset_iterators import *

    create_scfg_dataset = scfg_dataset(n_test=5, batch_size=10, eval_batch_size=10, max_batches_per_language=3)
    meta_dataset = MetaLMDataset(create_dataset=create_scfg_dataset, meta_train_size=10, meta_valid_size=3, meta_test_size=4, integer_vocab_size=10, context_size=128)

    for elt in meta_dataset.train:
        print(elt)


    lm_dataset = LMDatasetFromFiles(directory="CHILDES/pretraining_half_2/", add_eos=False, batch_size=3, context_size=7, batches_per_buffer=10, loop=False, stream=True, shuffle=False, stride=0, valid_stride=None, test_stride=None)
    for word in ["the", "of", "I", "elephant", "dog", "house", "purple", "basketball", "sitting"]:
        print(word, lm_dataset.tokenizer.encode(word))
    for elt in lm_dataset.train:
        print(elt)
        break

