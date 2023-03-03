
import math
import os
import logging

import config
os.environ['TRANSFORMERS_CACHE'] = config.TRANSFORMERS_PATH

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedModel


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Abstract base class for models
################################################################################################

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Every model class should implement these so they can do saving/loading
        self.save_dir = None
        self.name = None

        # This does not have to be overridden in every model; just in model
        # classes that act as wrappers for HuggingFace models
        self.model = None

    def forward(self, batch):

        # Input: batch - a dictionary containing any inputs that the model needs
        # Output: another dictionary, containing any outputs that will be needed from the model
        raise NotImplementedError

    def trainable_parameters(self):

        # Yield the model's trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def save(self):
        logging.info("Saving model checkpoint to %s", self.save_dir)
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            state_dict = self.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, self.name + ".weights"))
        else:
            output_dir = os.path.join(self.save_dir, self.name)
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)

    def load(self, model_name=None):
        
        # Default to loading the best saved weights for this model
        # (if model_name is provided, this default is overridden to
        # instead load a different pretrained model)
        if model_name is None:
            model_name = os.path.join(self.save_dir, self.name)

        logging.info("Loading model checkpoint from %s", model_name)

        if isinstance(self.model, GPT2LMHeadModel):
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        else:
            self.load_state_dict(torch.load(model_name + ".weights"))




class LanguageModel(Model):

    def __init__(self):
        super(LanguageModel, self).__init__()

        # All language models should include a tokenizer
        tokenizer = None

    def forward(self, batch):

        # Input: batch - a dictionary containing "input_ids" (the tokenized IDs for the input)
        #        If a loss is to be computed, it should also contain "labels" - the gold labels
        #        that the loss will be computed with respect to
        # Output: dict containing "logits" (the logits for the next tokens) and potentially "loss"
        #         (the loss for this batch)
        raise NotImplementedError




################################################################################################
# Language models (i.e., next-word prediction models)
################################################################################################


# Based on code from here: https://github.com/pytorch/examples/tree/main/word_language_model
class RNNLM(LanguageModel):

    def __init__(self, rnn_type="LSTM", vocab_size=None, emb_size=None, hidden_size=None, n_layers=None, dropout=0.5, tie_weights=False, save_dir=None, model_name=None):
        super(RNNLM, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(dropout)
        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_size, hidden_size, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != emb_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to emb_size')
            self.out_layer.weight = self.emb_layer.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.emb_layer.weight, -initrange, initrange)
        nn.init.zeros_(self.out_layer.bias)
        nn.init.uniform_(self.out_layer.weight, -initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.hidden_size),
                    weight.new_zeros(self.n_layers, bsz, self.hidden_size))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.hidden_size)

    def forward(self, batch, per_token_loss=False):

        emb = self.drop(self.emb_layer(batch["input_ids"].transpose(0,1)))
        hidden = self.init_hidden(len(batch["input_ids"]))

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.out_layer(output)
        decoded = decoded.transpose(0,1)

        logits = decoded

        loss = None
        if "labels" in batch:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            if per_token_loss:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits" : logits, "loss" : loss}


    def generate(self, input_ids, do_sample=True, max_length=500, top_p=1.0, top_k=0, early_stopping=True, pad_token_id=None, eos_token_id=3):
        batch_size = len(input_ids)
        done = torch.zeros(batch_size).type(torch.uint8).to(device)
        sentence = input_ids
        for _ in range(60):
            logits = self.forward({"input_ids" : sentence})["logits"]
            logits = logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          
            cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -1000000)
         
            probabilities = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probabilities, 1, replacement=True)
            pred[done != 0] = pad_token_id

            sentence = torch.cat([sentence, pred], dim=1)
            eos_match = (pred.squeeze(1) == eos_token_id)
          
            done = done | eos_match
            if done.sum() == batch_size:
                break

        return sentence



class GPT2LM(LanguageModel):

    def __init__(self, vocab_size=None, emb_size=None, n_positions=None, n_layer=None, n_head=None, dropout=0.1, save_dir=None, model_name=None):
        super(GPT2LM, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        config = AutoConfig.from_pretrained(
                    "gpt2",
                    vocab_size=vocab_size,
                    n_embd=emb_size,
                    n_ctx=n_positions,
                    n_positions=n_positions,
                    n_layer=n_layer,
                    n_head=n_head,
                    resid_pdrop=dropout,
                    embd_pdrop=dropout,
                    attn_pdrop=dropout,
                )

        self.model = GPT2LMHeadModel(config).to(device)

    def forward(self, batch):

        return self.model(batch)




