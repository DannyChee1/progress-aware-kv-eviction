"""Fixtures shared by the tests: a character-level tokenizer and a scripted stand-in model."""
import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


def tokenizer_fixture():
    vocab = {c: i for i, c in enumerate(['[UNK]', '[EOS]'] + list(dict.fromkeys(string.printable)))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
                                        clean_up_tokenization_spaces=False)
    tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}assistant:"
    return tokenizer


class ScriptedModel(torch.nn.Module):
    def __init__(self, tokens, vocab):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.tokens = tokens
        self.vocab = vocab
        self.calls = []

    def forward(self, input_ids, past_key_values, **kwargs):
        from types import SimpleNamespace
        self.calls.append((input_ids.shape[1], past_key_values.get_seq_length(), kwargs.get('position_ids')))
        for layer in range(2):
            kv = torch.zeros(1, 2, input_ids.shape[1], 4)
            past_key_values.update(kv, kv.clone(), layer)
        index = len(self.calls) - 1
        token = self.tokens[index] if index < len(self.tokens) else 1
        logits = torch.full((1, 1, self.vocab), -100.0)
        logits[0, 0, token] = 100
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)
