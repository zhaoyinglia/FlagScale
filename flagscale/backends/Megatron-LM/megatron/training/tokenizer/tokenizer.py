# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import base64
import json
import math
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer
from .gpt2_tokenization import AquilaTokenizer
from megatron.training.tokenizer.multimodal_tokenizer import MultimodalTokenizer
from megatron.training.tokenizer.sft_tokenizer import SFTTokenizer


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=args.vocab_file, lower_case=True, vocab_extra_ids=args.vocab_extra_ids
        )
    elif args.tokenizer_type == 'BertWordPieceCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=args.vocab_file, lower_case=False, vocab_extra_ids=args.vocab_extra_ids
        )
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(
            args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids
        )
    elif args.tokenizer_type == 'GPTSentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        tokenizer = _HuggingFaceTokenizer(args.tokenizer_model, **kwargs)
    elif args.tokenizer_type == 'Llama2Tokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _Llama2Tokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'TikTokenizer':
        assert args.tokenizer_model is not None
        assert args.tiktoken_pattern is not None
        assert args.tiktoken_pattern in {"v1", "v2"}
        pattern = PATTERN_TIKTOKEN if args.tiktoken_pattern == "v1" else PATTERN_TIKTOKEN_V2
        tokenizer = CustomTikTokenizer(
            path=args.tokenizer_model,
            pattern=pattern,
            vocab_size=args.vocab_size,
            num_special_tokens=args.tiktoken_num_special_tokens,
            special_tokens=args.tiktoken_special_tokens,
        )
    elif args.tokenizer_type == 'NullTokenizer':
        assert args.vocab_size is not None
        tokenizer = _NullTokenizer(args.vocab_size)
    elif args.tokenizer_type == "MultimodalTokenizer":
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "MultimodalTokenizer currently requires transformers library to be installed"
            )

        kwargs = dict()
        if args.tokenizer_prompt_format == "nvlm-yi-34b":
            kwargs = {
                "from_slow": True,
                "legacy": False,
                "add_bos_token": True,
            }

        # Currently, only HuggingFace tokenizers are supported.
        underlying_tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.tokenizer_model, **kwargs
        )

        tokenizer = MultimodalTokenizer(
            underlying_tokenizer,
            args.tokenizer_prompt_format,
            args.special_tokens,
            args.image_tag_type,
            args.force_system_message,
        )
    elif args.tokenizer_type == "SFTTokenizer":
        tokenizer = SFTTokenizer(
            args.tokenizer_model,
            args.sft_tokenizer_prompt_format, 
        )
    elif args.tokenizer_type == 'NullMultimodalTokenizer':
        assert args.vocab_size is not None
        tokenizer = _NullMultimodalTokenizer(args.vocab_size)
    elif args.tokenizer_type == 'AquilaTokenizerFS':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        assert args.special_tokens_file is not None
        tokenizer = _AquilaTokenizerFS(args.vocab_file, args.merge_file,
                                     args.special_tokens_file)
    elif args.tokenizer_type == "HFTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _HFTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "Llama3TokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _Llama3TokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "QwenTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _QwenTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "HFTokenizersTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _HFTokenizersTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "Qwen2TokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _Qwen2TokenizerFS(args.tokenizer_path, args)
    elif args.tokenizer_type == 'Qwen2VLTokenizer':
        assert args.tokenizer_path is not None
        tokenizer = _Qwen2VLTokenizer(args.tokenizer_path, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size # no padding
    else:
        raise NotImplementedError('{} tokenizer is not ' 'implemented.'.format(args.tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            ' > padded vocab (size: {}) with {} dummy tokens '
            '(new size: {})'.format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class _HuggingFaceTokenizer(MegatronTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface_tokenizer_provider"
            )

        # TODO(bnorick): download tokenizer once to lustre and use force offline to make sure all tasks read it from there
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._tokenizer)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    def tokenize(self, text, **kwargs):
        return self._tokenizer(text, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        return self._tokenizer.decode(token_ids, **kwargs)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        retok_ids: "transformers.BatchEncoding" = self._tokenizer(text)
        offsets, next_start_idx = [], 0
        for i in range(len(ids)):
            span = retok_ids.token_to_chars(i)
            if span is not None:
                offsets.append(span.start)
                next_start_idx = span.end
            else:
                offsets.append(next_start_idx)
        return offsets

    @property
    def eod(self):
        return self._tokenizer.eos_token_id

    @property
    def bos(self):
        return self._tokenizer.bos_token_id


class _BertWordPieceTokenizer(MegatronTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        super().__init__(vocab_file, lower_case=lower_case, vocab_extra_ids=vocab_extra_ids)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        SPECIAL_TOKENS = {'eos_token': '[EOS]', 'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)]
        )
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def detokenize(self, token_ids):
        """Copy of decode() method for inference pipeline compatibility"""
        return self.decode(token_ids)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos(self):
        """Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos(self):
        """Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def eod(self):
        """Copy of eod property for inference pipeline compatibility"""
        return self.eos

    @property
    def bos_token(self):
        """Beginning of sentence token id"""
        return self._bos_token

    @property
    def eos_token(self):
        """End of sentence token id"""
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def additional_special_tokens_ids(self):
        """Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _GPT2BPETokenizer(MegatronTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        super().__init__(vocab_file, merge_file)

        self.tokenizer = GPT2Tokenizer(
            vocab_file, merge_file, errors='replace', special_tokens=[], max_len=None
        )
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class _SentencePieceTokenizer(MegatronTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids=vocab_extra_ids)

        import sentencepiece

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    def offsets(self, ids: list[int], text: str) -> list[int]:
        return [p.begin for p in self.tokenizer.decode_ids_as_immutable_proto(ids).pieces]

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]


class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _Llama2Tokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        # BOS / EOS token IDs
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

    def tokenize(self, s: str, bos=True, eos=False):
        '''Default args for text completion, not chat/dialog.'''
        assert type(s) is str
        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def instruct_tokenize(self, s: str, bos=True, eos=False):
        '''Default args for text completion, not chat/dialog.'''
        assert type(s) is str
        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self.eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


def reload_mergeable_ranks(path: str, max_vocab: Optional[int] = None) -> Dict[bytes, int]:
    """
    Reload our tokenizer JSON file and convert it to Tiktoken format.
    """
    from ..utils import print_rank_0  # To prevent circular import.

    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print_rank_0(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print_rank_0(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


PATTERN_TIKTOKEN = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
PATTERN_TIKTOKEN_V2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"


class CustomTikTokenizer(MegatronTokenizer):
    def __init__(
        self,
        path: str,
        pattern: str,
        vocab_size: Optional[int],
        num_special_tokens: int,
        special_tokens: Optional[List[str]],
    ):
        super().__init__(
            path,
            pattern=pattern,
            vocab_size=vocab_size,
            num_special_tokens=num_special_tokens,
            special_tokens=special_tokens,
        )
        import tiktoken

        from .. import print_rank_0  # To prevent circular import.

        if vocab_size is None:
            vocab_size = 2**17  # Fallback vocab size is 131072.
        self._vocab_size = vocab_size

        SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()
        assert len(special_tokens) == len(
            set(special_tokens)
        ), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < self._vocab_size
        assert set(SPECIAL_TOKENS) <= set(
            special_tokens
        ), f"Custom special tokens should include {SPECIAL_TOKENS}"

        special_filler = [
            "<SPECIAL_{id}>".format(id=i) for i in range(len(special_tokens), num_special_tokens)
        ]
        if special_filler:
            print_rank_0(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        special_tokens = special_tokens + special_filler
        assert len(set(special_tokens)) == len(special_tokens) == num_special_tokens, special_tokens
        inner_vocab_size = self._vocab_size - num_special_tokens

        token_to_id_without_special_tokens = reload_mergeable_ranks(
            path, max_vocab=inner_vocab_size
        )
        # Create space for special tokens.
        token_to_id_without_special_tokens = {
            t: i + num_special_tokens for t, i in token_to_id_without_special_tokens.items()
        }

        special_tokens = {t: i for i, t in enumerate(special_tokens)}
        self._unk_id = special_tokens["<unk>"]
        self._bos_id = special_tokens["<s>"]
        self._eos_id = special_tokens["</s>"]

        # Create tiktoken model.
        self._model = tiktoken.Encoding(
            name=Path(path).parent.name,
            pat_str=pattern,
            mergeable_ranks=token_to_id_without_special_tokens,
            special_tokens=special_tokens,
        )

        # Create final _id_to_token and _token_to_id data structures with special tokens inserted
        # into appropriate locations.
        assert set(token_to_id_without_special_tokens.keys()).isdisjoint(set(special_tokens.keys()))
        self._token_to_id = token_to_id_without_special_tokens.copy()
        self._token_to_id.update(special_tokens)
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        assert set(range(self._vocab_size)) == set(self._id_to_token.keys())

    @property
    def bos(self) -> int:
        return self._bos_id

    @property
    def eos(self) -> int:
        return self._eos_id

    @property
    def unk(self) -> int:
        return self._unk_id

    @property
    def eod(self) -> int:
        return self._eos_id

    @property
    def vocab(self):
        return self._token_to_id

    @property
    def inv_vocab(self):
        return self._id_to_token

    def tokenize(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        tokens = self._model.encode_ordinary(s)
        if bos:
            tokens = [self.bos, *tokens]
        if eos:
            tokens = [*tokens, self.eos]

        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        return self._model.decode(tokens)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        try:
            return self._model.decode_with_offsets(ids)[1]
        except UnicodeDecodeError:
            # Tiktoken has an unnecessary check that raises UnicodeDecodeError
            # from `text = b"".join(token_bytes).decode("utf-8", errors="strict")`
            # which is not needed for our use case. So we re-implement it, without
            # the check.

            token_bytes = self._model.decode_tokens_bytes(ids)
            text_len = 0
            offsets = []
            for token in token_bytes:
                offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
                text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
            return offsets

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def encoder(self):
        return self._token_to_id

    @property
    def decoder(self):
        return self._id_to_token


class _NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None

class _NullMultimodalTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size, image_token=None, image_token_id=None):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

        from megatron.core.models.multimodal.llava_model import DEFAULT_IMAGE_TOKEN_INDEX, IMAGE_TOKEN
        self._image_token = image_token if image_token is not None else IMAGE_TOKEN
        self._image_token_id = image_token_id if image_token_id is not None else DEFAULT_IMAGE_TOKEN_INDEX

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    def convert_tokens_to_ids(self, tokens):
        ids = [(int(t) if t != self._image_token else self._image_token_id) for t in tokens.split('  ')]
        return ids if len(ids) > 1 else ids[0]

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _AquilaTokenizerFS(MegatronTokenizer):
    """Aquila tokenizer."""

    def __init__(self, vocab_file, merge_file, special_tokens_file):
        super().__init__(vocab_file, merge_file, special_tokens_file)

        special_tokens = []
        if special_tokens_file:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]

        self.tokenizer = AquilaTokenizer(vocab_file, merge_file, errors='replace',
                                         special_tokens=special_tokens, max_len=None)
        self.eod_id = self.tokenizer.encoder['</s>']
        self.cls_id = self.tokenizer.encoder['[CLS]']
        self.pad_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _HFTokenizerFS(MegatronTokenizer):
    """Huggingface tokenizer."""

    def __init__(self, tokenizer_path):
        name = 'HFTokenizer'
        super().__init__(name)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.eod_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self._inv_vocab = None
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab()
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _Llama3TokenizerFS(_HFTokenizerFS):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())


class _QwenTokenizerFS(_HFTokenizerFS):
    """Adapted Qwen tokenizer."""
    
    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]


class _HFTokenizersTokenizerFS(MegatronTokenizer):
    """Tokenizer from HuggingFace Tokenizers."""

    def __init__(self, json_file):
        super().__init__(json_file)

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(json_file)

        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")

        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size() 

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        # return self.tokenizer.decoder
        vocab = self.vocab()
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab 

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad(self):
        return self.pad_id


class _Qwen2TokenizerFS(_HFTokenizerFS):
    """Adapted Qwen tokenizer."""

    def __init__(self, tokenizer_path, args):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]
        assert args.vocab_size is not None
        self._vocab_size = args.vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size


class _Qwen2VLTokenizer(MegatronTokenizer):
    def __init__(self, tokenizer_path, extra_vocab_size):
        super().__init__(tokenizer_path)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=True,
            split_special_tokens=False,
            trust_remote_code=True,
            revision = "main",
            token = None,
        )
        self.extra_vocab_size = extra_vocab_size
        self.special_tokens_map = {k:v for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)}
        self.image_token = '<|image_pad|>'
        self.video_token = '<|video_pad|>'
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            revision = "main",
            token = None,
        )
        # NOTE: In Qwen2-VL, template in chat_template.json is same within tokenizer_config.json and both can be used.
        # However, in Qwen 2.5-VL, the two templates are different and only the one in chat_template.json is OK.
        self.chat_template = self.processor.chat_template

    def __call__(self, text, return_tensors=None,
                    padding=None, max_length=None, truncation=None, add_special_tokens=None):

        return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

    def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
        return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, chat_template=self.chat_template, **kwargs)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]
    
    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]
    
    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]
    
    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]
    
    def encode(self, x):
        return self.tokenizer.encode(x)