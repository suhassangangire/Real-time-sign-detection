import os
from ..tokenizers import CoreNLPTokenizer
from ..retriever import TfidfDocRanker
from ..retriever import DocDB
from .. import DATA_DIR

DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'reader_model': os.path.join(DATA_DIR, 'reader/multitask.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .drqa import DrQA
import torch
import regex
import heapq
import math
import time
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from ..reader.vector import batchify
from ..reader.data import ReaderDataset, SortedBatchSampler
from .. import reader
from .. import tokenizers
from . import DEFAULTS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Main DrQA pipeline
# ------------------------------------------------------------------------------


class DrQA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            db_config=None,
            ranker_config=None
    ):
        """Initialize the pipeline.
        Args:
            reader_model: model file from which to load the DocReader.
            embedding_file: if given, will expand DocReader dictionary to use
              all available pretrained embeddings.
            tokenizer: string option to specify tokenizer used on docs.
            fixed_candidates: if given, all predictions will be constrated to
              the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use the gpu.
            data_parallel: whether to use multile gpus.
            max_loaders: max number of async data loading workers when reading.
              (default is fine).
            num_workers: number of parallel CPU processes to use for tokenizing
              and post processing resuls.
            db_config: config for doc db.
            ranker_config: config for ranker.
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing model...')
        reader_model = reader_model or DEFAULTS['reader_model']
        self.reader = reader.DocReader.load(reader_model, normalize=False)
        if embedding_file:
            logger.info('Expanding dictionary...')
            words = reader.utils.index_embedding_words(embedding_file)
            added = self.reader.expand_dictionary(words)
            self.reader.load_embeddings(added, embedding_file)
        if cuda:
            self.reader.cuda()
        if data_parallel:
            self.reader.parallelize()

        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)
        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        db_config = db_config or {}
        db_class = db_config.get('class', DEFAULTS['db'])
        db_opts = db_config.get('options', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def _get_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )
        return loader

    def process(self, query, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        """Run a single query."""
        predictions = self.process_batch(
            [query], [candidates] if candidates else None,
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=1, n_docs=5,
                      return_context=False):
        """Run a batch of queries (more efficient)."""
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        for text in doc_texts:
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        for qidx in range(len(queries)):
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(q_tokens[qidx].words()) > 0 and
                            len(s_tokens[sidx].words()) > 0):
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': q_tokens[qidx].words(),
                            'qlemma': q_tokens[qidx].lemmas(),
                            'document': s_tokens[sidx].words(),
                            'lemma': s_tokens[sidx].lemmas(),
                            'pos': s_tokens[sidx].pos(),
                            'ner': s_tokens[sidx].entities(),
                        })

        logger.info('Reading %d paragraphs...' % len(examples))

        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_loader(examples, num_loaders):
            if candidates or self.fixed_candidates:
                batch_cands = []
                for ex_id in batch[-1]:
                    batch_cands.append({
                        'input': s_tokens[ex_id[2]],
                        'cands': candidates[ex_id[0]] if candidates else None
                    })
                handle = self.reader.predict(
                    batch, batch_cands, async_pool=self.processes
                )
            else:
                handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))

        # Iterate through the predictions, and maintain priority queues for
        # top scored answers for each question in the batch.
        queues = [[] for _ in range(len(queries))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)

        # Arrange final top prediction data.
        all_predictions = []
        for queue in queues:
            predictions = []
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)
                prediction = {
                    'doc_id': all_docids[qidx][rel_didx],
                    'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': float(score),
                }
                if return_context:
                    prediction['context'] = {
                        'text': s_tokens[sidx].untokenize(),
                        'start': s_tokens[sidx].offsets()[s][0],
                        'end': s_tokens[sidx].offsets()[e][1],
                    }
                predictions.append(prediction)
            all_predictions.append(predictions[-1::-1])

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))

        return all_predictions
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from ..tokenizers import CoreNLPTokenizer
from .. import DATA_DIR


DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'model': os.path.join(DATA_DIR, 'reader/single.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

from .model import DocReader
from .predictor import Predictor
from . import config
from . import vector
from . import data
from . import utils
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Model architecture/optimization options for DrQA document reader."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'embedding_dim', 'hidden_size', 'doc_layers',
    'question_layers', 'rnn_type', 'concat_rnn_layers', 'question_merge',
    'use_qemb', 'use_in_question', 'use_pos', 'use_ner', 'use_lemma', 'use_tf'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rnn_padding', 'dropout_rnn', 'dropout_rnn_output', 'dropout_emb',
    'max_len', 'grad_clipping', 'tune_partial'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('DrQA Reader Model Architecture')
    model.add_argument('--model-type', type=str, default='rnn',
                       help='Model architecture type')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size of RNN units')
    model.add_argument('--doc-layers', type=int, default=3,
                       help='Number of encoding layers for document')
    model.add_argument('--question-layers', type=int, default=3,
                       help='Number of encoding layers for question')
    model.add_argument('--rnn-type', type=str, default='lstm',
                       help='RNN type: LSTM, GRU, or RNN')

    # Model specific details
    detail = parser.add_argument_group('DrQA Reader Model Details')
    detail.add_argument('--concat-rnn-layers', type='bool', default=True,
                        help='Combine hidden states from each encoding layer')
    detail.add_argument('--question-merge', type=str, default='self_attn',
                        help='The way of computing the question representation')
    detail.add_argument('--use-qemb', type='bool', default=True,
                        help='Whether to use weighted question embeddings')
    detail.add_argument('--use-in-question', type='bool', default=True,
                        help='Whether to use in_question_* features')
    detail.add_argument('--use-pos', type='bool', default=True,
                        help='Whether to use pos features')
    detail.add_argument('--use-ner', type='bool', default=True,
                        help='Whether to use ner features')
    detail.add_argument('--use-lemma', type='bool', default=True,
                        help='Whether to use lemma features')
    detail.add_argument('--use-tf', type='bool', default=True,
                        help='Whether to use term frequency features')

    # Optimization details
    optim = parser.add_argument_group('DrQA Reader Optimization')
    optim.add_argument('--dropout-emb', type=float, default=0.4,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout-rnn', type=float, default=0.4,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout-rnn-output', type='bool', default=True,
                       help='Whether to dropout the RNN output')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for SGD only')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune-partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--rnn-padding', type='bool', default=False,
                       help='Explicitly account for padding in RNN encoding')
    optim.add_argument('--max-len', type=int, default=15,
                       help='The max span allowed during decoding')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER
    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimation, but leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data processing/loading helpers."""

import numpy as np
import logging
import unicodedata

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class ReaderDataset(Dataset):

    def __init__(self, examples, model, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model, self.single_answer)

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.
    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).
    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).
    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

from torch.autograd import Variable
from .config import override_model_args
from .rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)


class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).
        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).
        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.
        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

# --------------------------------------------------------------------------
#Learning
# --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.
        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return DocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.netowork = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader predictor"""

import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from .vector import vectorize, batchify
from .model import DocReader
from . import DEFAULTS, utils
from .. import tokenizers

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenize + annotate
# ------------------------------------------------------------------------------

PROCESS_TOK = None


def init(tokenizer_class, annotators):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(annotators=annotators)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Predictor class.
# ------------------------------------------------------------------------------


class Predictor(object):
    """Load a pretrained DocReader model and predict inputs on the fly."""

    def __init__(self, model=None, tokenizer=None,
                 embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file
            tokenizer: option string to select tokenizer class
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = DocReader.load(model or DEFAULTS['model'])

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.model.expand_dictionary(words)
            self.model.load_embeddings(added, embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = tokenizers.get_annotators_for_model(self.model)
        if not tokenizer:
            tokenizer_class = DEFAULTS['tokenizer']
        else:
            tokenizer_class = tokenizers.get_class(tokenizer)

        if num_workers is None or num_workers > 0:
            self.workers = ProcessPool(
                num_workers,
                initializer=init,
                initargs=(tokenizer_class, annotators),
            )
        else:
            self.workers = None
            self.tokenizer = tokenizer_class(annotators=annotators)

    def predict(self, document, question, candidates=None, top_n=1):
        """Predict a single document - question pair."""
        results = self.predict_batch([(document, question, candidates,)], top_n)
        return results[0]

    def predict_batch(self, batch, top_n=1):
        """Predict a batch of document - question pairs."""
        documents, questions, candidates = [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if any(candidates) else None

        # Tokenize the inputs, perhaps multi-processed.
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            d_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            d_tokens = list(d_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            d_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'qlemma': q_tokens[i].lemmas(),
                'document': d_tokens[i].words(),
                'lemma': d_tokens[i].lemmas(),
                'pos': d_tokens[i].pos(),
                'ner': d_tokens[i].entities(),
            })

        # Stick document tokens in candidates for decoding
        if candidates:
            candidates = [{'input': d_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(e, self.model) for e in examples])
        s, e, score = self.model.predict(batch_exs, candidates, top_n)

        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = d_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                predictions.append((span, score[i][j]))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA reader utilities."""

import json
import time
import logging
import string
import regex as re

from collections import Counter
from .data import Dictionary

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(args, filename, skip_no_answer=False):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]

    return examples


def load_text(filename):
    """Load the paragraphs only of a SQuAD dataset. Store as qid -> text."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    texts = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                texts[qa['id']] = paragraph['context']
    return texts


def load_answers(filename):
    """Load the answers only of a SQuAD dataset. Store as qid -> [answers]."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    ans = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])
    return words


def build_word_dict(args, examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)


def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Exact match features
    if args.use_in_question:
        _insert('in_question')
        _insert('in_question_uncased')
        if args.use_lemma:
            _insert('in_question_lemma')

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    return feature_dict


# ------------------------------------------------------------------------------
# Evaluation. Follows official evalutation script for v1.1 of the SQuAD dataset.
# ------------------------------------------------------------------------------


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        logger.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[3] for ex in batch])
            y_e = torch.cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Documents, in a sqlite database."""

import sqlite3
from . import utils
from . import DEFAULTS


class DocDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path or DEFAULTS['db_path']
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.
        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various retriever utilities."""

import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32


# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.
    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .corenlp_tokenizer import CoreNLPTokenizer
from .regexp_tokenizer import RegexpTokenizer
from .simple_tokenizer import SimpleTokenizer

# Spacy is optional
try:
    from .spacy_tokenizer import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'regexp':
        return RegexpTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Simple wrapper around the Stanford CoreNLP pipeline.
Serves commands to a java subprocess running the jar. Requires java 8.
"""

import copy
import json
import pexpect

from .tokenizer import Tokens, Tokenizer
from . import DEFAULTS


class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = (kwargs.get('classpath') or
                          DEFAULTS['corenlp_classpath'])
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None)
            ))
        return Tokens(data, self.annotators)
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Regex based tokenizer that emulates the Stanford/NLTK PTB tokenizers.
However it is purely in Python, supports robust untokenization, unicode,
and requires minimal dependencies.
"""

import regex
import logging
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class RegexpTokenizer(Tokenizer):
    DIGIT = r'\p{Nd}+([:\.\,]\p{Nd}+)*'
    TITLE = (r'(dr|esq|hon|jr|mr|mrs|ms|prof|rev|sr|st|rt|messrs|mmes|msgr)'
             r'\.(?=\p{Z})')
    ABBRV = r'([\p{L}]\.){2,}(?=\p{Z}|$)'
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]++'
    HYPHEN = r'{A}([-\u058A\u2010\u2011]{A})+'.format(A=ALPHA_NUM)
    NEGATION = r"((?!n't)[\p{L}\p{N}\p{M}])++(?=n't)|n't"
    CONTRACTION1 = r"can(?=not\b)"
    CONTRACTION2 = r"'([tsdm]|re|ll|ve)\b"
    START_DQUOTE = r'(?<=[\p{Z}\(\[{<]|^)(``|["\u0093\u201C\u00AB])(?!\p{Z})'
    START_SQUOTE = r'(?<=[\p{Z}\(\[{<]|^)[\'\u0091\u2018\u201B\u2039](?!\p{Z})'
    END_DQUOTE = r'(?<!\p{Z})(\'\'|["\u0094\u201D\u00BB])'
    END_SQUOTE = r'(?<!\p{Z})[\'\u0092\u2019\u203A]'
    DASH = r'--|[\u0096\u0097\u2013\u2014\u2015]'
    ELLIPSES = r'\.\.\.|\u2026'
    PUNCT = r'\p{P}'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
            substitutions: if true, normalizes some token types (e.g. quotes).
        """
        self._regexp = regex.compile(
            '(?P<digit>%s)|(?P<title>%s)|(?P<abbr>%s)|(?P<neg>%s)|(?P<hyph>%s)|'
            '(?P<contr1>%s)|(?P<alphanum>%s)|(?P<contr2>%s)|(?P<sdquote>%s)|'
            '(?P<edquote>%s)|(?P<ssquote>%s)|(?P<esquote>%s)|(?P<dash>%s)|'
            '(?<ellipses>%s)|(?P<punct>%s)|(?P<nonws>%s)' %
            (self.DIGIT, self.TITLE, self.ABBRV, self.NEGATION, self.HYPHEN,
             self.CONTRACTION1, self.ALPHA_NUM, self.CONTRACTION2,
             self.START_DQUOTE, self.END_DQUOTE, self.START_SQUOTE,
             self.END_SQUOTE, self.DASH, self.ELLIPSES, self.PUNCT,
             self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()
        self.substitutions = kwargs.get('substitutions', True)

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Make normalizations for special token types
            if self.substitutions:
                groups = matches[i].groupdict()
                if groups['sdquote']:
                    token = "``"
                elif groups['edquote']:
                    token = "''"
                elif groups['ssquote']:
                    token = "`"
                elif groups['esquote']:
                    token = "'"
                elif groups['dash']:
                    token = '--'
                elif groups['ellipses']:
                    token = '...'

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)
        #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens.
"""

import regex
import logging
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)

!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Tokenizer that is backed by spaCy (spacy.io).
Requires spaCy package and the spaCy english model.
"""

import spacy
import copy
from .tokenizer import Tokens, Tokenizer


class SpacyTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base tokenizer/tokens classes and utilities."""

import copy


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to convert the default SQuAD dataset to the format:
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = [a['text'] for a in qa['answers']]
                f.write(json.dumps({'question': question, 'answer': answer}))
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to convert the default WebQuestions dataset to the format:
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'
"""

import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for ex in dataset:
        question = ex['utterance']
        answer = ex['targetValue']
        answer = re.findall(
            r'(?<=\(description )(.+?)(?=\) \(description|\)\)$)', answer
        )
        answer = [a.replace('"', '') for a in answer]
        f.write(json.dumps({'question': question, 'answer': answer}))
        f.write('\n')
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to visually inspect generated data."""

import argparse
import json
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

with open(args.file) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        question = ' '.join(data['question'])
        start, end = data['answers'][0]
        doc = data['document']
        pre = ' '.join(doc[:start])
        ans = colored(' '.join(doc[start: end + 1]), 'red', attrs=['bold'])
        post = ' '.join(doc[end + 1:])
        print('-' * 50)
        print('Question: %s' % question)
        print('')
        print('Document: %s' % (' '.join([pre, ans, post])))
        input()
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to generate distantly supervised training data.
Using Wikipedia and available QA datasets, we search for a paragraph
that can be used as a supporting context.
"""

import argparse
import uuid
import heapq
import logging
import regex as re
import os
import json
import random

from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize

from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

from drqa import tokenizers
from drqa import retriever
from drqa.retriever import utils

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Fetch text, tokenize + annotate
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class=None, db_opts=None):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

    # optionally open a db connection
    if db_class:
        PROCESS_DB = db_class(**db_opts)
        Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def nltk_entity_groups(text):
    """Return all contiguous NER tagged chunks by NLTK."""
    parse_tree = ne_chunk(pos_tag(word_tokenize(text)))
    ner_chunks = [' '.join([l[0] for l in t.leaves()])
                  for t in parse_tree.subtrees() if t.label() != 'S']
    return ner_chunks


# ------------------------------------------------------------------------------
# Find answer candidates.
# ------------------------------------------------------------------------------


def find_answer(paragraph, q_tokens, answer, opts):
    """Return the best matching answer offsets from a paragraph.
    The paragraph is skipped if:
    * It is too long or short.
    * It doesn't contain the answer at all.
    * It doesn't contain named entities found in the question.
    * The answer context match score is too low.
      - This is the unigram + bigram overlap within +/- window_sz.
    """
    # Length check
    if len(paragraph) > opts['char_max'] or len(paragraph) < opts['char_min']:
        return

    # Answer check
    if opts['regex']:
        # Add group around the whole answer
        answer = '(%s)' % answer[0]
        ans_regex = re.compile(answer, flags=re.IGNORECASE + re.UNICODE)
        answers = ans_regex.findall(paragraph)
        answers = {a[0] if isinstance(a, tuple) else a for a in answers}
        answers = {a.strip() for a in answers if len(a.strip()) > 0}
    else:
        answers = {a for a in answer if a in paragraph}
    if len(answers) == 0:
        return

    # Entity check. Default tokenizer + NLTK to minimize falling through cracks
    q_tokens, q_nltk_ner = q_tokens
    for ne in q_tokens.entity_groups():
        if ne[0] not in paragraph:
            return
    for ne in q_nltk_ner:
        if ne not in paragraph:
            return

    # Search...
    p_tokens = tokenize_text(paragraph)
    p_words = p_tokens.words(uncased=True)
    q_grams = Counter(q_tokens.ngrams(
        n=2, uncased=True, filter_fn=utils.filter_ngram
    ))

    best_score = 0
    best_ex = None
    for ans in answers:
        try:
            a_words = tokenize_text(ans).words(uncased=True)
        except RuntimeError:
            logger.warn('Failed to tokenize answer: %s' % ans)
            continue
        for idx in range(len(p_words)):
            if p_words[idx:idx + len(a_words)] == a_words:
                # Overlap check
                w_s = max(idx - opts['window_sz'], 0)
                w_e = min(idx + opts['window_sz'] + len(a_words), len(p_words))
                w_tokens = p_tokens.slice(w_s, w_e)
                w_grams = Counter(w_tokens.ngrams(
                    n=2, uncased=True, filter_fn=utils.filter_ngram
                ))
                score = sum((w_grams & q_grams).values())
                if score > best_score:
                    # Success! Set new score + formatted example
                    best_score = score
                    best_ex = {
                        'id': uuid.uuid4().hex,
                        'question': q_tokens.words(),
                        'document': p_tokens.words(),
                        'offsets': p_tokens.offsets(),
                        'answers': [(idx, idx + len(a_words) - 1)],
                        'qlemma': q_tokens.lemmas(),
                        'lemma': p_tokens.lemmas(),
                        'pos': p_tokens.pos(),
                        'ner': p_tokens.entities(),
                    }
    if best_score >= opts['match_threshold']:
        return best_score, best_ex


def search_docs(inputs, max_ex=5, opts=None):
    """Given a set of document ids (returned by ranking for a question), search
    for top N best matching (by heuristic) paragraphs that contain the answer.
    """
    if not opts:
        raise RuntimeError('Options dict must be supplied.')

    doc_ids, q_tokens, answer = inputs
    examples = []
    for i, doc_id in enumerate(doc_ids):
        for j, paragraph in enumerate(re.split(r'\n+', fetch_text(doc_id))):
            found = find_answer(paragraph, q_tokens, answer, opts)
            if found:
                # Reverse ranking, giving priority to early docs + paragraphs
                score = (found[0], -i, -j, random.random())
                if len(examples) < max_ex:
                    heapq.heappush(examples, (score, found[1]))
                else:
                    heapq.heappushpop(examples, (score, found[1]))
    return [e[1] for e in examples]


def process(questions, answers, outfile, opts):
    """Generate examples for all questions."""
    logger.info('Processing %d question answer pairs...' % len(questions))
    logger.info('Will save to %s.dstrain and %s.dsdev' % (outfile, outfile))

    # Load ranker
    ranker = opts['ranker_class'](strict=False)
    logger.info('Ranking documents (top %d per question)...' % opts['n_docs'])
    ranked = ranker.batch_closest_docs(questions, k=opts['n_docs'])
    ranked = [r[0] for r in ranked]

    # Start pool of tokenizers with ner enabled
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {'annotators': {'ner'}}))

    logger.info('Pre-tokenizing questions...')
    q_tokens = workers.map(tokenize_text, questions)
    q_ner = workers.map(nltk_entity_groups, questions)
    q_tokens = list(zip(q_tokens, q_ner))
    workers.close()
    workers.join()

    # Start pool of simple tokenizers + db connections
    workers = Pool(opts['workers'], initializer=init,
                   initargs=(opts['tokenizer_class'], {},
                             opts['db_class'], {}))

    logger.info('Searching documents...')
    cnt = 0
    inputs = [(ranked[i], q_tokens[i], answers[i]) for i in range(len(ranked))]
    search_fn = partial(search_docs, max_ex=opts['max_ex'], opts=opts['search'])
    with open(outfile + '.dstrain', 'w') as f_train, \
         open(outfile + '.dsdev', 'w') as f_dev:
        for res in workers.imap_unordered(search_fn, inputs):
            for ex in res:
                cnt += 1
                f = f_dev if random.random() < opts['dev_split'] else f_train
                f.write(json.dumps(ex))
                f.write('\n')
                if cnt % 1000 == 0:
                    logging.info('%d results so far...' % cnt)
    workers.close()
    workers.join()
    logging.info('Finished. Total = %d' % cnt)


# ------------------------------------------------------------------------------
# Main & commandline options
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('data_name', type=str, help='Dataset name')
    parser.add_argument('out_dir', type=str, help='Output directory')

    dataset = parser.add_argument_group('Dataset')
    dataset.add_argument('--regex', action='store_true',
                         help='Flag if answers are expressed as regexps')
    dataset.add_argument('--dev-split', type=float, default=0,
                         help='Hold out for ds dev set (0.X)')

    search = parser.add_argument_group('Search Heuristic')
    search.add_argument('--match-threshold', type=int, default=1,
                        help='Minimum context overlap with question')
    search.add_argument('--char-max', type=int, default=1500,
                        help='Maximum allowed context length')
    search.add_argument('--char-min', type=int, default=25,
                        help='Minimum allowed context length')
    search.add_argument('--window-sz', type=int, default=20,
                        help='Use context on +/- window_sz for overlap measure')

    general = parser.add_argument_group('General')
    general.add_argument('--max-ex', type=int, default=5,
                         help='Maximum matches generated per question')
    general.add_argument('--n-docs', type=int, default=5,
                         help='Number of docs retrieved per question')
    general.add_argument('--tokenizer', type=str, default='corenlp')
    general.add_argument('--ranker', type=str, default='tfidf')
    general.add_argument('--db', type=str, default='sqlite')
    general.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    # Logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Read dataset
    dataset = os.path.join(args.data_dir, args.data_name)
    questions = []
    answers = []
    for line in open(dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']

        # Make sure the regex compiles
        if args.regex:
            try:
                re.compile(answer[0])
            except BaseException:
                logger.warning('Regex failed to compile: %s' % answer)
                continue

        questions.append(question)
        answers.append(answer)

    # Get classes
    ranker_class = retriever.get_class(args.ranker)
    db_class = retriever.get_class(args.db)
    tokenizer_class = tokenizers.get_class(args.tokenizer)

    # Form options
    search_keys = ('regex', 'match_threshold', 'char_max',
                   'char_min', 'window_sz')
    opts = {
        'ranker_class': retriever.get_class(args.ranker),
        'tokenizer_class': tokenizers.get_class(args.tokenizer),
        'db_class': retriever.get_class(args.db),
        'search': {k: vars(args)[k] for k in search_keys},
    }
    opts.update(vars(args))

    # Process!
    outname = os.path.splitext(args.data_name)[0]
    outfile = os.path.join(args.out_dir, outname)
    process(questions, answers, outfile, opts

    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Follows official evaluation script for v1.1 of the SQuAD dataset."""

import argparse
import json
from drqa.retriever.utils import normalize
from drqa.reader.utils import (
    exact_match_score,
    regex_match_score,
    metric_max_over_ground_truths
)


def evaluate(dataset_file, prediction_file, regex=False):
    print('-' * 50)
    print('Dataset: %s' % dataset_file)
    print('Predictions: %s' % prediction_file)

    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    predictions = []
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)
            prediction = normalize(data[0]['span'])
            predictions.append(prediction)

    exact_match = 0
    for i in range(len(predictions)):
        match_fn = regex_match_score if regex else exact_match_score
        exact_match += metric_max_over_ground_truths(
            match_fn, predictions[i], answers[i]
        )
    total = len(predictions)
    exact_match = 100.0 * exact_match / total
    print({'exact_match': exact_match})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('predictions', type=str)
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()
    evaluate(args.dataset, args.predictions, args.regex)
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full DrQA pipeline."""

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help='Path to trained Document Reader model')
parser.add_argument('--retriever-model', type=str, default=None,
                    help='Path to Document Retriever model (tfidf)')
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to "
                          "use (e.g. 'corenlp')"))
parser.add_argument('--candidate-file', type=str, default=None,
                    help=("List of candidates to restrict predictions to, "
                          "one candidate per line"))
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.candidate_file:
    logger.info('Loading candidates from %s' % args.candidate_file)
    candidates = set()
    with open(args.candidate_file) as f:
        for line in f:
            line = utils.normalize(line.strip()).lower()
            candidates.add(line)
    logger.info('Loaded %d candidates.' % len(candidates))
else:
    candidates = None

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    cuda=args.cuda,
    fixed_candidates=candidates,
    reader_model=args.reader_model,
    ranker_config={'options': {'tfidf_path': args.retriever_model}},
    db_config={'options': {'db_path': args.doc_db}},
    tokenizer=args.tokenizer
)


# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(question, candidates=None, top_n=1, n_docs=5):
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'],
                       '%.5g' % p['span_score'],
                       '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')


banner = """
Interactive DrQA
>> process(question, candidates=None, top_n=1, n_docs=5)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full DrQA retriever-reader pipeline."""

import torch
import os
import time
import json
import argparse
import logging

from drqa import pipeline
from drqa.retriever import utils


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--out-dir', type=str, default='/tmp',
                    help=("Directory to write prediction file to "
                          "(<dataset>-<model>-pipeline.preds)"))
parser.add_argument('--reader-model', type=str, default=None,
                    help="Path to trained Document Reader model")
parser.add_argument('--retriever-model', type=str, default=None,
                    help="Path to Document Retriever model (tfidf)")
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=("Expand dictionary to use all pretrained "
                          "embeddings in this file"))
parser.add_argument('--candidate-file', type=str, default=None,
                    help=("List of candidates to restrict predictions to, "
                          "one candidate per line"))
parser.add_argument('--n-docs', type=int, default=5,
                    help="Number of docs to retrieve per query")
parser.add_argument('--top-n', type=int, default=1,
                    help="Number of predictions to make per query")
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
parser.add_argument('--parallel', action='store_true',
                    help='Use data parallel (split across gpus)')
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Document paragraph batching size')
parser.add_argument('--predict-batch-size', type=int, default=1000,
                    help='Question batching size')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.candidate_file:
    logger.info('Loading candidates from %s' % args.candidate_file)
    candidates = set()
    with open(args.candidate_file) as f:
        for line in f:
            line = utils.normalize(line.strip()).lower()
            candidates.add(line)
    logger.info('Loaded %d candidates.' % len(candidates))
else:
    candidates = None

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    reader_model=args.reader_model,
    fixed_candidates=candidates,
    embedding_file=args.embedding_file,
    tokenizer=args.tokenizer,
    batch_size=args.batch_size,
    cuda=args.cuda,
    data_parallel=args.parallel,
    ranker_config={'options': {'tfidf_path': args.retriever_model,
                               'strict': False}},
    db_config={'options': {'db_path': args.doc_db}},
    num_workers=args.num_workers,
)


# ------------------------------------------------------------------------------
# Read in dataset and make predictions
# ------------------------------------------------------------------------------


logger.info('Loading queries from %s' % args.dataset)
queries = []
for line in open(args.dataset):
    data = json.loads(line)
    queries.append(data['question'])

model = os.path.splitext(os.path.basename(args.reader_model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '-full.preds')

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    batches = [queries[i: i + args.predict_batch_size]
               for i in range(0, len(queries), args.predict_batch_size)]
    for i, batch in enumerate(batches):
        logger.info(
            '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
        )
        predictions = DrQA.process_batch(
            batch,
            n_docs=args.n_docs,
            top_n=args.top_n,
        )
        for p in predictions:
            f.write(json.dumps(p) + '\n')

logger.info('Total time: %.2f' % (time.time() - t0))
# Document Reader

## Preprocessing

`preprocess.py` takes a SQuAD-formatted dataset and outputs a preprocessed, training-ready file. Specifically, it handles tokenization, mapping character offsets to token offsets, and any additional featurization such as lemmatization, part-of-speech tagging, and named entity recognition.

To preprocess SQuAD (assuming both input and output files are in `data/datasets`):

```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-train
```
```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-dev
```
- _You need to have [SQuAD](../../README.md#qa-datasets) train-v1.1.json and dev-v1.1.json in data/datasets (here renamed as SQuAD-v1.1-<train/dev>.json)_

## Training

`train.py` is the main train script for the Document Reader.

To get started with training a model on SQuAD with our best hyper parameters:

```bash
python scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000
```
- _You need to have the [glove embeddings](#note-on-word-embeddings) downloaded to data/embeddings/glove.840B.300d.txt._
- _You need to have done the preprocessing above._

The training has many options that you can tune:

```
Environment:
--no-cuda           Train on CPU, even if GPUs are available. (default: False)
--gpu               Run on a specific GPU (default: -1)
--data-workers      Number of subprocesses for data loading (default: 5)
--parallel          Use DataParallel on all available GPUs (default: False)
--random-seed       Random seed for all numpy/torch/cuda operations (for reproducibility).
--num-epochs        Train data iterations.
--batch-size        Batch size for training.
--test-batch-size   Batch size during validation/testing.
Filesystem:
--model-dir         Directory for saved models/checkpoints/logs (default: /tmp/drqa-models).
--model-name        Unique model identifier (.mdl, .txt, .checkpoint) (default: <generated uuid>).
--data-dir          Directory of training/validation data (default: data/datasets).
--train-file        Preprocessed train file (default: SQuAD-v1.1-train-processed-corenlp.txt).
--dev-file          Preprocessed dev file (default: SQuAD-v1.1-dev-processed-corenlp.txt).
--dev-json          Unprocessed dev file to run validation while training on (used to get original text for getting spans and answer texts) (default: SQuAD-v1.1-dev.json).
--embed-dir         Directory of pre-trained embedding files (default: data/embeddings).
--embedding-file    Space-separated pretrained embeddings file (default: None).
Saving/Loading:
--checkpoint        Save model + optimizer state after each epoch (default: False).
--pretrained        Path to a pretrained model to warm-start with (default: <empty>).
--expand-dictionary Expand dictionary of pretrained (--pretrained) model to include training/dev words of new data (default: False).
Preprocessing:
--uncased-question  Question words will be lower-cased (default: False).
--uncased-doc       Document words will be lower-cased (default: False).
--restrict-vocab    Only use pre-trained words in embedding_file (default: True).
General:
--official-eval     Validate with official SQuAD eval (default: True).
--valid-metric      The evaluation metric used for model selection (default: f1).
--display-iter      Log state after every <display_iter> epochs (default: 25).
--sort-by-len       Sort batches by length for speed (default: True).
DrQA Reader Model Architecture:
--model-type        Model architecture type (default: rnn).
--embedding-dim     Embedding size if embedding_file is not given (default: 300).
--hidden-size       Hidden size of RNN units (default: 128).
--doc-layers        Number of encoding layers for document (default: 3).
--question-layers   Number of encoding layers for question (default: 3).
--rnn-type          RNN type: LSTM, GRU, or RNN (default: lstm).
DrQA Reader Model Details:
--concat-rnn-layers Combine hidden states from each encoding layer (default: True).
--question-merge    The way of computing the question representation (default: self_attn).
--use-qemb          Whether to use weighted question embeddings (default: True).
--use-in-question   Whether to use in_question_* (cased, uncased, lemma) features (default: True).
--use-pos           Whether to use pos features (default: True).
--use-ner           Whether to use ner features (default: True).
--use-lemma         Whether to use lemma features (default: True).
--use-tf            Whether to use term frequency features (default: True).
DrQA Reader Optimization:
--dropout-emb           Dropout rate for word embeddings (default: 0.4).
--dropout-rnn           Dropout rate for RNN states (default: 0.4).
--dropout-rnn-output    Whether to dropout the RNN output (default: True).
--optimizer             Optimizer: sgd or adamax (default: adamax).
--learning-rate         Learning rate for SGD only (default: 0.1).
--grad-clipping         Gradient clipping (default: 10).
--weight-decay          Weight decay factor (default: 0).
--momentum              Momentum factor (default: 0).
--fix-embeddings        Keep word embeddings fixed (use pretrained) (default: True).
--tune-partial          Backprop through only the top N question words (default: 0).
--rnn-padding           Explicitly account for padding (and skip it) in RNN encoding (default: False).
--max-len MAX_LEN       The max span allowed during decoding (default: 15).
```

### Note on Word Embeddings

Using pre-trained word embeddings is very important for performance. The models we provide were trained with cased GloVe embeddings trained on Common Crawl, however we have also found that other embeddings such as FastText do quite well.

We suggest downloading the embeddings files and storing them under `data/embeddings/<file>.txt` (this is the default for `--embedding-dir`). The code expects space separated plain text files (\<token\> \<d1\> ... \<dN\>).

- [GloVe: Common Crawl (cased)](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
- [FastText: Wikipedia (uncased)](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec)

## Predicting

`predict.py` uses a trained Document Reader model to make predictions for an input dataset.

Required arguments:
```
dataset               SQuAD-like dataset to evaluate on (format B).
```

Optional arguments:
```
--model             Path to model to use.
--embedding-file    Expand dictionary to use all pretrained embeddings in this file.
--out-dir           Directory to write prediction file to (<dataset>-<model>.preds).
--tokenizer         String option specifying tokenizer type to use (e.g. 'corenlp').
--num-workers       Number of CPU processes (for tokenizing, etc).
--no-cuda           Use CPU only.
--gpu               Specify GPU device id to use.
--batch-size        Example batching size (Reduce in case of OOM).
--top-n             Store top N predicted spans per example.
--official          Only store single top span instead of top N list. (The SQuAD eval script takes a dict of qid: span).
```

Note: The CoreNLP NER annotator is not fully deterministic (depends on the order examples are processed). Predictions may fluctuate very slightly between runs if `num-workers` > 1 and the model was trained with `use-ner` on.

## Interactive

The Document Reader can also be used interactively (like the [full pipeline](../../README.md#quick-start-demo)).

```bash
python scripts/reader/interactive.py --model /path/to/model
```

```
>>> text = "Mary had a little lamb, whose fleece was white as snow. And everywhere that Mary went the lamb was sure to go.""
>>> question = "What color is Mary's lamb?"
>>> process(text, question)
+------+-------+---------+
| Rank |  Span |  Score  |
+------+-------+---------+
|  1   | white | 0.78002 |
+------+-------+---------+
```
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to run the DrQA reader model interactively."""

import torch
import code
import argparse
import logging
import prettytable
import time

from drqa.reader import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(args.model, args.tokenizer, num_workers=0)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(document, question, candidates=None, top_n=1):
    t0 = time.time()
    predictions = predictor.predict(document, question, candidates, top_n)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    print(table)
    print('Time: %.4f' % (time.time() - t0))


banner = """
DrQA Interactive Document Reader Module
>> process(document, question, candidates=None, top_n=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json

from tqdm import tqdm
from drqa.reader import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default=None,
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='/tmp',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true',
                    help='Only store single top span instead of top N list')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------


examples = []
qids = []
with open(args.dataset) as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qids.append(qa['id'])
                examples.append((context, qa['question']))

results = {}
for i in tqdm(range(0, len(examples), args.batch_size)):
    predictions = predictor.predict_batch(
        examples[i:i + args.batch_size], top_n=args.top_n
    )
    for j in range(len(predictions)):
        # Official eval expects just a qid --> span
        if args.official:
            results[qids[i + j]] = predictions[j][0][0]

        # Otherwise we store top N and scores for debugging.
        else:
            results[qids[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]

model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '.preds')

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    json.dump(results, f)

logger.info('Total time: %.2f' % (time.time() - t0))
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import argparse
import os
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from drqa import tokenizers

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='corenlp')
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main DrQA reader training script."""

import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging


from drqa.reader import utils, vector, config, data
from drqa.reader import DocReader
from drqa import DATA_DIR as DRQA_DATA

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = '/tmp/drqa-models/'
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-processed-corenlp.txt',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-corenlp.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError('No such file: %s' % args.dev_json)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    model = DocReader(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate_unofficial(args, data_loader, model, global_stats, mode):
    """Run one full unofficial validation.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = utils.Timer()
    start_acc = utils.AverageMeter()
    end_acc = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)
        target_s, target_e = ex[-3:-1]

        # We get metrics for independent start/end and joint start/end
        accuracies = eval_accuracies(pred_s, target_s, pred_e, target_e)
        start_acc.update(accuracies[0], batch_size)
        end_acc.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial: Epoch = %d | start = %.2f | ' %
                (mode, global_stats['epoch'], start_acc.avg) +
                'end = %.2f | exact = %.2f | examples = %d | ' %
                (end_acc.avg, exact_match.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'exact_match': exact_match.avg}


def validate_official(args, data_loader, model, global_stats,
                      offsets, texts, answers):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)

        for i in range(batch_size):
            s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
            e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
            prediction = texts[ex_id[i]][s_offset:e_offset]

            # Compute metrics
            ground_truths = answers[ex_id[i]]
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))

        examples += batch_size

    logger.info('dev valid official: Epoch = %d | EM = %.2f | ' %
                (global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    """An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e] for e in target_s]
        target_e = [[e] for e in target_e]

    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # If we are doing offician evals then we need to:
    # 1) Load the original text to retrieve spans from offsets.
    # 2) Load the (multiple) text answers for each question.
    if args.official_eval:
        dev_texts = utils.load_text(args.dev_json)
        dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
        dev_answers = utils.load_answers(args.dev_json)

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate unofficial (train)
        validate_unofficial(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)

# Document Retriever

## Storing the Documents

To efficiently store and access our documents, we store them in a sqlite database. The key is the `doc_id` and the value is the `text`.

To create a sqlite db from a corpus of documents, run:

```bash
python build_db.py /path/to/data /path/to/saved/db.db
```

Optional arguments:
```
--preprocess    File path to a python module that defines a `preprocess` function.
--num-workers   Number of CPU processes (for tokenizing, etc).
```

The data path can either be a path to a nested directory of files (such as what the [WikiExtractor](https://github.com/attardi/wikiextractor) script outputs) or a single file. Each file should consist of JSON-encoded documents that have `id` and `text` fields, one per line:

```python
{"id": "doc1", "text": "text of doc1"}
...
{"id": "docN", "text": "text of docN"}
```

`--preprocess /path/to/.py/file` is another optional argument that allows you to supply a python module that defines a `preprocess(doc_object)` function to filter/process documents before they are put in the db. See `prep_wikipedia.py` for an example.

## Building the TF-IDF N-grams

To build a TF-IDF weighted word-doc sparse matrix from the documents stored in the sqlite db, run:

```bash
python build_tfidf.py /path/to/doc/db /path/to/output/dir
```

Optional arguments:
```
--ngram         Use up to N-size n-grams (e.g. 2 = unigrams + bigrams). By default only ngrams without stopwords or punctuation are kept.
--hash-size     Number of buckets to use for hashing ngrams.
--tokenizer     String option specifying tokenizer type to use (e.g. 'corenlp').
--num-workers   Number of CPU processes (for tokenizing, etc).
```

The sparse matrix and its associated metadata will be saved to the output directory under `<db-name>-tfidf-ngram=<N>-hash=<N>-tokenizer=<T>.npz`.

## Interactive

The Document Retriever can also be used interactively (like the [full pipeline](../../README.md#quick-start-demo)).

```bash
python scripts/retriever/interactive.py --model /path/to/model
```

```
>>> process('question answering', k=5)
+------+-------------------------------+-----------+
| Rank |             Doc Id            | Doc Score |
+------+-------------------------------+-----------+
|  1   |       Question answering      |   327.89  |
|  2   |       Watson (computer)       |   217.26  |
|  3   |          Eric Nyberg          |   214.36  |
|  4   |   Social information seeking  |   212.63  |
|  5   | Language Computer Corporation |   184.64  |
+------+-------------------------------+-----------+
``` 
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((utils.normalize(doc['id']), doc['text']))
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.
    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers
    )
    #!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).
    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.
    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    retriever.utils.save_sparse_csr(filename, tfidf, metadata)
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.
    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return 1
    return 0


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=args.num_workers
    )
    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from drqa import retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Preprocess function to filter/prepare Wikipedia docs."""

import regex as re
from html.parser import HTMLParser

PARSER = HTMLParser()
BLACKLIST = set(['23443579', '52643645'])  # Conflicting disambig. pages


def preprocess(article):
    # Take out HTML escaping WikiExtractor didn't clean
    for k, v in article.items():
        article[k] = PARSER.unescape(v)

    # Filter some disambiguation pages not caught by the WikiExtractor
    if article['id'] in BLACKLIST:
        return None
    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None

    # Take out List/Index/Outline pages (mostly links)
    if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                article['title']):
        return None

    # Return doc with `id` set to `title`
    return {'id': article['title'], 'text': article['text']}
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='drqa',
    version='0.1.0',
    description='Reading Wikipedia to Answer Open-Domain Questions',
    long_description=readme,
    license=license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)