"""
program to test end task performance as a function of number of training steps the BERT language model was trained for
(Research question 1)
"""

import numpy 
import pandas as pd
import os
from os import path
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#fast ai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

#transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, BertConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import BertForMaskedLM
from transformers import AdamW

import fastai
import transformers
import argparse


class TransformerBaseTokenizer(BaseTokenizer):
  """
    Wrapper around PreTrainedTokenizer compatible with fastai
  """
  def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type='bert', **kwargs):
    self._pretrained_tokenizer = pretrained_tokenizer
    self.max_seq_len = pretrained_tokenizer.max_len
    self.model_type = model_type

  def __call__(self, *args, **kwargs):
    return self

  def tokenizer(self, t:str) -> List[str]:
    CLS = self._pretrained_tokenizer.cls_token
    SEP = self._pretrained_tokenizer.sep_token
    tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len-2]
    tokens = [CLS] + tokens + [SEP]
    return tokens

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer

    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

    #these two functions allow export and load_learnerto work properly in TransformersVocab
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})


class CustomTransformerModel(nn.Module):
  def __init__(self, transformer_model: PreTrainedModel, transformer_tokenizer : BertTokenizer):
    super(CustomTransformerModel, self).__init__()
    self.transformer = transformer_model
    self.transformer_tokenizer = transformer_tokenizer

  def forward(self, input_ids, attention_mask=None):
    #mask to avoid performing attention on padding
    attention_mask = (input_ids!=self.transformer_tokenizer.pad_token_id).type(input_ids.type())
    logits = self.transformer(input_ids, attention_mask=attention_mask)[0]
    return logits

def parse_option():
   parser = argparse.ArgumentParser('argument for training') 
   parser.add_argument('--models_path', type=str, help='path to all the different domain adapted models')
   parser.add_argument('--dataset', type=str, help='path to dataset')
   parser.add_argument("--bs", type=int, help='batch size')
   parser.add_argument('--seed', type=int, default=42, help="seed value")
   parser.add_argument('--use_domain_adapted', action='store_true',  help='if true, use the domain adapted model given by --model_path') 
   parser.add_argument('--topic', type=str, help='topic')
   parser.add_argument('--pretrained_model_name', type=str, default='bert-base-german-cased')
   opt = parser.parse_args()

   return opt


def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def prepare_data(args):
	X = pd.read_csv(args.dataset, sep=',', escapechar='\\', engine='python', names=['title', 'url', 'comment_text', 'date', 'label'], error_bad_lines=False)

	#the first row contains the original headers, removing that line
	X = X[1:]

	#english seperators work better 
	X['title'] = "TITLE" + X['title']
	X['comment_text'] = " COMMENT " + X['comment_text']
	X['complete_text'] = X['title'] + X['comment_text']

	#considering text with the titles
	X = X.filter(['complete_text', 'label'], axis=1)
	X.label.replace('Offline', 0, inplace=True)
	X.label.replace('Online', 1, inplace=True)
	X.label.replace('To activate', 2, inplace=True)
	X.label.replace('On hold', 3, inplace=True)

	#considering only online and offline classification
	X = X[X.label < 2]
	X = X.astype({'label' : 'int64'})

	X = X.reset_index()
	X.drop(['index'], inplace=True, axis=1)
	X= X.reset_index()
	X.rename(mapper={'index': 'id'}, inplace=True, axis=1)
	return X

def generate_inferences(model, train_df , args, TOPIC):
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer
    config_class = BertConfig

    transformer_tokenizer = tokenizer_class.from_pretrained(args.pretrained_model_name)
    transformer_base_tokenizer = TransformerBaseTokenizer(pretrained_tokenizer=transformer_tokenizer, model_type='bert')
    #tokenizer is a class in fastai
    fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])


    transformer_vocab = TransformersVocab(tokenizer=transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
    tokenize_processor = TokenizeProcessor(tokenizer = fastai_tokenizer, include_bos=False, include_eos=False)
    transformer_processor=[tokenize_processor, numericalize_processor]


    databunch = (TextList.from_df(train_df, cols='complete_text',processor=transformer_processor)
                      .split_by_rand_pct(0.2, seed=args.seed)
                      .label_from_df(cols='label')
                      .databunch(bs=args.bs,pad_first=False, pad_idx=transformer_tokenizer.pad_token_id))

    config = config_class.from_pretrained(args.pretrained_model_name)
    config.num_labels =2
    transformer_model = model_class.from_pretrained(model, config=config)
    custom_transformer_model = CustomTransformerModel(transformer_model=transformer_model, transformer_tokenizer=transformer_tokenizer)


    CustomAdamW = partial(AdamW, correct_bias=False)

    learner = Learner(databunch,
                      custom_transformer_model,
                      opt_func = CustomAdamW,
                      metrics=[accuracy, error_rate, Precision(pos_label=0),
                          Recall(pos_label=0),Precision(pos_label=1), Recall(pos_label=1)])

    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))

    learner.unfreeze()
    print('lr: 2e-6')
    learner.fit(1, lr=2e-6)
    print('lr:6.31e-7')
    learner.fit(3, lr=6.31e-7)


def main():
    args = parse_option()
    seed_all(args.seed)
    TOPIC = args.topic 
    category = 'label'
    model_type = 'bert'

    #preparing dataset
    train_df = prepare_data(args)
    
    if args.use_domain_adapted: 
        print('using topic adapted pretrained model')
        dirs = [name for name in os.listdir(args.models_path) if os.path.isdir(os.path.join(args.models_path, name))]
        dirs.sort()
        for directory in dirs:
            model_path = os.path.join(args.models_path , '{}/pytorch_model.bin'.format(directory))
            generate_inferences(model_path, train_df, args, TOPIC)
    else:
        print('using vanilla pretrained model')
        generate_inferences(args.pretrained_model_name,train_df, args, TOPIC)


if __name__ == '__main__':
    main()
