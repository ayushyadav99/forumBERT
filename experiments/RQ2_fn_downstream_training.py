"""
program to test end task performance as a function of number of downstream training examples
(Research question 2)
"""

import decimal
import os
from os import path
import pandas as pd
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#fast ai
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from fastai.tabular import *
from fastai.text import *
from fastai.metrics import accuracy


#transformers
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, BertConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, BertModel
from transformers import BertForMaskedLM
from transformers import AdamW
import argparse
import fastai
import transformers
print("fastai :", fastai.__version__)
print("transformers :", transformers.__version__)

def float_range(start, stop, step):
  while(start < stop):
    yield float(start)
    start += decimal.Decimal(step)

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


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
    PAD = self._pretrained_tokenizer.pad_token
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

class ConcatDataset(Dataset):
    def __init__(self, x1, x2, y): self.x1,self.x2,self.y = x1,x2,y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return (self.x1[i], self.x2[i]), self.y[i]

def my_collate(batch):    
    x,y = list(zip(*batch))
    x1,x2 = list(zip(*x))
    x1, y = pad_collate(list(zip(x1, y)), pad_idx=0, pad_first=False)
    x2, _ = pad_collate(list(zip(x2, y)), pad_idx=0, pad_first=False)
    return (x1, x2), y

class CustomTransformerModel(nn.Module):
  def __init__(self, siamese_transformer: PreTrainedModel, transformer_tokenizer : BertTokenizer, n_classes):
    super(CustomTransformerModel, self).__init__()
    #self.title_transformer = title_transformer
    
    ##### BOTH transformers are sharing weights ###	
    self.siamese_transformer = siamese_transformer
    
    self.transformer_tokenizer = transformer_tokenizer
    self.drop = nn.Dropout(p=0.5)
    self.h1 = nn.Linear(self.siamese_transformer.config.hidden_size*2, self.siamese_transformer.config.hidden_size)
    self.out = nn.Linear(self.siamese_transformer.config.hidden_size, n_classes)
  
  def forward(self, comment_input_ids, title_input_ids):
    #mask to avoid performing attention on padding
    comment_attention_mask = (comment_input_ids!=self.transformer_tokenizer.pad_token_id).type(comment_input_ids.type())
    title_attention_mask = (title_input_ids!=self.transformer_tokenizer.pad_token_id).type(title_input_ids.type())
    _, title_pooled_output = self.siamese_transformer(
        input_ids=title_input_ids,
        attention_mask=title_attention_mask)
    _, comment_pooled_output = self.siamese_transformer(
          input_ids=comment_input_ids,
          attention_mask=comment_attention_mask)
    pooled_output = torch.cat([comment_pooled_output, title_pooled_output], dim=1)
    output = self.drop(pooled_output)
    output = self.h1(output)
    output = self.drop(output)
    return self.out(output)

def prepare_data(TOPIC, train_size):
  #X = pd.read_csv('/raid/yadav/NDR/data/{}_for_lm.csv'.format(TOPIC), sep=',', escapechar='\\', engine='python', names=['title', 'url', 'comment_text', 'date', 'label'], error_bad_lines=False)
  validation_size = 50000
  X = pd.read_csv('/raid/yadav/NDR/data/ndrb-3576_filtered.csv'.format(TOPIC), sep=',', escapechar='\\', engine='python', names=['title', 'url', 'comment_text', 'date', 'label'], error_bad_lines=False)
  #X['validation'] = False
  #X = X.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), train_size//2), random_state=10))
  print("intial X shape: ", X.shape)
  X = X.sample(frac=1).reset_index(drop=True)
  valid_true = [True for _ in range(validation_size)]
  valid_false = [False for _ in range(len(X)-validation_size)]
  valid_false.extend(valid_true)
  X['validation'] = valid_false 
  print(X.label.value_counts())
  #val = pd.read_csv('/raid/yadav/NDR/data/{}_for_clf.csv'.format(TOPIC), sep=',', escapechar='\\', engine='python', names=['title', 'url', 'comment_text', 'date', 'label'], error_bad_lines=False)
  #val = val[1:]
  #val['validation'] = True
  #X = pd.concat([train, test]) 
  #print("validation data shape: ",val.shape)
  print("final X shape: ", X.shape)
  
  #considering text with the titles
  X = X.filter(['comment_text', 'title','label', 'validation'], axis=1)
  X.label.replace('Offline', 0, inplace=True)
  X.label.replace('Online', 1, inplace=True)
  X.label.replace('To activate', 2, inplace=True)
  X.label.replace('On hold', 3, inplace=True)
  X = X[X.label < 2]
  #X.rename(mapper={category: 'label'}, inplace=True, axis=1)
  
  X = X.astype({'label' : 'int64'})
  return X


def generate_inferences(X, model, args, config):
  model_class = BertModel
  tokenizer_class = BertTokenizer
  config_class = BertConfig
  model_type = 'bert'
  bs = args.bs
  pretrained_model_name = 'bert-base-german-cased'
  transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name, padding='max_length')
  transformer_base_tokenizer = TransformerBaseTokenizer(pretrained_tokenizer=transformer_tokenizer,
                              model_type='bert')
  #tokenizer is a class in fastai
  fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])
  
  transformer_vocab = TransformersVocab(tokenizer=transformer_tokenizer)
  numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
  tokenize_processor = TokenizeProcessor(tokenizer = fastai_tokenizer, include_bos=False, include_eos=False)
  transformer_processor=[tokenize_processor, numericalize_processor]
  
  
  pad_idx = transformer_tokenizer.pad_token_id
  
  
  tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
  #print(tokenizer.tokenize('Es freut mich, dich kennenzulernen.'))
  
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  comment_databunch = (TextList.from_df(X, cols=('comment_text'),processor=transformer_processor)
                .split_from_df(col='validation')
                .label_from_df(cols='label')
                .databunch(bs=bs,pad_first=False, pad_idx=pad_idx))
  title_databunch = (TextList.from_df(X, cols=('title'),processor=transformer_processor)
                .split_from_df(col='validation')
                .label_from_df(cols='label')
                .databunch(bs=bs,pad_first=False, pad_idx=pad_idx))
  
      
  train_ds = ConcatDataset(comment_databunch.train_ds.x, title_databunch.train_ds.x, comment_databunch.train_ds.y)
  valid_ds = ConcatDataset(comment_databunch.valid_ds.x, title_databunch.valid_ds.x, comment_databunch.valid_ds.y)
  
  
  train_sampler = SortishSampler(comment_databunch.train_ds.x, key=lambda t: len(comment_databunch.train_ds[t][0].data), bs=bs)
  valid_sampler = SortSampler(comment_databunch.valid_ds.x, key=lambda t: len(comment_databunch.valid_ds[t][0].data))
  #databunch = DataBunch.create(train_ds, valid_ds, num_workers=4)
  train_dl = DataLoader(train_ds, bs, sampler=train_sampler)
  valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)
  data = DataBunch(train_dl, valid_dl, device=defaults.device, collate_fn=my_collate)
  
  
  
  #config = config_class.from_pretrained('/raid/yadav/NDR/etc/checkpoint-12000/config.json')
  #config = config_class.from_pretrained('/raid/yadav/NDR/domain/culture/config.json')
  config = config_class.from_pretrained(config)
  config.num_labels =2
  
  #transformer_model = model_class.from_pretrained('/raid/yadav/NDR/etc/checkpoint-12000/pytorch_model.bin', config=config)
  #4th epoch model
  siamese_transformer_model = model_class.from_pretrained(model, config=config)
  #title_transformer_model = model_class.from_pretrained('/raid/yadav/NDR/domain/sport/pytorch_model.bin', config=config)
  #title_transformer_model = model_class.from_pretrained(model, config=config)
  #siamese_transformer_model = model_class.from_pretrained(pretrained_model_name, config=config)
  custom_transformer_model = CustomTransformerModel(siamese_transformer=siamese_transformer_model,transformer_tokenizer =transformer_tokenizer, n_classes=2)
  
  
  CustomAdamW = partial(AdamW, correct_bias=False)
  
  learner = Learner(data, 
                    custom_transformer_model, 
                    opt_func = CustomAdamW, 
                    metrics=[accuracy, error_rate, Precision(pos_label=0), Recall(pos_label=0), Precision(pos_label=1), Recall(pos_label=1)])
  
  # Show graph of learner stats and metrics after each epoch.
  learner.callbacks.append(ShowGraph(learner))
  
  print(learner.loss_func)
  learner.unfreeze()
  if (args.use_domain_adapted): 
    learner.fit(2, lr=2e-6)
    learner.fit(2, lr=6.31e-7)
  else:
    learner.fit(2, lr=2e-6)
    learner.fit(2, lr=6.31e-7)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
#    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--bs', type=int,default=8, help='batch size')
    parser.add_argument('--seed', type=int,default=4, help='seed')
    parser.add_argument('--topic', type=str, help='topic')
    parser.add_argument('--use_domain_adapted', action='store_true', help='if true, ise the domain adapted model')
    opt = parser.parse_args()

    return opt

def main():
  args = parse_option()
  TOPIC = args.topic 
  seed=args.seed
  bs=args.bs
  seed_all(seed)
  sizes = [int(2**x) for x in list(float_range(2, 17, '0.5'))]
  #sizes = [10] 
  #sizes = [1024, 8196, 92680] 
  if args.use_domain_adapted:
    #directory = 'ep13_30150'
    #model = '/raid/yadav/NDR/domain/{}/{}/pytorch_model.bin'.format(TOPIC, directory)
    model = '/raid/yadav/NDR/domain/{}/pytorch_model.bin'.format(TOPIC)
  else:
    model = 'bert-base-german-cased'
  for train_size in sizes:
    if (not(train_size > 9 and train_size < 80000)):
        continue 
    print(model)
    config = '/raid/yadav/NDR/domain/{}/config.json'.format(TOPIC)
    X = prepare_data(TOPIC, train_size)
    generate_inferences(X, model, args, config) 
  
if __name__ == "__main__":
    main()
