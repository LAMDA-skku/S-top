import pandas as pd 
import numpy as np
import random
import os 
import re
import json
import torch
import pickle
import matplotlib.pyplot as plt
from attrdict import AttrDict
from sklearn.utils import shuffle 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer, BertModel, AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification

class BertDataset(Dataset):
    def __init__(self, data_file):
        self.data = data_file
    
    def __len__(self):
        return len(self.data.label)
    
    def reset_index(self):
        self.data.reset_index(inplace=True, drop=True)
    
    def __getitem__(self, idx):
        '''
        return text, label
        '''
        self.reset_index()
        text = self.data.text[idx]
        label = self.data.label[idx]
        return text, label

class BertProcessor():
    def __init__(self, config, training_config, tokenizer, truncation=True):
        self.tokenizer = tokenizer 
        self.max_len = config.max_position_embeddings
        self.pad = training_config.pad
        self.batch_size = training_config.train_batch_size
        self.truncation = truncation
    
    def convert_data(self, data_file):
        context2 = None    # single sentence classification

        batch_encoding = self.tokenizer.batch_encode_plus(
            [(data_file[idx][0], context2) for idx in range(len(data_file))],   # text, 
            max_length = self.max_len,
            padding = self.pad,
            truncation = self.truncation
        )
        
        features = []
        for i in range(len(data_file)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            try:
                inputs['label'] = data_file[i][1] 
            except:
                inputs['label'] = 0 
            features.append(inputs)
        
        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset
    
    def convert_sentence(self, sent_list):   # 사용자 입력 문장 1개 -> 입력 형태 변환
        context2 = None 
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(sent_list, context2)], max_length=self.max_len, padding=self.pad, truncation=self.truncation
        )
        
        features = []
        inputs = {k: batch_encoding[k][0] for k in batch_encoding}
        inputs['label'] = 0 
        features.append(inputs)

        input_id = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        input_am = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        input_tts = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
        input_lb = torch.tensor([f['label'] for f in features], dtype=torch.long)
        dataset = TensorDataset(input_id, input_am, input_tts, input_lb)
        return dataset
    
    def shuffle_data(self, dataset, data_type):
        if data_type == 'train':
            return RandomSampler(dataset)
        elif data_type == 'eval' or data_type == 'test':
            return SequentialSampler(dataset)
        
    def load_data(self, dataset, sampler):
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
    
    
class BertRegressor(nn.Module):
    def __init__(self, config, model):
        super(BertRegressor, self).__init__()
        self.model = model
        self.linear = nn.Linear(config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.last_hidden_state[:, 0, :]
        x = self.linear(logits)
        x = self.relu(x)
        score = self.out(x)
        return score 
    
    
class BertTester():
    def __init__(self, training_config, model):
        self.training_config = training_config
        self.model = model

    def get_label(self, test_dataloader, test_type):
        '''
        test_type: 0  -> Test dataset 
        test_type: 1  -> Test sentence
        '''
        preds = []
        labels = []

        for batch in test_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.training_config.device) for t in batch)   # args.device: cuda 
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                outputs = self.model(**inputs)
                test_loss, logits = outputs[:2] 
                pred = logits.detach().cpu().numpy()
                if test_type == 0:
                    preds.extend(np.argmax(pred, axis=1))
                elif test_type == 1:
                    preds.append(np.argmax(pred))  
                            
            label = inputs["labels"].detach().cpu().numpy()
            labels.extend(label)
  
        return preds, labels 

    def get_score(self, test_dataloader, test_type):
        '''
        test_type: 0  -> Test dataset 
        test_type: 1  -> Test sentence
        '''
        preds = []
        labels = []

        for batch in test_dataloader:
            self.model.eval()   # self 안 붙이면 이상한 Output (BaseModelOutputWithPoolingAndCrossAttentions) 출력 
            batch = tuple(t for t in batch)   # args.device: cuda 
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = self.model(**inputs)
                if test_type == 0:
                    preds.extend(outputs.squeeze())
                elif test_type == 1:
                    preds.extend(outputs[0])            
            label = batch[3]
            labels.extend(label)
        return preds, labels 

def main():
    model_path = './model'
    config_path = './config'

    with open(os.path.join(config_path, 'training_config.json')) as f:
        training_config = AttrDict(json.load(f))
    training_config.pad = 'max_length'

    label = dict()
    label[0] = '우울'
    label[1] = '무기력'
    label[2] = '급격한 체중(식욕)변화'
    label[3] = '수면장애'
    label[4] = '정서불안'
    label[5] = '피로'
    label[6] = '과도한 죄책감 및 무가치함'
    label[7] = '인지기능저하'
    label[8] = '자살충동'
    label[9] = '일상'

    bws_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, 'bert-mini'), model_max_length=32)
    dsm_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, 'bert-mini'), model_max_length=32)
    bws_config = BertConfig.from_pretrained(os.path.join(model_path, 'bert-mini', 'bert_config.json'))
    dsm_config = BertConfig.from_pretrained(os.path.join(model_path, 'bert-mini', 'bert_config.json'), num_labels=10)
    bws_model = BertModel.from_pretrained(os.path.join(model_path, 'bert-mini'), config=bws_config)
    dsm_model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, 'bert-mini'), config=dsm_config)
    
    bws_config.max_position_embeddings = 32
    dsm_config.max_position_embeddings = 32
    
    bws_reg = BertRegressor(bws_config, bws_model)
    
    bws_model_name = os.path.join(model_path, 'bert_bws_mini.pt')
    dsm_model_name = os.path.join(model_path, 'bert_dsm_mini.pt')
    bws_reg.load_state_dict(torch.load(bws_model_name, map_location=torch.device('cpu')))
    dsm_model.load_state_dict(torch.load(dsm_model_name, map_location=torch.device('cpu')))
    
    test_processor = BertProcessor(bws_config, training_config, bws_tokenizer)
    bws_tester = BertTester(training_config, bws_reg)
    dsm_tester = BertTester(training_config, dsm_model)
    test_sent = "I'm very lonely" 
    test_data = test_processor.convert_sentence(test_sent)
    test_sampler = test_processor.shuffle_data(test_data, 'test')
    test_loader = test_processor.load_data(test_data, test_sampler)
    bws_score, _ = bws_tester.get_score(test_loader, 1)
    dsm_label, _ = dsm_tester.get_label(test_loader, 1)
    print(bws_score, dsm_label)

if __name__ == '__main__':
    main()