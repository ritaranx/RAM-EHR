import torch
import torch.nn as nn
from transformers import *


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2, model_name_or_path = 'bert-base-uncased'):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.linear1 = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128))
        self.linear2 = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128))
        self.linear3 = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128))
        
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, length=256):
        # Encode input text
        all_hidden, pooler = self.bert(x)

        pooled_output = torch.mean(all_hidden, 1)
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict


class ClassificationBert(nn.Module):
    def __init__(self, model_name_or_path='bert-base-uncased', num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.model_name_or_path = model_name_or_path
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        # self.bert2 = AutoModel.from_pretrained(model_name_or_path)
        # self.bert3 = AutoModel.from_pretrained(model_name_or_path)
        self.linear1 = nn.Sequential(nn.Linear(768, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_labels))
        self.linear2 = nn.Sequential(nn.Linear(768, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_labels))
        self.linear3 = nn.Sequential(nn.Linear(768, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_labels))
        if 'BiomedBERT-large' in model_name_or_path:
            self.linear = nn.Sequential(nn.Linear(3*1024, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_labels))
        elif 'clinical-mobilebert' in model_name_or_path:
            self.linear = nn.Sequential(nn.Linear(3*512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, num_labels))
        else:
            self.linear = nn.Sequential(nn.Linear(3*768, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_labels))

    def forward(self, **x):
        # Encode input text
        inputs_a = {
            "input_ids": x['input_ids_a'], # : batch[0],
            "attention_mask": x['attention_mask_a'], #: batch[1],
            "token_type_ids": x['token_type_ids_a']  #: batch[2],
        }
        inputs_b = {
            "input_ids": x['input_ids_b'], # : batch[0],
            "attention_mask": x['attention_mask_b'], #: batch[1],
            "token_type_ids": x['token_type_ids_b']  #: batch[2],
        }
        inputs_c = {
            "input_ids": x['input_ids_c'], # : batch[0],
            "attention_mask": x['attention_mask_c'], #: batch[1],
            "token_type_ids": x['token_type_ids_c']  #: batch[2],
        }
        if 'distil' in self.model_name_or_path:
            del inputs_a['token_type_ids']
            del inputs_b['token_type_ids']
            del inputs_c['token_type_ids']
        outputs_a = self.bert(**inputs_a)
        outputs_b = self.bert(**inputs_b)
        outputs_c = self.bert(**inputs_c)

        outputs_a = torch.mean(outputs_a[0], 1)
        outputs_b = torch.mean(outputs_b[0], 1)
        outputs_c = torch.mean(outputs_c[0], 1)
        # print("all_hidden", len(all_hidden))
        # pooled_output = torch.cat([outputs_a[1], outputs_b[1], outputs_c[1]], dim = 1)
        # pooled_output = torch.cat([torch.mean(outputs_a[0], 1), torch.mean(outputs_b[0], 1), torch.mean(outputs_c[0], 1)], dim = 1)
        pooled_output = torch.cat([outputs_a, outputs_b, outputs_c], dim = 1)

        # assert 0
        # pooled_output = torch.mean(all_hidden, 1)
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)
        return predict