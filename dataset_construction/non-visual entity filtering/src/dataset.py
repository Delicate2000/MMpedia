import json
from transformers import AutoTokenizer

import transformers
from transformers import BertTokenizer, BertModel, DetrForObjectDetection,DetrConfig
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import ImageFile
from tqdm import tqdm
import time
from torchvision import transforms
import os

def get_dict():
    # 读数据
    with open("../dataset/wordnet/train_data.json", "r", encoding='utf-8') as f:
        train_data = json.load(f)
    with open("../dataset/wordnet/test_data.json", "r", encoding='utf-8') as f:
        test_data = json.load(f)
    with open("../dataset/wordnet/valid_data.json", "r", encoding='utf-8') as f:
        valid_data = json.load(f)
    
    for key in train_data:
        if key in ['input_ids', 'token_type_ids', 'attention_mask']:
            train_data[key] = torch.LongTensor(train_data[key])
        if key == 'ans':
            train_data[key] = torch.FloatTensor(train_data[key])
            
    
    for key in test_data:
        if key in ['input_ids', 'token_type_ids', 'attention_mask']:
            test_data[key] = torch.LongTensor(test_data[key])
        if key == 'ans':
            test_data[key] = torch.FloatTensor(test_data[key])
    
    for key in valid_data:
        if key in ['input_ids', 'token_type_ids', 'attention_mask']:
            valid_data[key] = torch.LongTensor(valid_data[key])
        if key == 'ans':
            valid_data[key] = torch.FloatTensor(valid_data[key])
    
    return train_data, valid_data, test_data