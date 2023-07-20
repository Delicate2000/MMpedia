import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import transformers
from transformers import BertTokenizer, BertModel, DetrForObjectDetection,DetrConfig
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from PIL import ImageFile
from transformers import ViltForMaskedLM

ImageFile.LOAD_TRUNCATED_IMAGES = True


bert_path = './bert-base-cased'

class RGMM(nn.Module):
    def __init__(self):
        super(RGMM, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.classifer = self.classify = nn.Sequential(
            nn.Linear(768,64),
            nn.ReLU(),            
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(768,768, bias=False),
            nn.Sigmoid()
        )
        
        self.size = nn.Sequential(
            nn.Linear(768*2,768),
        )
        self.res = nn.Sequential(
            nn.Linear(2048,768),
        )
        self.attn1 = nn.MultiheadAttention(768, 8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(768, 8, batch_first=True)
        self.attn3 = nn.MultiheadAttention(768*2, 8, batch_first=True)
        self.my_loss = nn.BCELoss()
    
    def mean_pooling(self, x_texts, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x_texts.size()).float()
        return torch.sum(x_texts * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, x_images, input_ids, token_type_ids, attention_mask):
        # attn_mask= ~attention_mask.bool()
        # self_attn_mask = ~torch.cat((torch.ones(attention_mask.size()), attention_mask),dim=-1).bool()

        x_texts = self.bert(input_ids,token_type_ids,attention_mask)["last_hidden_state"]
        # x_images (batch_size, img_num, img_features) (32,5,2048)
        x_images_new = self.res(x_images)
        batch_size = x_images_new.size()[0]       
        i_num = x_images_new.size()[1]

        H1 = x_texts
        # H1 = torch.mean(x_texts, dim=1)
        H1 = self.mean_pooling(x_texts, attention_mask)
        H1 = H1.reshape(batch_size,1,768)
        
        x_images_new = x_images_new.permute(1, 0, 2)
        for i in range(i_num):
            img_new = x_images_new[i].reshape(batch_size, 1, 768)
            img_c_text, _ = self.attn1(img_new, H1, H1)
            
            text_c_img, _ = self.attn2(H1, img_new, img_new)     
            X_e = torch.cat((text_c_img, img_c_text), dim= -1) 

            new_X_e, _ = self.attn3(X_e, X_e, X_e)
            new_X_e = self.size(new_X_e)
            
            if i == 0:            
                H1 = new_X_e
                H0 = H1
            else:
                Z = self.gate(new_X_e)
                H1 = Z*new_X_e + (1-Z)*H1
        
        H_final = H0 + H1
        H_final = H_final.squeeze(dim=1)
            
        output = self.classifer(H_final)
        
        return output


def get_models(name):
    if name == "RGMM":
        model = RGMM()
    return model
