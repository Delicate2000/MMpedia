import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import transformers
from transformers import BertTokenizer, BertModel, AdamW, ViltProcessor, ViltForMaskedLM
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import random_split
from PIL import ImageFile
from tqdm import tqdm
import time

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from dataset import get_dict
from models import get_models
import random
import json
import argparse




train_transforms = transforms.Compose(
        [
        transforms.CenterCrop(size=224),#中心裁剪到224*224
        transforms.ToTensor(),#转化成张量
        transforms.Normalize([0.485, 0.456, 0.406],#归一化
                             [0.229, 0.224, 0.225])
])

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, model_name=None, all_data=None, transform=None):
        self.transform = transform
        # self.loader = loader
        self.all_data = all_data
        self.model_name = model_name
        self.processor = ViltProcessor.from_pretrained("./vilt")
    def __getitem__(self,index):
        keyword = self.all_data["use_word"][index]
        # root_dir = img_root_dir + keyword
        
        imgs = torch.tensor(words2images[self.all_data["use_word"][index]])
        return (
            imgs,
            self.all_data['ans'][index],
            self.all_data['input_ids'][index],
            self.all_data['token_type_ids'][index],
            self.all_data['attention_mask'][index],
            self.all_data["use_word"][index],
        )

    def __len__(self):
        return len(self.all_data["use_word"])


def train(model, early_stop=False):
    max_f1 = -0.1 # 验证集上
    access_test_f1 = -0.1
    access_test_idx  = -0.1
    
    my_loss = nn.BCELoss()
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()
        for step,(imgs,label,input_ids,token_type_ids,attention_mask,word) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            label = label.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            y_predict = model(imgs,input_ids,token_type_ids,attention_mask)
            loss = my_loss(y_predict, label)
            print("train_loss:", loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        print("train_loss:", loss)
        f1 = valid(model)
        if f1 > max_f1:
            print("dev F-score: ", f1)
            print("Update model!")
            model_to_save = model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_path + output_model_file)
            max_f1 = f1

            test_f1 = test(model) 
            if test_f1 > access_test_f1:
                access_test_idx = epoch
                access_test_f1 = test_f1
        else:
            if early_stop: # valid上f1下降就停止训练
                break
    
    best_model = get_models(model_name).to(device)
    best_model.load_state_dict(torch.load(output_model_path + output_model_file))
    return best_model

def valid(valid_model):
    print("----------------start validing----------------")
    valid_model.eval()
    nonvisual_correct = 0
    visual_correct = 0
    visual_wrong = 0
    all_visual = 0
    with torch.no_grad():
        for step,(imgs,label,input_ids,token_type_ids,attention_mask,word) in enumerate(valid_dataloader):
            imgs = imgs.to(device)
            label = label.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            y_predict = valid_model(imgs,input_ids,token_type_ids,attention_mask)
            for i in range(y_predict.shape[0]):
                if label[i][0]>label[i][1]:
                    if y_predict[i][0]>y_predict[i][1]:
                        nonvisual_correct += 1
                    else:
                        visual_wrong += 1
                else:
                    all_visual += 1
                    if y_predict[i][0]<y_predict[i][1]:
                        visual_correct += 1

    recall = visual_correct/all_visual
    precision = visual_correct/(visual_correct+ visual_wrong)
    f1 = 2*(recall*precision)/(recall+precision)

    print('precision:',precision)
    print('recall:',recall)
    print('f1:',f1)
    print("--------------------------------\n")
    return f1

def test(test_model, case=False):   # case打印 bad case
    print("----------------start Testing----------------")
    test_model.eval() 
    # all_length = 0
    nonvisual_correct = 0
    visual_correct = 0
    visual_wrong = 0
    all_visual = 0
    # all_nonvisual = 0
    with torch.no_grad():
        for step,(imgs,label,input_ids,token_type_ids,attention_mask,word) in enumerate(test_dataloader):
            imgs = imgs.to(device)
            label = label.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            y_predict = test_model(imgs,input_ids,token_type_ids,attention_mask)
            
            for i in range(y_predict.shape[0]):
                # all_length += 1
                if label[i][0]>label[i][1]:
                    # all_nonvisual += 1
                    if y_predict[i][0]>y_predict[i][1]:
                        nonvisual_correct += 1
                    else:
                        visual_wrong += 1

                else:
                    all_visual += 1
                    if y_predict[i][0]<y_predict[i][1]:
                        visual_correct += 1
                        
    recall = visual_correct/all_visual
    precision = visual_correct/(visual_correct+ visual_wrong)
    f1 = 2*(recall*precision)/(recall+precision)

    print('test_precision:',precision)
    print('test_recall:',recall)
    print('test_f1:',f1)
    print("--------------------------------\n")
    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RGMM")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--date", type=str, default='10_25')
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    eps = 1e-8
    num_workers = 4

    model_name = args.model_name
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    device = args.device
    date = args.date

    output_model_path = '../models/{}/'.format(model_name)
    output_model_file = '{}+{}.pth'.format(model_name, date)

    # img_root_dir = "../dataset/download_images/"
    
    with open("./words2images.json","r",encoding='utf-8') as f:
        words2images = json.load(f)
    print("读取图像特征", len(words2images))
    
    print("output_model_file:", output_model_path+output_model_file)
    print("batch_size:", batch_size)
    print("lr:", lr)
    print("epochs:", epochs)
    print("model_name:", model_name)
    print("device:", args.device)

    train_data, valid_data, test_data = get_dict()
    train_dataset = MyDataset(all_data=train_data, model_name=model_name)
    valid_dataset = MyDataset(all_data=valid_data, model_name=model_name)
    test_dataset = MyDataset(all_data=test_data, model_name=model_name)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    model = get_models(model_name).to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                eps=eps,
                                weight_decay = weight_decay,
                                )
    
    best_model = train(model, early_stop=False)
    bad_case = {'word':[], 'correct_label':[]}
    test(best_model, case=True) # 测试最终结果