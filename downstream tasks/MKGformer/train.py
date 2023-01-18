import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('111')

import torch
import torch.nn as nn
import argparse
import importlib
import numpy as np
from models import UnimoForMaskedLM, UnimoForREC
# import pytorch_lightning as pl
# from models.modeling_clip import CLIPModel
from transformers import CLIPConfig, BertConfig, BertModel, CLIPModel
from transformers import CLIPProcessor
from transformers import AutoTokenizer
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import json
import argparse

import pickle

max_length=256

class MyDataset(Dataset):
    def __init__(self, lines, pretrain=False, transform=None, black=False):
        self.lines = lines
        self.tokenizer = tokenizer
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.pretrain = pretrain
        self.black=black
    def __getitem__(self,index):
        line = self.lines[index]      
        input_ids, token_type_ids, attention_mask = self.convert_examples_to_features(line)
        query = line[0]
        rel = line[1]
        ans = line[2]

        image_path = imgpath = '../' + word2photos[query]
        label = entities_to_id[ans]
        img = Image.open(image_path).convert('RGB')
        input_img = self.clip_processor(images=img, return_tensors='pt')['pixel_values'][0]
        
        return (
            torch.tensor(entities_to_id[query]),
            torch.tensor(relations_to_id[rel]),
            label,
            input_ids,
            token_type_ids,
            attention_mask,
            input_img,
            torch.tensor(label)
        )

    def __len__(self):
        return len(self.lines)
    
    def convert_examples_to_features(self, line):
        query = line[0]
        rel = line[1]
        ans = line[2]
        mode = line[3]
            
        pretrain = self.pretrain

        text_a = query
        text_b = allword2abstract[query]
        text_c = rel
        
        if pretrain:
            # the des of xxx is [MASK] .
            # xxx is the description of [MASK].
            text = "The description of [MASK] is that {} .".format(text_b)
            x_texts = self.tokenizer(text,return_tensors="pt",padding='max_length',max_length=max_length,truncation=True)
            input_ids = x_texts['input_ids'][0]
            token_type_ids = x_texts['token_type_ids'][0]   
            attention_mask = x_texts['attention_mask'][0]
            
        else:
            if mode == 'predict_tail':
                deh_encoding = self.tokenizer(text_a+text_b,max_length=int(max_length-40),truncation=True)
                deh_inputing_ids = deh_encoding['input_ids']
                deh_token_type_ids = deh_encoding['token_type_ids']
                deh_attention_mask = deh_encoding['attention_mask']
                other_text = text_c + ' [SEP] ' + ' [MASK]'
                other_encoding = self.tokenizer(other_text)
                input_ids = deh_inputing_ids + other_encoding['input_ids'][1:]
                token_type_ids = deh_token_type_ids + other_encoding['token_type_ids'][1:]
                attention_mask = deh_attention_mask + other_encoding['attention_mask'][1:]
            else:
                deh_encoding = self.tokenizer(text_a+text_b,max_length=int(max_length-40),truncation=True)
                deh_inputing_ids = deh_encoding['input_ids'][1:] 
                deh_token_type_ids = deh_encoding['token_type_ids'][1:] 
                deh_attention_mask = deh_encoding['attention_mask'][1:] 

                other_text = ' [MASK] ' + ' [SEP] ' + text_c
                other_encoding = self.tokenizer(other_text)
                input_ids = other_encoding['input_ids'] + deh_inputing_ids
                token_type_ids = other_encoding['token_type_ids'] + deh_token_type_ids
                attention_mask = other_encoding['attention_mask'] + deh_attention_mask

            while len(input_ids) < max_length:
                input_ids.append(0)
                token_type_ids.append(0)
                attention_mask.append(0)
            assert len(input_ids)==len(token_type_ids)==len(attention_mask)==max_length
            input_ids = torch.tensor(np.array(input_ids))
            token_type_ids = torch.tensor(np.array(token_type_ids))
            attention_mask = torch.tensor(np.array(attention_mask))
            
        return input_ids, token_type_ids, attention_mask

def load_state_dict():
    """Load bert and vit pretrained weights"""
    vision_names, text_names = [], []
    model_dict = model.state_dict()
    for name in model_dict:
        if 'vision' in name:
            clip_name = name.replace('vision_', '').replace('model.', '').replace('unimo.', '')
            if clip_name in clip_model_dict:
                vision_names.append(clip_name)
                model_dict[name] = clip_model_dict[clip_name]
        elif 'text' in name:
            text_name = name.replace('text_', '').replace('model.', '').replace('unimo.', '')
            if text_name in text_model_dict:
                text_names.append(text_name)
                model_dict[name] = text_model_dict[text_name]
    assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(text_model_dict), \
                (len(vision_names), len(text_names), len(clip_model_dict), len(text_model_dict))
    model.load_state_dict(model_dict)
    print('Load model state dict successful.')

def training_step(model, train_dataloader):
    loss_fn = nn.CrossEntropyLoss()
    valid_hits1 = -1
    best_epoch = -1
    best_epoch = -1
    best_test_hits1 = -1
    best_test_hits3 = -1
    best_test_hits10 = -1
    best_test_MR = -1
    best_test_MRR = -1
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()
        for head, rel, tail, input_ids, token_type_ids, attention_mask, input_img, label in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_img = input_img.to(device)
            label = label.to(device)
            logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, pixel_values=input_img)

            _, mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
            bsz = input_ids.shape[0]
            logits = logits[torch.arange(bsz), mask_idx] # bsz, entites
            assert mask_idx.shape[0] == bsz, "only one mask in sequence!"
            loss = loss_fn(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("train loss:", loss)
        hits1, _, _, _, _ = _eval(model, valid_dataloader1, valid_dataloader2, "valid")
        if hits1> valid_hits1: 
            valid_hits1 = hits1
            test_hits1, test_hits3, test_hits10, test_MR, test_MRR = _eval(model, test_dataloader1, test_dataloader2, "test")


    print('------------final_result------------------')
    print("best_test_hits1",test_hits1)
    print("best_test_hits3",test_hits3)
    print("best_test_hits10",test_hits10)
    print("best_test_MR",test_MR)
    print("best_test_MRR",test_MRR)
    print("best_epoch",epoch)
    print('------------------------------------------')
    
def _eval(model, eval_data1, eval_data2, eval_type):
    print("start {}".format(eval_type))
    model.eval()
    all_ranks = []
    respect_ranks = {}
    respect_ranks['predict_tail'] = []
    respect_ranks['predict_head'] = []
    # print("mode:", mode)
    with torch.no_grad():
        for eval_num, eval_data in enumerate([eval_data1, eval_data2]):
            one_ranks = []
            if eval_num == 0:
                filter_out = ans_filter['rhs']
            else:
                filter_out = ans_filter['lhs']
            # for head, rel, tail, input_ids, token_type_ids, attention_mask, input_img, label in tqdm(eval_data):
            for query, rel, ans, input_ids, token_type_ids, attention_mask, input_img, label in tqdm(eval_data):
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                input_img = input_img.to(device)
                label = label.to(device)
                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, pixel_values=input_img)
                bsz = input_ids.shape[0]

                _, mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
                logits = logits[torch.arange(bsz), mask_idx] # bsz, entites
                assert mask_idx.shape[0] == bsz, "only one mask in sequence!"

                for index, _ in enumerate(input_ids):
                    if eval_num == 1:
                        logits_filter = filter_out[(query[index].item(), rel[index].item()+158)]
                    if eval_num == 0:
                        logits_filter = filter_out[(query[index].item(), rel[index].item())]
                    for one_ans in logits_filter:
                        if one_ans != label[index]:
                            logits[index][one_ans] -= torch.tensor(1e6)

                _, outputs = torch.sort(logits, dim=1, descending=True)
                _, outputs = torch.sort(outputs, dim=1)
                ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
                ranks = ranks.tolist()
                all_ranks += ranks
                one_ranks += ranks
            if eval_num == 0:
                respect_ranks['predict_tail'] = one_ranks
            if eval_num == 1:
                respect_ranks['predict_head'] = one_ranks
    all_ranks = np.array(all_ranks)
    hits10 = (all_ranks<=10).mean()
    hits3 = (all_ranks<=3).mean()
    hits1 = (all_ranks<=1).mean()
    
    if eval_type == 'valid':
        print("{}/mean_rank".format(eval_type), all_ranks.mean())
        print("{}/mrr".format(eval_type), (1. / all_ranks).mean())
        
        print("{}/hits1".format(eval_type), hits1)
        print("{}/hits3".format(eval_type), hits3)
        print("{}/hits10".format(eval_type), hits10)
    else:
        head_ranks = respect_ranks['predict_head']
        head_ranks = np.array(head_ranks)
        head_hits10 = (head_ranks<=10).mean()
        head_hits3 = (head_ranks<=3).mean()
        head_hits1 = (head_ranks<=1).mean()
        print("{}/head_mean_rank".format(eval_type), head_ranks.mean())
        print("{}/head_mrr".format(eval_type), (1. / head_ranks).mean())
        print("{}/head_hits1".format(eval_type), head_hits1)
        print("{}/head_hits3".format(eval_type), head_hits3)
        print("{}/head_hits10".format(eval_type), head_hits10)

        tail_ranks = respect_ranks['predict_tail']
        tail_ranks = np.array(tail_ranks)
        tail_hits10 = (tail_ranks<=10).mean()
        tail_hits3 = (tail_ranks<=3).mean()
        tail_hits1 = (tail_ranks<=1).mean()
        print("{}/tail_mean_rank".format(eval_type), tail_ranks.mean())
        print("{}/tail_mrr".format(eval_type), (1. / tail_ranks).mean())
        print("{}/tail_hits1".format(eval_type), tail_hits1)
        print("{}/tail_hits3".format(eval_type), tail_hits3)
        print("{}/tail_hits10".format(eval_type), tail_hits10)

    
    if eval_type == 'valid':
        return hits1, hits3, hits10, all_ranks.mean(), (1. / all_ranks).mean()
    else:
        return head_hits1, head_hits3, head_hits10, head_ranks.mean(), (1. / head_ranks).mean()

def get_triplets(lines):
    examples = []
    for line in lines:
        RDF = line.split(' ')[:-1] 
        tail = RDF[-1].split('/')[-1][:-1]
        rel = RDF[1].split('/')[-1][:-1]
        head = RDF[0].split('/')[-1][:-1]
        examples.append((head, rel, tail, 'predict_tail'))
        examples.append((tail, rel, head, 'predict_head'))
    return examples


def get_respect_triplets(lines):
    examples1 = []
    examples2 = []
    for line in lines:
        RDF = line.split(' ')[:-1]
        tail = RDF[-1].split('/')[-1][:-1]
        rel = RDF[1].split('/')[-1][:-1]
        head = RDF[0].split('/')[-1][:-1]
        examples1.append((head, rel, tail, 'predict_tail'))
        examples2.append((tail, rel, head, 'predict_head'))
    return examples1, examples2


if __name__ == '__main__':
    # try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
        '--epochs', default=12, type=int,
        help="Number of epochs."
        )
        parser.add_argument(
        '--lr', default=4e-5
        )
        parser.add_argument(
            '--batch_size', default=12, type=int,
            help="Factorization rank."
        )
        parser.add_argument(
            '--device', default='cuda:3'
        )
        parser.add_argument(
            '--noise', default=False
        )
        parser.add_argument(
            '--no_type', default=False
        )
        args = parser.parse_args()
        
        eps = 1e-8
        weight_decay = 1e-8
        num_workers = 4
        
        lr= args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        device = args.device
        # mode= args.mode
        noise= args.noise
        black= args.black
        no_type = args.no_type

        clip_path = '../openaiclip-vit-base-patch32'
        bert_path = '../bert-base-uncased'
        path = "../DBtail_"

        print("noise:", noise)
        print("lr:", lr)
        print("batch_size:", batch_size)
        print("epochs:", epochs)
        print("device:", device)

        with open("../entity2id.json","r",encoding='utf-8') as f1:
            entities_to_id = json.load(f1)
        with open("../relation2id.json","r",encoding='utf-8') as f1:
            relations_to_id = json.load(f1)

        dataset = []
        files = ['train', 'valid', 'test']
        for f in files:
            file_path = path + f + '.json'
            with open(file_path,"r",encoding='utf-8') as f1:
                lines = json.load(f1)
            dataset.append(lines)
        print(len(dataset))

        tokenizer = AutoTokenizer.from_pretrained(bert_path)

        vision_config = CLIPConfig.from_pretrained(clip_path).vision_config
        text_config = BertConfig.from_pretrained(bert_path)

        bert = BertModel.from_pretrained(bert_path)
        clip_model = CLIPModel.from_pretrained(clip_path)
        clip_vit = clip_model.vision_model
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()

        vision_config.device = device

        vocab_size = len(entities_to_id)
        model = UnimoForMaskedLM(vision_config, text_config, vocab_size=vocab_size)
        
        load_state_dict()
        model = model.to(device)

        with open("../entity2text.json","r",encoding='utf-8') as f:
            allword2abstract = json.load(f)
        print(len(allword2abstract))
        if noise:
            print("noise image!")
            with open("../word2noise_images.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        elif black:
            print("random image!")
            with open("../word2random_images.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        elif no_type:
            print("no_type image!")
            with open("../entity2image_notype.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        else:
            with open("../entity2image.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        print(len(word2photos))


        train_examples = get_triplets(dataset[0])
        valid_examples1, valid_examples2 = get_respect_triplets(dataset[1])
        test_examples1, test_examples2 = get_respect_triplets(dataset[2])

        train_dataset = MyDataset(train_examples, black=black)
        # train_dataset1 = MyDataset(dataset[0], mode="predict_tail", black=black)
        valid_dataset1 = MyDataset(valid_examples1, black=black)
        valid_dataset2 = MyDataset(valid_examples2, black=black)
        # valid_dataset1 = MyDataset(dataset[1], mode="predict_tail", black=black)
        test_dataset1 = MyDataset(test_examples1, black=black)
        test_dataset2 = MyDataset(test_examples2, black=black)
        # test_dataset1 = MyDataset(dataset[2], mode="predict_tail", black=black)

        print("num_workers:", num_workers)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader1 = DataLoader(valid_dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        valid_dataloader2 = DataLoader(valid_dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        ans_filter = pickle.load(open('../MoSE+RSME/data/DB15K/to_skip.pickle', 'rb'))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        training_step(model, train_dataloader)

