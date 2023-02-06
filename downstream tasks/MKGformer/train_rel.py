import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('111')

import torch
import torch.nn as nn
import argparse
import importlib
import numpy as np
from models_rel import UnimoForMaskedLM, UnimoForREC
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
from pytorch_pretrained_bert.tokenization import BertTokenizer

import pickle

max_seq_length=256

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
        head = line[0]
        rel = line[1]
        tail = line[2]

        image_path1 = imgpath = '../' + word2photos[head]
        image_path2 = imgpath = '../' + word2photos[tail]
        label = relations_to_id[rel]

        img1 = Image.open(image_path1).convert('RGB')
        input_img = self.clip_processor(images=img1, return_tensors='pt')['pixel_values'][0]

        img2 = Image.open(image_path2).convert('RGB')
        aux_image = self.clip_processor(images=img2, return_tensors='pt')['pixel_values'][0]
        
        return (
            torch.tensor(entities_to_id[head]),
            torch.tensor(relations_to_id[rel]),
            torch.tensor(entities_to_id[tail]),
            input_ids,
            token_type_ids,
            attention_mask,
            input_img,
            aux_image,
            torch.tensor(label)
        )

    def __len__(self):
        return len(self.lines)
    
    def convert_examples_to_features(self, line):
        head = line[0]
        rel = line[1]
        tail = line[2]
            
        pretrain = self.pretrain

        text_a = head.replace('_', ' ') + ' ' + allword2abstract[head]
        text_b = tail.replace('_', ' ') + ' ' + allword2abstract[tail]
        text_c = "[MASK]"
        
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        tokens_c = tokenizer.tokenize(text_c)

        tokens_a, tokens_b, tokens_c = _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        tokens += tokens_c + ["[SEP]"]
        segment_ids += [0] * (len(tokens_c) + 1) 
        
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [0] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(segment_ids)
        attention_mask = torch.tensor(input_mask)
            
        return input_ids, token_type_ids, attention_mask
    

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b, tokens_c

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
    valid_hits10 = -1
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
        for head, rel, tail, input_ids, token_type_ids, attention_mask, input_img, aux_image, label in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_img = input_img.to(device)
            label = label.to(device)
            aux_image = aux_image.to(device)
            logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, pixel_values=input_img, aux_values=aux_image)
            _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)    # bsz
            bsz = input_ids.shape[0]
            logits = logits[torch.arange(bsz), mask_idx] # bsz, entites
            assert mask_idx.shape[0] == bsz, "only one mask in sequence!"
            loss = loss_fn(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("train loss:", loss)
        hits1, _, hits10, _, _ = _eval(model, valid_dataloader1, "valid")
        if hits10> valid_hits10: 
            valid_hits10 = hits10
            test_hits1, test_hits3, test_hits10, test_MR, test_MRR = _eval(model, test_dataloader1, "test")
            if test_hits1> best_test_hits1:
                best_test_hits1 = test_hits1.copy()
                best_test_hits3 = test_hits3.copy()
                best_test_hits10 = test_hits10.copy()
                best_test_MR = test_MR.copy()
                best_test_MRR = test_MRR.copy()
                best_epoch = epoch
    
    print('------------final_result------------------') 
    print("best_test_hits1",test_hits1)
    print("best_test_hits3",test_hits3)
    print("best_test_hits10",test_hits10)
    print("best_test_MR",test_MR)
    print("best_test_MRR",test_MRR)
    print("best_epoch",epoch)
    print('------------------------------------------')

    with open('result.txt',"a") as f1:
        f1.write('noise:{}, batch_size:{} \n mrank:{}, mrr:{}, hits1:{}, hits3:{}, hits10:{} \n'\
        .format(str(noise), str(batch_size), str(test_MR), str(test_MRR), str(test_hits1), str(test_hits3), str(test_hits10)))
    
def _eval(model, eval_data, eval_type):
    print("start {}".format(eval_type))
    model.eval()
    all_ranks = []
    respect_ranks = {}
    respect_ranks['predict_tail'] = []
    respect_ranks['predict_head'] = []
    with torch.no_grad():
        one_ranks = []
        for query, rel, ans, input_ids, token_type_ids, attention_mask, input_img, aux_image, label in tqdm(eval_data):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_img = input_img.to(device)
            aux_image = aux_image.to(device)
            label = label.to(device)
            logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, pixel_values=input_img, aux_values=aux_image)
            bsz = input_ids.shape[0]

            _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)    # bsz
            logits = logits[torch.arange(bsz), mask_idx] # bsz, entites
            assert mask_idx.shape[0] == bsz, "only one mask in sequence!"


            _, outputs = torch.sort(logits, dim=1, descending=True) # bsz, entities   
            _, outputs = torch.sort(outputs, dim=1) 

            ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
            ranks = ranks.tolist()
            all_ranks += ranks
            one_ranks += ranks

        respect_ranks = one_ranks

    all_ranks = np.array(all_ranks)
    hits10 = (all_ranks<=10).mean()
    hits3 = (all_ranks<=3).mean()
    hits1 = (all_ranks<=1).mean()
    
    print("{}/mean_rank".format(eval_type), all_ranks.mean())
    print("{}/mrr".format(eval_type), (1. / all_ranks).mean())
    
    print("{}/hits1".format(eval_type), hits1)
    print("{}/hits3".format(eval_type), hits3)
    print("{}/hits10".format(eval_type), hits10)
   
    
    return hits1, hits3, hits10, all_ranks.mean(), (1. / all_ranks).mean()


def get_triplets(lines):
    examples = []
    for line in lines:
        RDF = line.split(' ')[:-1] # . 不要
        tail = RDF[-1].split('/')[-1][:-1]
        rel = RDF[1].split('/')[-1][:-1]
        head = RDF[0].split('/')[-1][:-1]
        examples.append((head, rel, tail))
    return examples


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
        '--epochs', default=12, type=int,
        help="Number of epochs."
        )
        parser.add_argument(
        '--lr', default=4e-5
        )
        parser.add_argument(
            '--batch_size', default=2, type=int,
            help="Factorization rank."
        )
        parser.add_argument(
            '--device', default='cuda:1'
        )
        parser.add_argument(
            '--noise', default=False
        )

        args = parser.parse_args()
        
        eps = 1e-8
        weight_decay = 1e-8
        num_workers = 8
        
        lr= args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        device = args.device
        noise= args.noise
        black= args.black

        clip_path = '../openaiclip-vit-base-patch32'
        bert_path = '../bert-base-uncased'
        path = "../DBtail_"

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

        tokenizer = BertTokenizer.from_pretrained(bert_path)

        vision_config = CLIPConfig.from_pretrained(clip_path).vision_config
        text_config = BertConfig.from_pretrained(bert_path)

        bert = BertModel.from_pretrained(bert_path)
        clip_model = CLIPModel.from_pretrained(clip_path)
        clip_vit = clip_model.vision_model
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()

        vision_config.device = device
        vocab_size = len(relations_to_id)
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
            with open("../entity2noise_images.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        else:
            with open("../entity2image.json","r",encoding='utf-8') as f:
                word2photos = json.load(f)
        print(len(word2photos))


        train_examples = get_triplets(dataset[0])
        valid_examples1 = get_triplets(dataset[1])
        test_examples1 = get_triplets(dataset[2])

        train_dataset = MyDataset(train_examples, black=black)
        valid_dataset1 = MyDataset(valid_examples1, black=black)
        test_dataset1 = MyDataset(test_examples1, black=black)



        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader1 = DataLoader(valid_dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        training_step(model, train_dataloader)
    except:
        pass

