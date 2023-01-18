from tkinter import image_types
from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from imageFeature_torch import ImageFeature
from transformers import ViltProcessor, ViltForMaskedLM, AdamW
import json
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from copy import deepcopy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AverageMeter:
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

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--train_file", type=str, default="DBtail_train.json")
parser.add_argument("--valid_file", type=str, default="DBtail_valid.json")
parser.add_argument("--test_file", type=str, default="DBtail_test.json")
parser.add_argument("--do_train", action="store_true", default=True)
parser.add_argument("--do_eval", action="store_true", default=False)
parser.add_argument("--with_image", action="store_true", default=False)
parser.add_argument("--log_dir", type=str, default="runs")
parser.add_argument("--dataset_dir", type=str, default="../")
parser.add_argument("--image_type", type=str, default="Our")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--task", type=str, default="pt")


args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

with open("../entity2id.json","r",encoding='utf-8') as f:
    entity2id = json.load(f)

if args.image_type == 'Noise':
  with open("../entity2noise_images.json","r",encoding='utf-8') as f:
      entity2img = json.load(f)

if args.image_type == 'Our':
  with open("../entity2image.json","r",encoding='utf-8') as f:
      entity2img = json.load(f)
  
num_workers=4

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
transform = transforms.Compose([
  transforms.Resize((pretrained_size, pretrained_size)),
  transforms.ToTensor(),
  transforms.Normalize(mean = pretrained_means, 
                        std = pretrained_stds)
])



class WordDataset(Dataset):
  def __init__(self, data_file, data_dir, with_image=False, image_token_num=2, max_length=40):
    super(WordDataset, self).__init__()
    self.data_dir = data_dir
    self.with_image = with_image
    self.processor = ViltProcessor.from_pretrained("./vilt")

    with open(data_dir + data_file,"r",encoding='utf-8') as f:
        data = json.load(f)
    self.data = data
    self.max_length = max_length
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    line = self.data[idx]
    RDF = line.split(' ')[:-1] # . 不要
    tail = RDF[-1].split('/')[-1][:-1]
    rel = RDF[1].split('/')[-1][:-1]
    head = RDF[0].split('/')[-1][:-1]


    if args.task == 'pt':
      text = "The {} of {} is [MASK]".format(rel, head)
      image_path = "../"+entity2img[head]
      o = entity2id[tail]
      o = torch.tensor(o)
    elif args.task == 'ph':
      text = "The {} of [MASK] is {}".format(rel, tail)
      o = entity2id[head]
      o = torch.tensor(o)
      image_path = "../"+entity2img[tail]

    if self.with_image:
      img = Image.open(image_path).resize((384, 384)).convert('RGB')

      encoding = self.processor(img, text, return_tensors="pt", padding='max_length', max_length=self.max_length)

      return text, o, encoding.input_ids[0], encoding.attention_mask[0], encoding.pixel_values[0]


# pytorch dataloader
      
class WordModel(nn.Module):
  def __init__(self, encoder_hidden_dim, device, maxlength=40, image_token_num=2, 
              with_image=False, image_encode_dim=2048, entity_num=len(entity2id)):
    super(WordModel, self).__init__()
    
    self.ViLT_1 = list(ViltForMaskedLM.from_pretrained("./vilt").children())[0]
    
    self.device = device
    self.max_length = maxlength

    self.with_image = with_image

    self.classifer = nn.Linear(encoder_hidden_dim, entity_num)

  def forward(self, text, o, input_ids, attention_mask, image=None):
    bs = len(text)
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    image = image.to(self.device)

    mask_pos = torch.nonzero(input_ids == 103, as_tuple=True)[1]
    assert len(mask_pos) == bs 

    output = self.ViLT_1(input_ids = input_ids, pixel_values = image, attention_mask=attention_mask)['last_hidden_state']
    logits = self.classifer(output) 

    re = []
    for i in range(bs):
      re.append(logits[i][mask_pos[i]])

    labels = o

    return torch.stack(re, dim=0).to(self.device), labels.to(self.device)


test_data = WordDataset(args.test_file, args.dataset_dir, args.with_image)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

def test(baseModel):
  baseModel = baseModel.to(device)
  baseModel.eval()

  tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
  mrank = torch.empty((0,), dtype=torch.float32)
  mrr = torch.empty((0,), dtype=torch.float32)

  hits = [0, 0, 0]
  test_num = 0
  with torch.no_grad():
    count = 0
    for _, data in enumerate(tk):
      logits, labels = baseModel(*data)
      logits = logits.cpu()
      labels = labels.cpu()

      sorted_logits = torch.sort(-logits, dim=-1)[1] 
      labels = labels.unsqueeze(1).expand(sorted_logits.shape)
      rank = torch.nonzero(sorted_logits == labels, as_tuple=True)[1]+1
      mrank = torch.cat([mrank, rank], dim=0) 
      mrr = torch.cat([mrr, 1/rank], dim=0)
      for i, k in enumerate([1, 3, 10]):
        hits_k = torch.sum(torch.any(labels[:, :k] == sorted_logits[:, :k], dim=-1)).item()
        hits[i] += hits_k
      test_num += labels.shape[0]
      count += 1

  print('mrr: {:.4f}\t mrank: {:.4f}\t hits@1: {:.4f}\t hits@3: {:.4f}\t hits@10: {:.4f}'
              .format(mrr.mean().item(), mrank.mean().item(), hits[0]/test_num, hits[1]/test_num, hits[2]/test_num))    
  
  return mrr.mean().item(), mrank.mean().item(), hits[0]/test_num, hits[1]/test_num, hits[2]/test_num
  

def valid(loader, baseModel):
  cnt = 0
  tk = tqdm(loader, total=len(loader), position=0, leave=True)
  with torch.no_grad():
    for _, data in enumerate(tk):
      logits, labels = baseModel(*data)
      preds = logits.argmax(dim=-1)
      cnt += torch.sum(preds == labels).item()
  return cnt


def train():
  print('learning rate: {}\nbatch_size: {}\nepochs: {}\n'.format(args.lr, args.batch_size, args.epochs))
  if args.with_image:
    print('with_image')
  train_data = WordDataset(args.train_file, args.dataset_dir, args.with_image)
  valid_data = WordDataset(args.valid_file, args.dataset_dir, args.with_image)

  baseModel = WordModel(768, device, with_image=args.with_image).to(device)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
  valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
  optimizer = AdamW(baseModel.parameters(), lr=args.lr)
  losses = AverageMeter()

  model_name = []

  best_epoch = -1
  best_test_mrr = -1
  best_test_mr = -1
  best_test_hit1 = -1
  best_test_hit3 = -1
  best_test_hit10 = -1

  best_cnt = -1
  for epoch in range(args.epochs):
    baseModel.train()
    loss_fn = nn.CrossEntropyLoss()
    tk = tqdm(train_loader)
    for data in tk:
      logits, labels = baseModel(*data)
      loss = loss_fn(logits, labels)
      loss.requires_grad_(True)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses.update(loss.item(), len(data))
      tk.set_postfix(loss=losses.avg)

    baseModel.eval()
    cnt = valid(valid_loader, baseModel)
    eval_acc = 1.0*cnt/valid_data.__len__()
    print('epoch {}: eval_acc: {:.4f}\n'.format(epoch, eval_acc))

    if best_cnt < cnt:
      best_cnt = cnt
      print('start test!')
      test_mrr, test_mr, test_hits1, test_hits3, test_hits10 = test(baseModel) # 最后保留的就是最终结构

  print("----final_result-----")
  print("test_mrr:", test_mrr)
  print("test_mr:", test_mr)
  print("test_hits1:", test_hits1)
  print("test_hits3:", test_hits3)
  print("test_hits10:", test_hits10)
  print("-------------------")

if args.do_train:
    train()
