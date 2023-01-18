import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch

from models import KBCModel

DATA_PATH = Path('../data')


class Dataset(object):
    def __init__(self, name: str, mode=None, device='cpu'):
        self.root = DATA_PATH / name 

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file) 

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)

        self.n_entities = int(max(maxis[0], maxis[2]) + 1) 
        self.n_predicates = int(maxis[1] + 1)
        # if mode != 'rel':
        #     self.n_predicates *= 2
            
        self.mode = mode
        self.device = device

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f) 
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        return self.data['train']
        
    def get_valid(self):
        return self.data['valid'] 

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        # examples = torch.from_numpy(test.astype('int64')).cuda()
        examples = torch.from_numpy(test.astype('int64'))
        device = self.device
        examples = examples.to(device)
        missing = [missing_eval]
        if missing_eval == 'both' and self.mode != 'rel':
            missing = ['rhs', 'lhs']
        else:
            missing = ['rhs']
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}


        missing = ['rhs'] 
        for m in missing:
            q = examples.clone()
            if n_queries > 0:  
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':  
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=2000)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            mean_rank[m] = torch.mean(ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_rank['rhs'], mean_reciprocal_rank['rhs'], hits_at['rhs']
       
    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
