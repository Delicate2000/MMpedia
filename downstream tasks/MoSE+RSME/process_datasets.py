import errno
import os
import pickle
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

DATA_PATH = '../data'


def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = path + f + '.json'
        with open(file_path,"r",encoding='utf-8') as f1:
            lines = json.load(f1)
        for line in lines:
            RDF = line.split(' ')[:-1] # remove .
            lhs = RDF[-1].split('/')[-1][:-1]
            rel = RDF[1].split('/')[-1][:-1]
            rhs = RDF[0].split('/')[-1][:-1]

            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)

    with open("../../entity2id.json","r",encoding='utf-8') as f1:
        entities_to_id = json.load(f1)
    with open("../../relation2id.json","r",encoding='utf-8') as f1:
        relations_to_id = json.load(f1)

    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)
    print(len(entities_to_id))
    print(len(relations_to_id))

    assert n_relations==len(relations_to_id) and n_entities == len(entities_to_id)
    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except:
        print("dir existed")

    for f in files:
        examples = []
        file_path = path + f + '.json'
        with open(file_path,"r",encoding='utf-8') as f1:
            lines = json.load(f1)
        for line in lines:
            RDF = line.split(' ')[:-1] # . 不要
            lhs = RDF[0].split('/')[-1][:-1]
            rel = RDF[1].split('/')[-1][:-1]
            rhs = RDF[-1].split('/')[-1][:-1]

            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue

        print(Path(DATA_PATH) / name / (f + '.pickle'))       
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))


    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    prepare_dataset('../../DBtail_', 'DB15K')
