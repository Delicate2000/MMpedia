import logging
import os

import torch
import json
import numpy as np

from config import config
from model import TransE, TransH, TransD, STransE, LineaRE, DistMult, ComplEx, RotatE, SimpleTransR, TransIJ
from utils import set_logger, read_elements, read_triples, log_metrics, save_model, train_data_iterator, \
    get_optim, train_step, test_step, rel_type, test_data_sets, get_true_ents
import argparse

os.environ['CUDA_VISIBLE_DEVICES']= '3'

def train(model, triples, ent_num):
    logging.info("Start Training...")
    logging.info("batch_size = %d" % config.batch_size)
    logging.info("dim = %d" % config.ent_dim)
    logging.info("gamma = %f" % config.gamma)

    current_lr = config.learning_rate
    #train_triples, valid_triples, test_triples, symmetry_test, inversion_test, composition_test, others_test = triples
    train_triples, valid_triples, test_triples = triples
    all_true_triples = train_triples + valid_triples + test_triples
    r_tp = rel_type(train_triples)

    optimizer = get_optim("Adam", model, current_lr)

    if config.init_checkpoint:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"), map_location=torch.device("cuda:0"))
        init_step = checkpoint["step"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.use_old_optimizer:
            current_lr = checkpoint["current_lr"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        init_step = 1

    true_all_heads, true_all_tails = get_true_ents(all_true_triples)
    train_iterator = train_data_iterator(train_triples, ent_num)
    test_data_list = test_data_sets(valid_triples, true_all_heads, true_all_tails, ent_num, r_tp)

    max_mrr = 0.0
    training_logs = []
    modes = ["Prediction Head", "Prediction Tail"]
    rtps = ["1-1", "1-M", "M-1", "M-M"]
    # Training Loop
    for step in range(init_step, config.max_step + 1):
        log = train_step(model, optimizer, next(train_iterator))
        training_logs.append(log)

        # log
        if step % config.log_step == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics("Training", step, metrics)
            training_logs.clear()

        # valid
        if step % config.valid_step == 0:
            logging.info("-" * 10 + "Evaluating on Valid Dataset" + "-" * 10)
            metrics = test_step(model, test_data_list, True)
            log_metrics("Valid", step, metrics[0])
            cnt_mode_rtp = 1
            for mode in modes:
                for rtp in rtps:
                    logging.info("-" * 10 + mode + "..." + rtp + "-" * 10)
                    log_metrics("Valid", step, metrics[cnt_mode_rtp])
                    cnt_mode_rtp += 1
            if metrics[0]["MRR"] >= max_mrr:
                max_mrr = metrics[0]["MRR"]
                save_variable_list = {
                    "step": step,
                    "current_lr": current_lr,
                }
                save_model(model, optimizer, save_variable_list)
            if step / config.max_step in [0.2, 0.5, 0.8]:
                current_lr *= 0.1
                logging.info("Change learning_rate to %f at step %d" % (current_lr, step))
                optimizer = get_optim("Adam", model, current_lr)

    # load best state
    checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"))
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint["step"]

    test_data_list = test_data_sets(test_triples, true_all_heads, true_all_tails, ent_num, r_tp)
    logging.info("----------Evaluating on Test Dataset----------")
    metrics = test_step(model, test_data_list, True)
    log_metrics("Test", step, metrics[0])
    cnt_mode_rtp = 1
    for mode in modes:
        for rtp in rtps:
            logging.info("-" * 10 + mode + "..." + rtp + "-" * 10)
            log_metrics("Test", step, metrics[cnt_mode_rtp])
            cnt_mode_rtp += 1


def run():
    # load data
    
    with open("../entity2id.json","r",encoding='utf-8') as f1:
        ent2id = json.load(f1)
 
    with open("../relation2id.json","r",encoding='utf-8') as f1:
            rel2id = json.load(f1)
    ent_num = len(ent2id)
    rel_num = len(rel2id)
    train_triples = read_triples(config.data_path + "train.json", ent2id, rel2id)
    valid_triples = read_triples(config.data_path +  "valid.json", ent2id, rel2id)
    test_triples = read_triples(config.data_path +  "test.json", ent2id, rel2id)

    logging.info("#ent_num: %d" % ent_num)
    logging.info("#rel_num: %d" % rel_num)
    logging.info("#train triple num: %d" % len(train_triples))
    logging.info("#valid triple num: %d" % len(valid_triples))
    logging.info("#test triple num: %d" % len(test_triples))
    logging.info("#Model: %s" % config.model)

    kge_model = TransE(ent_num, rel_num)
    kge_model = kge_model.cuda(torch.device("cuda:0"))
    logging.info("Model Parameter Configuration:")
    for name, param in kge_model.named_parameters():
        logging.info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))

    train(
        model=kge_model,
        triples=(
            train_triples,
            valid_triples,
            test_triples,
        ),
        ent_num=ent_num
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relational learning contraption"
    )
    set_logger()
    run()
