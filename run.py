import os
import time
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25
from copy import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from collections import defaultdict, deque



get_args()
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-k", "--rtopk", type=int, default=3)
    parser.add_argument("-s", "--sample", type=int, default=1000)
    parser.add_argument("-f", "--fewshot", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config_path
    rtopk = args.rtopk
    sample = args.sample
    if args.fewshot is not None:
        fewshot=args.fewshot
    else:
        fewshot=None
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    args.rtopk = rtopk
    args.sample = sample
    if fewshot is not None:
        args.fewshot=fewshot
    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True
    return args

hand_args = get_args()

args = get_config(hand_args.config_path)
args.retrieve_topk = hand_args.rtopk
args.sample=hand_args.sample
args.fewshot=hand_args.fewshot

MAX_N = 5
MAX_K_DRAFT=7

if args.dataset == "strategyqa":
    data = StrategyQA(args.data_path)
elif args.dataset == "2wikimultihopqa":
    data = WikiMultiHopQA(args.data_path)
else:
    raise NotImplementedError
    
data.format(fewshot=args.fewshot)
data = data.dataset
if args.shuffle:
    data = data.shuffle()
if args.sample != -1:
    samples = min(len(data), args.sample)
    data = data.select(range(samples))

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model_config = LlamaConfig.from_pretrained(args.model_name_or_path,
                                                       trust_remote_code="falcon" in args.model_name_or_path)

model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, config=model_config, device_map="auto",
                                                          trust_remote_code="falcon" in args.model_name_or_path)


rag_model = RAG(args, tokenizer, model_config, model)
rag_model.retrieve_topk = args.retrieve_topk
rag_model.init_cache(data[0]["demo"])
rag_model.cache.past_key_values[1][0].shape,rag_model.cache.past_key_values[1][0].shape

for i in tqdm(range(len(data))):
    last_counter = copy(rag_model.counter)
    batch = data[i]
    pred, _ = rag_model.inference(batch["question"], batch["demo"], batch["case"], return_retrieve_logs=True)
    
    pred = pred.strip()
    print(pred)

