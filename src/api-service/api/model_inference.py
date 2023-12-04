import sys
import torch
import re
import json
import gzip
import numpy as np
from tqdm import tqdm
import random
import os
import wandb
import argparse
import shutil
import time
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)


def process_document(documents, doc_sep, max_source_length, tokenizer, DOCSEP_TOKEN_ID, PAD_TOKEN_ID):
    """Helper function to remove newlines, insert separator tokens, and apply padding. Returns a list of
    processed tensors representing a set of reviews.
    """
    input_ids_all=[]
    for data in documents:
        all_docs = data.split(doc_sep)[:-1]
        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc

        #### Add separator tokens
        input_ids = []
        for doc in all_docs:
            input_ids.extend(tokenizer.encode(doc,truncation=True,max_length=max_source_length // len(all_docs),)[1:-1])
            input_ids.append(DOCSEP_TOKEN_ID)
        input_ids = (
            [tokenizer.bos_token_id]
            + input_ids
            + [tokenizer.eos_token_id]
        )
        input_ids_all.append(torch.tensor(input_ids))
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID)
    return input_ids


def preprocess_function(examples, tokenizer, text_column, max_source_length, padding="max_length", prefix=""):
    """Preprocess review data in preparation for model training.
    Tokenize a set of reviews, add separator tokens, replace pad tokens, and setup global attention mask.
    """
    model_inputs = {}
    PAD_TOKEN_ID = tokenizer.pad_token_id
    DOCSEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("<doc-sep>")
    
    inputs= []
    for i in range(len(examples[text_column])):
        if examples[text_column][i]:
            inputs.append(examples[text_column][i])

    inputs = [prefix + inp for inp in inputs]
    
    model_inputs['input_ids'] = process_document(inputs,
                                                 doc_sep='|||||', 
                                                 max_source_length=max_source_length,
                                                 tokenizer=tokenizer, 
                                                 DOCSEP_TOKEN_ID=DOCSEP_TOKEN_ID, 
                                                 PAD_TOKEN_ID=PAD_TOKEN_ID)

    
    global_attention_mask = torch.zeros_like(model_inputs['input_ids']).to(model_inputs['input_ids'])
    
    # Global attention should be on separator token
    global_attention_mask[:, 0] = 1
    global_attention_mask[model_inputs['input_ids'] == DOCSEP_TOKEN_ID] = 1
    
    model_inputs["global_attention_mask"] = global_attention_mask
    
    return model_inputs

def inference_batch(examples, model, tokenizer, max_len=512, num_beams=3):
    """Run inference using a trained model. Tokenize inputs and use the trained model to generate a summary.
    Decode the summary and return it as regular text. Maximum length of the generated summary is max_len tokens.
    """
    input_ids = torch.tensor(examples['input_ids']).to(model.device)
    global_attention_mask=torch.tensor(examples['global_attention_mask']).to(model.device)
    
    generated_ids = model.generate(
        input_ids=input_ids,
        global_attention_mask=global_attention_mask,
        use_cache=True,
        max_length=max_len,
        num_beams=num_beams,
    )
    generated_str = tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
    result={}
    result['generated_summaries'] = generated_str
    return result

def generate_summary(text, use_finetuned=False):
    """Generate a summary of a group of reviews. Load the appropriate model,
    tokenize the reviews, and perform inference.
    """

    # If we are using our finetuned model, download from wandb, checking to 
    # see if the model has already been downloaded
    if use_finetuned:
        
        file_path = "../../secrets/wandb_key.txt"
        with open(file_path, "r") as wandb_file:
            key = wandb_file.readline().strip()
        
        wandb_download_folder = 'flavorfusion-team/FlavorFusion/model-pu67h5bz:v0'
        local_download_folder = "./model-pu67h5bz:v0"
        if not os.path.exists(local_download_folder):
            os.environ["WANDB_PROJECT"]="FlavorFusion"
            os.environ["WANDB_LOG_MODEL"]="false"
            os.environ["WANDB_WATCH"]="false"
            wandb.login(key=key)
            api = wandb.Api()
            artifact = api.artifact(wandb_download_folder)
            artifact_dir = artifact.download(root=local_download_folder)
            print("Model downloaded from wandb to: ", artifact_dir)
        else:
            artifact_dir = local_download_folder
        model_name = artifact_dir

    # If we are not using our finetuned model, we use the original PRIMERA
    # model trained on multi-news summarization
    else:
        model_name = 'allenai/PRIMERA-multinews'

    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    my_input = {"review": text}
    dataset = Dataset.from_dict(my_input)
    
    fn_kwargs = {'text_column': 'review', 
                 'tokenizer': tokenizer,
                 'max_source_length': 1024}

    dataset = dataset.map(
        preprocess_function,
        fn_kwargs=fn_kwargs,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset")

    fn_kwargs = {'model': model, 'tokenizer': tokenizer, 'max_len': 256, 'num_beams': 4}
    x = dataset.map(inference_batch, fn_kwargs=fn_kwargs, batched=True, batch_size=1)
    
    elapsed_time = time.time() - start_time
    print("Elapsed time for inference: " + str(elapsed_time))

    return x['generated_summaries']




