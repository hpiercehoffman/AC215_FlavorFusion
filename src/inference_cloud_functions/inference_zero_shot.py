import functions_framework
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

from datasets import load_dataset, load_metric, Dataset

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
    generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
    result={}
    result['generated_summaries'] = generated_str
    return result

@functions_framework.http
def hello_http(request):
    """HTTP cloud function. The input to this function is a Flask request object, which can be produced from 
    an API call or by directly manipulating the endpoint URL. We download the zero-shot model (PRIMERA-multinews)
    and run inference on the text input from the Flask request. We return a generated summary as a Flask response.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    print("request_json:", request_json)
    print("request_args:", request_args)
    
    # Default text which is used as input to first deployment
    text = "The new 110 Grill in Malden was a great first experience. The outside patio is perfect for eating outside during a nice summer night. It can get a little noisy with the train and traffic but it is nice to sit outside. Our waitress Viviane was amazing and catered to our every need. The food was also amazing! The firecracker shrimp appetizer was delicious. Their drinks were also very good too. Will definitely be going again to try new things!|||||Amazing staff and environment! I am gluten free so seeing a whole menu dedicated to me was something I was missing for a year!! Food was delicious, looked good and was prepared with care and passion!!\n\nDefinitely recommend everyone to come here and enjoy it as much as us... specially if like me you have a problem with gluten!!|||||Their attitude is very polite. Their service is perfect. You can find the best steak there. There is also a loyalty program that you can save on your meals for next visits. They have this Sandae brownie in the picture which tastes so good!"

    # If input text is provided to the endpoint, use this for inference instead of default text
    if request_args and "text" in request_args:
        text = request_args["text"]
    print(text)
    
    model_name = 'allenai/PRIMERA-multinews'
    local_cache_dir = "./PRIMERA-multinews"
    
    # Load the PRIMERA-multinews model from Huggingface
    if not os.path.exists("./PRIMERA-multinews"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
        config = AutoConfig.from_pretrained(model_name, cache_dir=local_cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_cache_dir)
        config = AutoConfig.from_pretrained(local_cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    model.resize_token_embeddings(len(tokenizer))

    # Create a dataset in proper format for inference
    my_input = {"review": [text]}
    dataset = Dataset.from_dict(my_input)
    fn_kwargs = {'text_column': 'review', 
                 'tokenizer': tokenizer,
                 'max_source_length': 1024}
    dataset = dataset.map(
        preprocess_function,
        fn_kwargs=fn_kwargs,
        batched=True,
        num_proc=1,
        desc="Running tokenizer dataset")

    # Run inference and return a generated summary
    fn_kwargs = {'model': model, 'tokenizer': tokenizer, 'max_len': 128, 'num_beams': 1}
    x = dataset.map(inference_batch, fn_kwargs=fn_kwargs, batched=True, batch_size=1)
    return x[0]['generated_summaries']


