import sys
import torch
import re
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import random
import os
import wandb
import argparse
import shutil

from google.cloud import storage

from datasets import load_dataset, load_metric
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


def download_data(local_folder):
    GCS_BUCKET_NAME = os.environ["GCS_DATA_BUCKET"]
    
    bucket_name = GCS_BUCKET_NAME
    print("Downloading data from " + str(bucket_name))
    
    # Clear existing dataset folders so we get a clean copy
    #shutil.rmtree(local_folder, ignore_errors=True, onerror=None)
    #os.makedirs(local_folder, exist_ok=True)

    # Initiate storage client and download data
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="translated_data/")
    for blob in blobs:
        print("Data file:", blob.name)
        if not blob.name.endswith("translated_data/"):
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(local_folder, filename)
            blob.download_to_filename(local_file_path)


def get_global_attention_mask(input_ids, token_ids):
    """Returns a corresponding global attention mask for `input_ids`, which is 1 for any tokens in
    `token_ids` (indicating the model should attend to those tokens) and 0 elsewhere (indicating the
    model should not attend to those tokens).
    # Parameters
    input_ids : `List[List[str]]`
        The input ids that will be provided to a model during the forward pass.
    token_ids : `List[List[str]]`
        The token ids that should be globally attended to.
    """
    global_attention_mask = [[1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids]
    return global_attention_mask


def split_docs(text: str, doc_sep_token: str):
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns a list of each individual documents. Ignores any documents that are empty.
    order of documents in each example.
    """
    # It's possible to have a doc_sep_token at the very end of the string. Strip it here
    # so that we get the correct number of documents when we split on doc_sep_token.
    text = re.sub(rf"{doc_sep_token}$", "", text.strip())
    return [doc.strip() for doc in text.split(doc_sep_token)]


def get_num_docs(text: str, doc_sep_token: str) -> int:
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns the number of individual documents.
    """
    # See: https://stackoverflow.com/a/3393470
    return len(list(filter(bool, split_docs(text, doc_sep_token=doc_sep_token))))


def truncate_multi_doc(
    text: str,
    doc_sep: str,
    doc_sep_token: str,
    max_length: int,
    tokenizer):
    
    """Given some `text`, which is assumed to be multiple documents joined by `doc_sep_token`,
    truncates each document (using `tokenizer`) so that the length of the concatenation of all
    documents does not exceed max_length. See https://aclanthology.org/2021.naacl-main.380/ and
    https://arxiv.org/abs/2110.08499 for more details. If `num_docs` is provided, the truncation
    is done as if there are `num_docs` number of input documents. This is useful to control
    for truncation when applying pertubations (e.g. additiion and deletion).
    """
    input_docs = split_docs(text, doc_sep_token=doc_sep)
    # If num_docs is not provided, determine it from the input text
    num_docs = get_num_docs(text, doc_sep_token=doc_sep)
    # -2 to make room for the special tokens, -(len(docs) - 1) to make room for the doc sep tokens.
    max_doc_length = (max_length - 2 - (num_docs - 1)) // num_docs
    # Truncate each doc to its maximum allowed length
    truncated_docs = [
        # Going to join everything on a space at the end, so strip it off here.
        tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc, max_length=max_doc_length, truncation=True)).strip()
        for doc in input_docs
    ]
    return f" {doc_sep_token} ".join(truncated_docs)

def sample_reviews(examples, max_docs_per_review=5, k_top_longest=20):
    text_column = 'review_str'
    summary_column = 'summary'
    
    new_reviews = []
    new_summaries = []
    for i in range(len(examples[text_column])):
        summary = examples[summary_column][i]
        docs = examples[text_column][i]
        
        docs = split_docs(docs, '|||||')
        longest_docs = sorted(docs, key=len, reverse=True)[:k_top_longest]
        random.shuffle(longest_docs)
        new_docs = [longest_docs[i:i + max_docs_per_review] for i in range(0, len(longest_docs), max_docs_per_review)]
        new_docs = ['|||||'.join(new_docs_i) for new_docs_i in new_docs]
        new_reviews += new_docs
        new_summaries += [summary]*len(new_docs)
    return {'augmented_review_str': new_reviews, 'new_summary': new_summaries}


def preprocess_function(examples, tokenizer, text_column, summary_column, max_source_length, max_target_length, 
                        padding="max_length", ignore_pad_token_for_loss=True, prefix=""):
    # remove pairs where at least one record is None
    
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]

    inputs = [
        truncate_multi_doc(
            text,
            doc_sep="|||||",
            doc_sep_token=tokenizer.additional_special_tokens[0],
            max_length=max_source_length,
            tokenizer=tokenizer,
        )
        for text in inputs
    ]

    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    # Add a global attention mask to models inputs. We don't bother checking if the model will
    # actually use it, as it will be ignored if not. For summarization, we place global attention
    # on the document seperator token and the bos token (if it exists).
    global_attention_tokens = [tokenizer.bos_token, tokenizer.additional_special_tokens[0]]
    model_inputs["global_attention_mask"] = get_global_attention_mask(
        model_inputs.input_ids,
        token_ids=tokenizer.convert_tokens_to_ids(global_attention_tokens),
    )
    #print(f"Using global attention on the following tokens: {global_attention_tokens}")
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def length_of_iterable_dataset(dataset):
    count=0
    for i in dataset:
        count+=1
    return count

def main(args):
    
    # Make sure you have done 'wandb auth' 
    if args.wandb:
        os.environ["WANDB_PROJECT"]="FlavorFusion"
        os.environ["WANDB_LOG_MODEL"]="true"
        os.environ["WANDB_WATCH"]="false"
        wandb.init(dir=args.input_dir)
    
    # Load pre-trained model, tokenizer and config files
    tokenizer = AutoTokenizer.from_pretrained('allenai/PRIMERA')
    config = AutoConfig.from_pretrained('allenai/PRIMERA')
    model = AutoModelForSeq2SeqLM.from_pretrained('allenai/PRIMERA')
    model.resize_token_embeddings(len(tokenizer))
    
    # TODO: Have to check if this works
    if args.download:
        download_data(local_folder = args.input_dir)
        
    # Reads all csv files in the given folder
    data_files = {'train': os.path.join(args.input_dir, '*.csv')}
    raw_dataset = load_dataset('csv', data_files=data_files, split='train', streaming = True if args.streaming else False)
    
    # Shuffle dataset and create train-test split
    raw_dataset = raw_dataset.shuffle(seed=42)
    if args.streaming:
        TOTAL_SIZE = 5100
        raw_dataset_test = raw_dataset.take(int(args.test_ratio*TOTAL_SIZE))
        raw_dataset_train = raw_dataset.skip(int(args.test_ratio*TOTAL_SIZE))
        column_names = ['review_str', 'summary', 'id']
    else:
        column_names = raw_dataset.column_names
        raw_dataset = raw_dataset.train_test_split(test_size=args.test_ratio)
    
    # Augment the data by shortening each multi-review set to only 'args.max_docs_per_review' (default=5) per review set
    aug_kwargs = {'max_docs_per_review':args.max_docs_per_review, 'k_top_longest': args.k_top_longest}
    
    if args.streaming:
        aug_dataset_train = raw_dataset_train.map(
            sample_reviews,
            fn_kwargs=aug_kwargs,
            batched=True,
            remove_columns=column_names)
        aug_dataset_test = raw_dataset_test.map(
            sample_reviews,
            fn_kwargs=aug_kwargs,
            batched=True,
            remove_columns=column_names)
        num_data_points = length_of_iterable_dataset(aug_dataset_train)
    else:
        aug_dataset = raw_dataset.map(
            sample_reviews,
            fn_kwargs=aug_kwargs,
            batched=True,
            num_proc=args.num_processes,
            remove_columns=column_names,
            desc="Augmenting train and test datasets")
    
    # Tokenize data and construct attention masks
    token_kwargs = {'text_column': 'augmented_review_str', 
                    'tokenizer': tokenizer,
                    'summary_column': 'new_summary', 
                    'max_source_length': args.max_source_length, 
                    'max_target_length': args.max_target_length}
    
    
    if args.streaming:
        train_dataset = aug_dataset_train.map(
            preprocess_function,
            fn_kwargs=token_kwargs,
            batched=True)
        test_dataset = aug_dataset_test.map(
            preprocess_function,
            fn_kwargs=token_kwargs,
            batched=True)
    else:
        full_dataset = aug_dataset.map(
            preprocess_function,
            fn_kwargs=token_kwargs,
            batched=True,
            num_proc=args.num_processes,
            desc="Running tokenizer on train and test datasets")
    
    # If you want to make dataset smaller (for debugging)
    if args.subset_dataset_to is not None:
        if args.streaming:
            train_dataset = train_dataset.take(args.subset_dataset_to)
            test_dataset = test_dataset.take(args.subset_dataset_to)
        else:
            train_dataset = full_dataset['train'].select(range(args.subset_dataset_to))
            test_dataset = full_dataset['test'].select(range(args.subset_dataset_to))
    else:
        train_dataset = full_dataset['train']
        test_dataset = full_dataset['test']
        
    # Prepare data collators
    label_pad_token_id = -100 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    # Prepare metric
    metric = load_metric("rouge")
    
    # Function to compute metrics for validation data at every epoch
    def compute_metrics(eval_preds):
        ignore_pad_token_for_loss = True
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    # Generate training arguments 
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.input_dir, "results"),
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="wandb" if args.wandb else None,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=num_data_points//args.batch_size if args.streaming else -1,
        logging_steps=1,
        fp16=True,
        predict_with_generate=True)
    
    # Initialize main trainer 
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    # Train
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(args.model_output_path)
    
    # Close wandb
    if args.wandb:
        wandb.finish()
    

if __name__ == "__main__":
    
    # Needed to calculate metrics via ROUGE score
    nltk.download('punkt')
    
    parser = argparse.ArgumentParser(description='Training PRIMERA model using huggingface')
    
    # General args
    parser.add_argument('--input_dir', type=str, help='Path to folder containing LSARS csv files')
    parser.add_argument('--model_output_path', type=str, help='Path to model output')
    parser.add_argument('--download', action="store_true", help="Download processed LSARS data from GCS bucket")
    parser.add_argument('--wandb', action="store_true", default=False, help='Whether to use wandb for logging')
    parser.add_argument('--streaming', action="store_true", default=False, help='Whether to stream data')

    
    # Data processing args
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Test ratio for splitting')
    parser.add_argument('--max_docs_per_review', type=int, default=5, help='Maximum number of reviews per data point')
    parser.add_argument('--k_top_longest', type=int, default=20, help='Keep only the k longest reviews for each data point')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes used for multiprocessing the dataset')
    parser.add_argument('--max_source_length', type=int, default=4096, help='Maximum number of tokens for each set of reviews')
    parser.add_argument('--max_target_length', type=int, default=1024, help='Maximum number of tokens for each summary in a review set')
    parser.add_argument('--subset_dataset_to', type=int, default=None, help='Whether to subset dataset for debugging')
    
    # Training args
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_train_epochs', type=int, default=20, help='Total number of epochs for training')
    
    args = parser.parse_args()
    
    main(args)


