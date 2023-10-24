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
import zipfile
import tempfile

import evaluate

import torch.nn.utils.prune as prune

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
    set_seed
)

# Some functions in this script are adapted from the following example:
# https://gist.github.com/JohnGiorgi/8c7dcabd3ee8a362b9174c5d145029ab

def download_data(local_folder):
    """Download preprocessed LSARS data from GCS bucket.
    """
    bucket_name = args.gcs_bucket_name
    print("Downloading data from " + str(bucket_name))
    
    # Clear existing dataset folders so we get a clean copy
    shutil.rmtree(local_folder, ignore_errors=True, onerror=None)
    os.makedirs(local_folder, exist_ok=True)

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
    """Create global attention mask for Longformer Encoder Decoder."""
    global_attention_mask = [[1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids]
    return global_attention_mask


def split_docs(text: str, doc_sep_token: str):
    """Split a string into multiple reviews based on separator token."""
    text = re.sub(rf"{doc_sep_token}$", "", text.strip())
    return [doc.strip() for doc in text.split(doc_sep_token)]

def get_num_docs(text: str, doc_sep_token: str) -> int:
    """Get number of reviews in a string with separator token."""
    return len(list(filter(bool, split_docs(text, doc_sep_token=doc_sep_token))))


def truncate_multi_doc(text: str, doc_sep: str, doc_sep_token: str, max_length: int, tokenizer):
    """Truncate reviews in a token-separated string so that the length of all reviews together
    does not exceed max_length."""
    input_docs = split_docs(text, doc_sep_token=doc_sep)
    num_docs = get_num_docs(text, doc_sep_token=doc_sep)
    
    # -2 to make room for the special tokens, - num_docs - 1) to make room for the doc sep tokens
    max_doc_length = (max_length - 2 - (num_docs - 1)) // num_docs
    
    # Truncate each doc to its maximum allowed length
    truncated_docs = [
        tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc, max_length=max_doc_length, truncation=True)).strip()
        for doc in input_docs
    ]
    return f" {doc_sep_token} ".join(truncated_docs)

def sample_reviews(examples, max_docs_per_review=5, k_top_longest=20):
    """Perform data augmentation by sampling the k longest reviews in a given data point,
    then dividing into max_docs number of new data points.
    """
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
            input_ids.extend(tokenizer.encode(doc, truncation=True, max_length=max_source_length // len(all_docs),)[1:-1])
            input_ids.append(DOCSEP_TOKEN_ID)
        input_ids = ([tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id])
        input_ids_all.append(torch.tensor(input_ids))
        
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID)
    return input_ids


def preprocess_function(examples, tokenizer, text_column, summary_column, max_source_length, max_target_length, 
                        padding="max_length", ignore_pad_token_for_loss=True, prefix=""):
    """Preprocess review data in preparation for model training.
    Tokenize reviews and summaries, add separator tokens, replace pad tokens, and setup global attention mask.
    """
    model_inputs = {}
    PAD_TOKEN_ID = tokenizer.pad_token_id
    DOCSEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("<doc-sep>")
    
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    # Add prefix if using any special prefix string
    inputs = [prefix + inp for inp in inputs]
    
    # Truncate reviews where necessary and add separator tokens
    model_inputs['input_ids'] = process_document(inputs,
                                                 doc_sep='|||||', 
                                                 max_source_length=max_source_length,
                                                 tokenizer=tokenizer, 
                                                 DOCSEP_TOKEN_ID=DOCSEP_TOKEN_ID, 
                                                 PAD_TOKEN_ID=PAD_TOKEN_ID)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # Replace pad tokens so we can ignore padding when calculating the loss
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [ [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"] ]

    model_inputs["labels"] = labels["input_ids"]
    global_attention_mask = torch.zeros_like(model_inputs['input_ids']).to(model_inputs['input_ids'])
    
    # Global attention should be on separator token
    global_attention_mask[:, 0] = 1
    global_attention_mask[model_inputs['input_ids'] == DOCSEP_TOKEN_ID] = 1
    
    model_inputs["global_attention_mask"] = global_attention_mask
    return model_inputs


def postprocess_text(preds, labels):
    """Decode predictions after forward pass so we can do evaluation metrics."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def length_of_iterable_dataset(dataset):
    """Helper function to get the length of a streaming dataset."""
    count=0
    for i in dataset:
        count+=1
    return count

def get_model_size(model):
    """Helper function to calculate the unzipped and zipped size of a model file."""
    _, model_file = tempfile.mkstemp(".h5")
    torch.save(model.state_dict(), model_file)

    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(model_file)
        
    uncompressed_size = os.path.getsize(model_file) / float(1000)
    compressed_size = os.path.getsize(zip3) / float(1000)
    
    os.remove(model_file)
    os.remove(zip3)
        
    print("Model before zip: %.2f Kb"% (uncompressed_size))
    print("Model after zip: %.2f Kb"% (compressed_size))
    return uncompressed_size, compressed_size
        
def inference(batch, model, tokenizer, summary_field):
    """Run inference using a trained model. Tokenize inputs and use the trained model to generate one or more summaries.
    Decode the summaries and return them as regular text.
    """
    input_ids = batch['input_ids']

    # get the input ids and attention masks together
    global_attention_mask = batch['global_attention_mask']
    
    generated_ids = model.generate(
        input_ids=torch.tensor(input_ids).to(model.device),
        global_attention_mask=torch.tensor(global_attention_mask).to(model.device),
        use_cache=True,
        max_length=100,
        num_beams=1)
    
    generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
    result={}
    result['generated_summaries'] = generated_str
    result['gt_summaries']=batch[summary_field]
    return result
    

def main(args):
    
    # Sanity check if we are using GPU
    print("Now checking GPU availability")
    print(torch.cuda.is_available())
    
    # If using wandb, WANDB_KEY should be passed as an environment variable
    if args.wandb:
        os.environ["WANDB_PROJECT"]="FlavorFusion"
        os.environ["WANDB_LOG_MODEL"]="true"
        os.environ["WANDB_WATCH"]="false"
        wandb.login(key=args.wandb_key)
    
    # Option to use a model downloaded from wandb (for further training or pruning)
    if args.wandb_download_folder:
        run = wandb.init()
        artifact = run.use_artifact(args.wandb_download_folder, type="model")
        artifact_dir = artifact.download(root=args.input_dir)
        print("Model downloaded from wandb to: ", artifact_dir)
        args.model_name = artifact_dir
            
    # Load pre-trained model, tokenizer and config files
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    #Calculate base model size and compression metric so we can compare after pruning
    if (args.prune):
       base_uncompressed, base_compressed = get_model_size(model)
    
    if args.download:
        download_data(local_folder = args.input_dir)
        
    # Read all csv files in the data folder
    data_files = {'train': os.path.join(args.input_dir, '*.csv')}
    raw_dataset = load_dataset('csv', data_files=data_files, split='train', streaming = True if args.streaming else False)
    
    # Shuffle dataset and create train-test split. If using streaming, training and test datasets must be split manually
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
    
    # Run data augmentation to create more data points
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
    
    # Run data preprocessing
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
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=None)

    # Prepare evaluation metric
    #metric = load_metric("rouge")
    metric = evaluate.load("rouge")
    
    # Some pruning code adapted from https://github.com/pytorch/tutorials/issues/1054
    # Pruning is a post-training action, so if we are pruning, we assume we are training for 0 epochs
    if args.prune:
        #args.num_train_epochs = 0
        print("Now pruning the model to sparsity " + str(1 - args.prune_amount))
        
        # Construct a list of layers to prune
        suffix_list = ['output', 'fc1']
        #suffix_list = ['output']
        parameters_to_prune = [
            (v, "weight") 
            for k, v in dict(model.named_modules()).items()
            if ((len(list(v.children())) == 0) and (k.endswith(tuple(suffix_list))))
        ]
        
        # Apply global unstructured pruning to specified layers, removing the desired amount of weights
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=args.prune_amount
        )
        
        # We need to call prune.remove to make pruning permanent
        for k, v in dict(model.named_modules()).items():
            if ((len(list(v.children())) == 0) and (k.endswith(tuple(suffix_list)))):
                prune.remove(v, 'weight')
        
        # Calculate compression metrics after pruning
        prune_uncompressed, prune_compressed = get_model_size(model)
        #torch.save(model, args.model_output_path + "pruned_model.bin")
        
    # Generate training arguments 
    training_args = Seq2SeqTrainingArguments( 
        output_dir=args.model_output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end = False if (args.prune) else True,
        report_to="wandb" if args.wandb else None,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        max_steps=num_data_points//args.batch_size if args.streaming else -1,
        logging_steps=1,
        fp16=True,
        predict_with_generate=True)
    
    # Function to compute metrics for validation data at every epoch. This function must be declared
    # inside main so it can be passed to the trainer initialization.
    def compute_metrics(eval_preds):
        """Function to compute metrics for validation data at every epoch"""
        ignore_pad_token_for_loss = True
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Extract a few results from ROUGE metric
        result = {key: value * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    # Train the model with specified parameters
    if not args.prune:
        train_result = trainer.train()
    
    # Print metrics to show the effect of pruning
    if (args.prune):
        print(f"The compressed size of the base model is {round(base_compressed)} KB while the compressed size of the pruned model is {round(prune_compressed)} KB.")
        print(f"The optimized model is {round(base_compressed / prune_compressed, 2)}x smaller than the base model after compression.")
        
        #trainer.evaluate()
    
    # Close wandb and save artifacts
    if args.wandb:
        if 'run' not in locals():
            run = wandb.init()
        if args.prune:
            trainer.save_model(args.model_output_path)
            artifact = wandb.Artifact('pruned_model', type='model')
            artifact.add_dir(args.model_output_path)
            run.log_artifact(artifact)
        if not args.prune:
            wandb.finish()
    
    # Debugging option to run inference on a sample data point after traning to check results
    if args.inference:
        import random
        random.seed(42)
        
        fn_kwargs = {'model': trainer.model, 'tokenizer': tokenizer, 'summary_field': 'new_summary'}
        
        data_idx = random.choices(range(len(test_dataset)), k=1)
        test_dataset_small = test_dataset.select(data_idx)
        results = test_dataset_small.map(inference, fn_kwargs=fn_kwargs, batched=True, batch_size=1)
        
        print('Ground-truth summary: ')
        print(results[0]['gt_summaries'])
        
        print('Generated summary: ')
        print(results[0]['generated_summaries'])
        

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
    parser.add_argument('--wandb_key', dest="wandb_key", default="16", type=str, help="WandB API Key")
    parser.add_argument('--gcs_bucket_name', type=str, help='Path to GCS bucket to download data')
    parser.add_argument('--inference', action="store_true", help='Whether to do inference on one random test example after training')

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
    parser.add_argument('--prune', action="store_true", help='Whether to perform pruning on the model during the training')
    parser.add_argument('--prune_amount', type=float, default=0.5, help='Proportion of parameters to prune in prunable layers')
    parser.add_argument('--model_name', type=str, default='allenai/PRIMERA-multinews', help='Hugginface Name or local path of pretrained model to use')
    parser.add_argument('--wandb_download_folder', default=None, help='Full locaton of the model on wandb to download')
    
    
    args = parser.parse_args()
    
    main(args)