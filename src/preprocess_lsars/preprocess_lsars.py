import json
import gc
import pandas as pd
import argparse
from tqdm import tqdm
from itertools import islice

from google.cloud import translate_v2 as translate

def load_jsons(filepath, start_line, stop_line):
    with open(filepath, 'r') as fp:
        json_lines = islice(fp, start_line, stop_line)
        review_jsons = [json.loads(line) for line in json_lines]
    return review_jsons

def join_tokens(token_list):
    return "".join(token_list)

def translate_json(json, translator):
    item_id = json["item_id"]
    summary = join_tokens(json["hq_tokens"])
    reviews = json["lq_tokens_list"]
    reviews = [join_tokens(review) for review in reviews]
    translated_summary = translator.translate(summary, target_language="en")["translatedText"]
    translated_reviews = translator.translate(reviews, target_language="en")
    translated_reviews = [review["translatedText"] for review in translated_reviews]
    return item_id, translated_reviews, translated_summary

def translate_reviews(review_jsons):
    ids, reviews, summaries = [], [], []
    translator = translate.Client()
    for json in tqdm(review_jsons):
        item_id, translated_reviews, translated_summary = translate_json(json, translator)
        ids.append(item_id)
        reviews.append(translated_reviews)
        summaries.append(translated_summary)
    result_df = pd.DataFrame({"id": ids, "reviews": reviews, "summary": summaries})
    result_df["review_str"] = result_df["reviews"].apply(lambda x: "|||||".join(x))
    result_df = result_df[["id", "summary", "review_str"]]
    return result_df

def main(args):
    review_jsons = load_jsons(args.reviews_file_path, int(args.start_line), int(args.stop_line))
    result_df = translate_reviews(review_jsons)
    result_df.to_csv(args.output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSARS dataset preprocessing')
    parser.add_argument('--reviews_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--start_line', type=str, help='File line where processing should start')
    parser.add_argument('--stop_line', type=str, help='File line where processing should stop')
    parser.add_argument('--output_file_path', type=str, help='Filepath to output the results')
    args = parser.parse_args()
    
    main(args)

