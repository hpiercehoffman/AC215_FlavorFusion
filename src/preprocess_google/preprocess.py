import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json
import numpy as np
from utils import parse, RESTS

def make_metadata_df(fl):
    parser = parse(fl)
    rest_records = []
    print('Processing metadata')
    for record in parser:
        if record['category'] != None:
            if not set(record['category']).isdisjoint(RESTS):
                rest_records.append([record['name'],
                                     record['gmap_id'],
                                     record['address'],
                                     record['avg_rating'],
                                     record['relative_results'],
                                     record['num_of_reviews']])
    
    df = pd.DataFrame(rest_records, columns=['Name', 'gmap_id', 'address', 'avg_rating', 
                                             'relative_results', 'num_of_reviews'])
    return df

def make_reviews_df(fl, min_char=0, max_char=10000):
    parser = parse(fl)
    reviews = []
    print('Processing reviews data')
    for review in parser:
        if review['text'] != None:
            if len(review['text']) >= min_char and len(review['text']) < max_char:
                reviews.append([review['name'],
                                review['rating'],
                                review['text'],
                                review['gmap_id']
                               ])
    df = pd.DataFrame(reviews, columns=['name', 'rating', 'text', 'gmap_id'])
    return df

def main(args):
    reviews_df = make_reviews_df(args.reviews_file_path, min_char=args.min_char, max_char=args.max_char)
    meta_df = make_metadata_df(args.metadata_file_path)
    
    meta_df = meta_df[meta_df['num_of_reviews'] < args.max_num_reviews]
    combined_df = reviews_df.merge(meta_df, on="gmap_id", how="inner")
    
    sub_df = combined_df.loc[:,['text', 'Name', 'address']]
    sub_df = sub_df.groupby(["Name", "address"]).agg({"text": "|||||".join}).reset_index()

    sub_df.to_csv(args.output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Google Review dataset preprocessing')
    parser.add_argument('--reviews_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--metadata_file_path', type=str, help='Path of the business metadata file')
    parser.add_argument('--output_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--min_char', type=int, default=0, help='Minimum number of characters in each review')
    parser.add_argument('--max_char', type=int, default=1000000, help='Minimum number of characters in each review')
    parser.add_argument('--max_num_reviews', type=int, default=1000, help='Maximum number of reviews for each business')
    args = parser.parse_args()
    
    main(args)
