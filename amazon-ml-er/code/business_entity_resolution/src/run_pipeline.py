import pandas as pd
import numpy as np
import os
import argparse
import re
from collections import defaultdict
import time

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    # Remove common legal suffixes
    name = re.sub(r'\b(inc|llc|llp|pvt|private|ltd|limited|corp|corporation|co|company)\b', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--work-dir', default='artifacts')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    
    test_dir = os.path.join(data_dir, 'test')
    
    print("Loading test data...")
    t0 = time.time()
    
    # We only read necessary columns to save memory
    usecols = ['entity_id', 'business_name', 'country']
    dtype = str
    
    s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    
    print(f"Data loaded in {time.time() - t0:.2f}s")
    
    print("Normalizing names...")
    t0 = time.time()
    s1['norm_name'] = s1['business_name'].apply(normalize_name)
    s2['norm_name'] = s2['business_name'].apply(normalize_name)
    s3['norm_name'] = s3['business_name'].apply(normalize_name)
    print(f"Normalization done in {time.time() - t0:.2f}s")
    
    print("Building inverted index for S2 and S3...")
    t0 = time.time()
    
    # Inverted index: (norm_name, country) -> list of entity_ids
    index = defaultdict(list)
    
    # Process S2
    for row in s2.itertuples(index=False):
        n = row.norm_name
        if n:  # skip empty
            index[(n, row.country)].append(row.entity_id)
            
    # Process S3
    for row in s3.itertuples(index=False):
        n = row.norm_name
        if n:
            index[(n, row.country)].append(row.entity_id)
            
    print(f"Index built in {time.time() - t0:.2f}s")
    
    print("Matching S1 entities...")
    t0 = time.time()
    
    matching_results = []
    
    for row in s1.itertuples(index=False):
        n = row.norm_name
        c = row.country
        matches = []
        if n:
            matches = index.get((n, c), [])
        
        # Deduplicate
        matches = list(set(matches))
        matching_results.append({
            'source1_entity_id': row.entity_id,
            'matched_entity_ids': ','.join(matches)
        })
        
    print(f"Matching done in {time.time() - t0:.2f}s")
    
    print("Writing outputs...")
    t0 = time.time()
    
    # candidate_pairs.tsv is exactly the same as matching_results for this baseline
    df_out = pd.DataFrame(matching_results)
    
    df_out.rename(columns={'matched_entity_ids': 'candidate_entity_ids'}, inplace=True)
    df_out.to_csv(os.path.join(out_dir, 'candidate_pairs.tsv'), sep='\t', index=False)
    
    df_out.rename(columns={'candidate_entity_ids': 'matched_entity_ids'}, inplace=True)
    df_out.to_csv(os.path.join(out_dir, 'matching_results.tsv'), sep='\t', index=False)
    
    print(f"Outputs written in {time.time() - t0:.2f}s")
    print(f"Pipeline complete! Outputs are in {out_dir}")

if __name__ == '__main__':
    main()
