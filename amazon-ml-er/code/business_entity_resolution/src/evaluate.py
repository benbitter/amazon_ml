import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import time

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\b(inc|llc|llp|pvt|private|ltd|limited|corp|corporation|co|company)\b', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def calculate_f05(pred_list, gt_list):
    # Both are lists of strings
    if not pred_list and not gt_list:
        return 1.0 # True negative (singleton) gets 1.0
    if not pred_list or not gt_list:
        return 0.0 # One is empty, other is not -> 0.0 precision or recall
    
    pred_set = set(pred_list)
    gt_set = set(gt_list)
    
    tp = len(pred_set.intersection(gt_set))
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    
    # F0.5 formula
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def main():
    data_dir = r"C:\amazon_ml_project\dataset\train"
    
    print("Loading train data...")
    t0 = time.time()
    
    usecols = ['entity_id', 'business_name', 'country']
    dtype = str
    
    s1 = pd.read_csv(os.path.join(data_dir, 'train_source1.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    s2 = pd.read_csv(os.path.join(data_dir, 'train_source2.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    s3 = pd.read_csv(os.path.join(data_dir, 'train_source3.tsv'), sep='\t', usecols=usecols, dtype=dtype)
    gt = pd.read_csv(os.path.join(data_dir, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    
    print(f"Data loaded in {time.time() - t0:.2f}s")
    
    print("Normalizing names...")
    s1['norm_name'] = s1['business_name'].apply(normalize_name)
    s2['norm_name'] = s2['business_name'].apply(normalize_name)
    s3['norm_name'] = s3['business_name'].apply(normalize_name)
    
    print("Building inverted index for S2 and S3...")
    index = defaultdict(list)
    
    for row in s2.itertuples(index=False):
        n = row.norm_name
        if n:
            index[(n, row.country)].append(row.entity_id)
            
    for row in s3.itertuples(index=False):
        n = row.norm_name
        if n:
            index[(n, row.country)].append(row.entity_id)
            
    print("Matching S1 entities and calculating F0.5...")
    
    # Create dictionary for fast ground truth lookup
    gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: str(x).split(',') if pd.notna(x) and str(x).strip() != '' else [])
    gt_dict = dict(zip(gt['source1_entity_id'], gt['match_list']))
    
    f05_scores = []
    
    for row in s1.itertuples(index=False):
        n = row.norm_name
        c = row.country
        matches = []
        if n:
            matches = index.get((n, c), [])
        
        matches = list(set(matches))
        
        # Look up ground truth
        gt_matches = gt_dict.get(row.entity_id, [])
        
        score = calculate_f05(matches, gt_matches)
        f05_scores.append(score)
        
    final_score = np.mean(f05_scores)
    
    print(f"\n======================================")
    print(f"Final Macro-Average F_0.5 Score: {final_score:.4f}")
    print(f"======================================\n")

if __name__ == '__main__':
    main()
