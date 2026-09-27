import pandas as pd
import numpy as np
import os
import argparse
import re
from collections import defaultdict
import time
import xgboost as xgb
import jellyfish
from tqdm import tqdm
from multiprocessing import Pool
import joblib

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\b(inc|llc|llp|pvt|private|ltd|limited|corp|corporation|co|company)\b', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_features(row):
    n1, a1, c1, n2, a2, c2 = row
    
    n1 = str(n1) if pd.notna(n1) else ""
    n2 = str(n2) if pd.notna(n2) else ""
    a1 = str(a1).lower() if pd.notna(a1) else ""
    a2 = str(a2).lower() if pd.notna(a2) else ""
    
    # Name features
    jw = jellyfish.jaro_winkler_similarity(n1, n2)
    lev = jellyfish.levenshtein_distance(n1, n2) / max(1, max(len(n1), len(n2)))
    
    # Address features
    t1 = set(a1.split())
    t2 = set(a2.split())
    inter = len(t1.intersection(t2))
    union = len(t1.union(t2))
    addr_jaccard = inter / union if union > 0 else 0
    
    # Match Country?
    country_match = 1 if c1 == c2 else 0
    
    return [jw, lev, addr_jaccard, country_match]

def build_candidates(s1, s2, s3):
    print("Building blocks based on normalized name...")
    # Very simple blocking: exact match on normalized name or first word of name
    index = defaultdict(list)
    for src in [s2, s3]:
        for row in tqdm(src.itertuples(index=False), total=len(src), desc="Indexing"):
            n = row.norm_name
            if n:
                index[n].append(row.entity_id)
                # First word blocking
                words = n.split()
                if words:
                    index[words[0]].append(row.entity_id)
                    
    candidates = defaultdict(list)
    for row in tqdm(s1.itertuples(index=False), total=len(s1), desc="Blocking"):
        n = row.norm_name
        if n:
            matches = index.get(n, [])
            words = n.split()
            if words:
                matches.extend(index.get(words[0], []))
            
            matches = list(set(matches))
            # Keep top K if too many to avoid OOM
            if len(matches) > 100:
                matches = matches[:100]
            candidates[row.entity_id] = matches
            
    return candidates

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # --- TRAINING PHASE ---
    print("Loading train data...")
    usecols = ['entity_id', 'business_name', 'business_address', 'country']
    train_dir = os.path.join(data_dir, 'train')
    
    s1_tr = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', usecols=usecols, dtype=str)
    s2_tr = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', usecols=usecols, dtype=str)
    s3_tr = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', usecols=usecols, dtype=str)
    gt = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    
    # For training, we need positive and negative pairs.
    # To keep memory low, we sample a subset.
    print("Preparing training pairs...")
    gt = gt.dropna()
    gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: x.split(','))
    
    pos_pairs = []
    # Just take 50,000 positive pairs
    count = 0
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for m_id in row['match_list']:
            pos_pairs.append((s1_id, m_id))
            count += 1
            if count > 50000: break
        if count > 50000: break
            
    # Lookup dictionaries for fast feature extraction
    s1_dict = s1_tr.set_index('entity_id').to_dict('index')
    s23_tr = pd.concat([s2_tr, s3_tr]).set_index('entity_id').to_dict('index')
    
    X_train = []
    y_train = []
    
    print("Extracting features for positive pairs...")
    for s1_id, m_id in tqdm(pos_pairs):
        r1 = s1_dict.get(s1_id)
        r2 = s23_tr.get(m_id)
        if r1 and r2:
            row = (r1['business_name'], r1['business_address'], r1['country'],
                   r2['business_name'], r2['business_address'], r2['country'])
            X_train.append(extract_features(row))
            y_train.append(1)
            
    # Add some negatives (random pairings)
    print("Adding negative pairs...")
    s1_ids = list(s1_dict.keys())
    s23_ids = list(s23_tr.keys())
    for _ in tqdm(range(50000)):
        s1_id = np.random.choice(s1_ids)
        m_id = np.random.choice(s23_ids)
        r1 = s1_dict.get(s1_id)
        r2 = s23_tr.get(m_id)
        if r1 and r2:
            row = (r1['business_name'], r1['business_address'], r1['country'],
                   r2['business_name'], r2['business_address'], r2['country'])
            X_train.append(extract_features(row))
            y_train.append(0)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print("Training XGBoost on RTX 3050...")
    clf = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    clf.fit(X_train, y_train)
    print("Training complete!")
    
    # Clean memory
    del s1_tr, s2_tr, s3_tr, s23_tr, s1_dict, gt
    
    # --- INFERENCE PHASE ---
    print("Loading test data...")
    test_dir = os.path.join(data_dir, 'test')
    s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', usecols=usecols, dtype=str)
    s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', usecols=usecols, dtype=str)
    s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', usecols=usecols, dtype=str)
    
    s1['norm_name'] = s1['business_name'].apply(normalize_name)
    s2['norm_name'] = s2['business_name'].apply(normalize_name)
    s3['norm_name'] = s3['business_name'].apply(normalize_name)
    
    candidates = build_candidates(s1, s2, s3)
    
    s23_test = pd.concat([s2, s3]).set_index('entity_id').to_dict('index')
    
    candidate_pairs_output = []
    matching_results_output = []
    
    print("Running inference...")
    for row in tqdm(s1.itertuples(index=False), total=len(s1), desc="Scoring"):
        s1_id = row.entity_id
        cands = candidates.get(s1_id, [])
        candidate_pairs_output.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ','.join(cands)})
        
        final_matches = []
        if cands:
            X_infer = []
            for m_id in cands:
                r2 = s23_test.get(m_id)
                if r2:
                    feat_row = (row.business_name, row.business_address, row.country,
                                r2['business_name'], r2['business_address'], r2['country'])
                    X_infer.append(extract_features(feat_row))
                else:
                    X_infer.append([0, 1, 0, 0]) # Default bad features
                    
            X_infer = np.array(X_infer)
            preds = clf.predict_proba(X_infer)[:, 1]
            
            # High threshold for Precision since F0.5 favors Precision!
            for m_id, prob in zip(cands, preds):
                if prob > 0.8:
                    final_matches.append(m_id)
                    
        matching_results_output.append({'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(final_matches)})
        
    pd.DataFrame(candidate_pairs_output).to_csv(os.path.join(out_dir, 'candidate_pairs.tsv'), sep='\t', index=False)
    pd.DataFrame(matching_results_output).to_csv(os.path.join(out_dir, 'matching_results.tsv'), sep='\t', index=False)
    print("Done!")

if __name__ == '__main__':
    main()
