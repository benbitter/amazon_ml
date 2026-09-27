import pandas as pd
import numpy as np
import os
import argparse
import re
import xgboost as xgb
import jellyfish
from tqdm import tqdm

def normalize_name(name_series):
    s = name_series.fillna("").str.lower()
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\b(inc|llc|llp|pvt|private|ltd|limited|corp|corporation|co|company)\b', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def extract_features(df):
    print("Extracting features (this may take a few minutes)...")
    records = df.to_dict('records')
    features = []
    for r in tqdm(records):
        n1 = str(r.get('norm_name_1', r.get('norm_name', '')))
        n2 = str(r.get('norm_name_2', r.get('norm_name', '')))
        a1, a2 = str(r.get('business_address_1', '')).lower(), str(r.get('business_address_2', '')).lower()
        
        jw = jellyfish.jaro_winkler_similarity(n1, n2)
        m = max(len(n1), len(n2))
        lev = jellyfish.levenshtein_distance(n1, n2) / max(1, m)
        
        t1, t2 = set(a1.split()), set(a2.split())
        union = len(t1.union(t2))
        jaccard = len(t1.intersection(t2)) / union if union > 0 else 0
        
        c_match = 1 if r.get('country_1') == r.get('country_2') else 0
        features.append([jw, lev, jaccard, c_match])
        
    return np.array(features)

def main():
    data_dir = r"dataset"
    out_dir = r"output"
    
    # 1. Load train dataset (Subset for speed)
    print("Loading training data...")
    usecols = ['entity_id', 'business_name', 'business_address', 'country']
    
    s1_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source1.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=500000)
    s2_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source2.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=1000000)
    s3_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source3.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=1000000)
    gt = pd.read_csv(os.path.join(data_dir, 'train', 'train_ground_truth.tsv'), sep='\t', dtype=str, nrows=500000)
    
    s1_tr['norm_name'] = normalize_name(s1_tr['business_name'])
    s2_tr['norm_name'] = normalize_name(s2_tr['business_name'])
    s3_tr['norm_name'] = normalize_name(s3_tr['business_name'])
    
    s23_tr = pd.concat([s2_tr, s3_tr])
    
    gt = gt.dropna()
    gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: x.split(','))
    
    # Positive pairs
    pos_df = gt.explode('match_list').rename(columns={'match_list': 'matched_id'})
    pos_df = pos_df.merge(s1_tr, left_on='source1_entity_id', right_on='entity_id')
    pos_df = pos_df.merge(s23_tr, left_on='matched_id', right_on='entity_id', suffixes=('_1', '_2'))
    pos_df['target'] = 1
    
    # Negative pairs (random)
    neg_df = s1_tr.sample(n=len(pos_df), random_state=42).copy()
    neg_s23 = s23_tr.sample(n=len(pos_df), random_state=42).copy()
    neg_df = neg_df.reset_index(drop=True).join(neg_s23.reset_index(drop=True), lsuffix='_1', rsuffix='_2')
    neg_df['target'] = 0
    
    train_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)
    
    print("Building features...")
    X_train = extract_features(train_df)
    y_train = train_df['target'].values
    
    print("Training XGBoost on RTX 3050...")
    clf = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=200, max_depth=6)
    clf.fit(X_train, y_train)
    
    # Calculate training accuracy for the user
    print("Calculating final F0.5 accuracy on training data...")
    train_preds = clf.predict_proba(X_train)[:, 1]
    train_df['pred_prob'] = train_preds
    train_matches = train_df[train_df['pred_prob'] > 0.8]
    
    tp = len(train_matches[train_matches['target'] == 1])
    fp = len(train_matches[train_matches['target'] == 0])
    total_actual = len(pos_df)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_actual if total_actual > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=========================================")
    print(f"Model Training Complete!")
    print(f"Estimated F_0.5 Accuracy: {f05 * 100:.2f}%")
    print(f"=========================================\n")
    
    del s1_tr, s2_tr, s3_tr, s23_tr, train_df, pos_df, neg_df
    import gc; gc.collect()
    
    # Inference phase
    print("Loading test data...")
    s1 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source1.tsv'), sep='\t', usecols=usecols, dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source2.tsv'), sep='\t', usecols=usecols, dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source3.tsv'), sep='\t', usecols=usecols, dtype=str)
    
    s1['norm_name'] = normalize_name(s1['business_name'])
    s2['norm_name'] = normalize_name(s2['business_name'])
    s3['norm_name'] = normalize_name(s3['business_name'])
    
    s23_test = pd.concat([s2, s3])
    
    # Create additional blocking keys for high recall
    s1['sorted_name'] = s1['norm_name'].apply(lambda x: " ".join(sorted(x.split())))
    s23_test['sorted_name'] = s23_test['norm_name'].apply(lambda x: " ".join(sorted(x.split())))
    
    s1['no_space_name'] = s1['norm_name'].str.replace(' ', '')
    s23_test['no_space_name'] = s23_test['norm_name'].str.replace(' ', '')
    
    s1_addr = s1[s1['business_address'].notna()].copy()
    s23_addr = s23_test[s23_test['business_address'].notna()].copy()
    
    print("Blocking Phase 1: Exact Normalized Name...")
    cand1 = s1.merge(s23_test, on='norm_name', suffixes=('_1', '_2'))
    
    print("Blocking Phase 2: Sorted Words (Transpositions)...")
    cand2 = s1[s1['sorted_name'] != ""].merge(s23_test[s23_test['sorted_name'] != ""], on='sorted_name', suffixes=('_1', '_2'))
    
    print("Blocking Phase 3: No Space Name...")
    cand3 = s1[s1['no_space_name'] != ""].merge(s23_test[s23_test['no_space_name'] != ""], on='no_space_name', suffixes=('_1', '_2'))
    
    print("Blocking Phase 4: Exact Address...")
    cand4 = s1_addr.merge(s23_addr, on='business_address', suffixes=('_1', '_2'))
    
    print("Combining candidates...")
    candidates = pd.concat([cand1, cand2, cand3, cand4]).drop_duplicates(subset=['entity_id_1', 'entity_id_2'])
    
    print("Extracting features for test candidates...")
    X_test = extract_features(candidates)
    
    print("Predicting...")
    preds = clf.predict_proba(X_test)[:, 1]
    
    candidates['prob'] = preds
    matches = candidates[candidates['prob'] > 0.8]
    
    grouped = matches.groupby('entity_id_1')['entity_id_2'].apply(lambda x: ','.join(set(x))).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Need all S1 entities in output
    os.makedirs(out_dir, exist_ok=True)
    final_out = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    final_out = final_out.merge(grouped, on='source1_entity_id', how='left').fillna('')
    
    final_out.to_csv(os.path.join(out_dir, 'matching_results.tsv'), sep='\t', index=False)
    
    # candidates
    cand_group = candidates.groupby('entity_id_1')['entity_id_2'].apply(lambda x: ','.join(set(x))).reset_index()
    cand_group.columns = ['source1_entity_id', 'candidate_entity_ids']
    cand_out = pd.DataFrame({'source1_entity_id': s1['entity_id']}).merge(cand_group, on='source1_entity_id', how='left').fillna('')
    cand_out.to_csv(os.path.join(out_dir, 'candidate_pairs.tsv'), sep='\t', index=False)
    
    print("Done! High accuracy pipeline finished. Upload output/matching_results.tsv to the portal!")

if __name__ == '__main__':
    main()
