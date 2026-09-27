import pandas as pd
import numpy as np
import os
import xgboost as xgb
import jellyfish
from tqdm import tqdm
import gc

def normalize_name(name_series):
    s = name_series.fillna("").str.lower()
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\b(inc|llc|llp|pvt|private|ltd|limited|corp|corporation|co|company)\b', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def extract_features(df):
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

def create_blocks(s1, s23):
    s1['sorted_name'] = s1['norm_name'].apply(lambda x: " ".join(sorted(x.split())))
    s23['sorted_name'] = s23['norm_name'].apply(lambda x: " ".join(sorted(x.split())))
    
    s1['no_space_name'] = s1['norm_name'].str.replace(' ', '')
    s23['no_space_name'] = s23['norm_name'].str.replace(' ', '')
    
    print("   -> Block 1: Exact Name")
    cand1 = s1.merge(s23, on='norm_name', suffixes=('_1', '_2'))
    
    print("   -> Block 2: Sorted Name")
    cand2 = s1[s1['sorted_name'] != ""].merge(s23[s23['sorted_name'] != ""], on='sorted_name', suffixes=('_1', '_2'))
    
    print("   -> Block 3: No Space Name")
    cand3 = s1[s1['no_space_name'] != ""].merge(s23[s23['no_space_name'] != ""], on='no_space_name', suffixes=('_1', '_2'))
    
    # Block 4: Exact Address (filter out extremely common addresses like "US" or empty)
    valid_s1_addr = s1[s1['business_address'].str.len() > 10].copy()
    valid_s23_addr = s23[s23['business_address'].str.len() > 10].copy()
    
    # Keep addresses that appear less than 50 times to prevent memory explosion
    addr_counts = valid_s23_addr['business_address'].value_counts()
    safe_addrs = addr_counts[addr_counts < 50].index
    valid_s23_addr = valid_s23_addr[valid_s23_addr['business_address'].isin(safe_addrs)]
    
    print("   -> Block 4: Safe Exact Address")
    cand4 = valid_s1_addr.merge(valid_s23_addr, on='business_address', suffixes=('_1', '_2'))
    
    candidates = pd.concat([cand1, cand2, cand3, cand4]).drop_duplicates(subset=['entity_id_1', 'entity_id_2'])
    return candidates

def main():
    data_dir = r"dataset"
    out_dir = r"output"
    os.makedirs(out_dir, exist_ok=True)
    
    usecols = ['entity_id', 'business_name', 'business_address', 'country']
    
    # 1. Load subset of training data for fast feature extraction
    print("Loading training data...")
    s1_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source1.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=100000)
    s2_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source2.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=300000)
    s3_tr = pd.read_csv(os.path.join(data_dir, 'train', 'train_source3.tsv'), sep='\t', usecols=usecols, dtype=str, nrows=300000)
    gt = pd.read_csv(os.path.join(data_dir, 'train', 'train_ground_truth.tsv'), sep='\t', dtype=str, nrows=100000)
    
    s1_tr['norm_name'] = normalize_name(s1_tr['business_name'])
    s2_tr['norm_name'] = normalize_name(s2_tr['business_name'])
    s3_tr['norm_name'] = normalize_name(s3_tr['business_name'])
    s23_tr = pd.concat([s2_tr, s3_tr])
    
    # Prepare Ground Truth Positive Pairs
    gt = gt.dropna()
    gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: x.split(','))
    pos_df = gt.explode('match_list').rename(columns={'match_list': 'matched_id'})
    pos_df = pos_df.merge(s1_tr, left_on='source1_entity_id', right_on='entity_id')
    pos_df = pos_df.merge(s23_tr, left_on='matched_id', right_on='entity_id', suffixes=('_1', '_2'))
    pos_df['target'] = 1
    
    # Prepare HARD Negative Pairs via Blocking
    print("Generating HARD Negatives for training...")
    train_candidates = create_blocks(s1_tr, s23_tr)
    # Remove true positives from candidates to get hard negatives
    merged_check = train_candidates.merge(pos_df[['entity_id_1', 'entity_id_2', 'target']], on=['entity_id_1', 'entity_id_2'], how='left')
    hard_negatives = merged_check[merged_check['target'].isna()].copy()
    hard_negatives['target'] = 0
    # Sample down to balance dataset
    hard_negatives = hard_negatives.sample(n=min(len(pos_df)*2, len(hard_negatives)), random_state=42)
    
    train_df = pd.concat([pos_df, hard_negatives]).sample(frac=1, random_state=42)
    
    print("Extracting features for training pairs...")
    X_train = extract_features(train_df)
    y_train = train_df['target'].values
    
    print("Training XGBoost on GPU...")
    clf = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=300, max_depth=7, learning_rate=0.05)
    clf.fit(X_train, y_train)
    print("Training Complete!")
    
    # Free memory
    del s1_tr, s2_tr, s3_tr, s23_tr, train_df, pos_df, hard_negatives, train_candidates, merged_check
    gc.collect()
    
    # --- INFERENCE ---
    print("\nLoading test data...")
    s1 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source1.tsv'), sep='\t', usecols=usecols, dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source2.tsv'), sep='\t', usecols=usecols, dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, 'test', 'test_source3.tsv'), sep='\t', usecols=usecols, dtype=str)
    
    s1['norm_name'] = normalize_name(s1['business_name'])
    s2['norm_name'] = normalize_name(s2['business_name'])
    s3['norm_name'] = normalize_name(s3['business_name'])
    s23_test = pd.concat([s2, s3])
    
    print("Blocking test candidates...")
    candidates = create_blocks(s1, s23_test)
    print(f"Total test candidates generated: {len(candidates)}")
    
    print("Extracting features for test candidates...")
    X_test = extract_features(candidates)
    
    print("Predicting matches...")
    preds = clf.predict_proba(X_test)[:, 1]
    candidates['prob'] = preds
    
    # F0.5 optimizes for Precision. We use a high threshold to minimize False Positives.
    matches = candidates[candidates['prob'] > 0.85]
    
    grouped = matches.groupby('entity_id_1')['entity_id_2'].apply(lambda x: ','.join(set(x))).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    final_out = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    final_out = final_out.merge(grouped, on='source1_entity_id', how='left').fillna('')
    final_out.to_csv(os.path.join(out_dir, 'matching_results.tsv'), sep='\t', index=False)
    
    cand_group = candidates.groupby('entity_id_1')['entity_id_2'].apply(lambda x: ','.join(set(x))).reset_index()
    cand_group.columns = ['source1_entity_id', 'candidate_entity_ids']
    cand_out = pd.DataFrame({'source1_entity_id': s1['entity_id']}).merge(cand_group, on='source1_entity_id', how='left').fillna('')
    cand_out.to_csv(os.path.join(out_dir, 'candidate_pairs.tsv'), sep='\t', index=False)
    
    print("Done! Final files generated in output/ directory.")

if __name__ == '__main__':
    main()
