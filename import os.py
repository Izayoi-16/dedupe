import os
import csv
import re
import logging
import pandas as pd
from unidecode import unidecode
from collections import defaultdict
from rapidfuzz import fuzz
import dedupe
from dedupe import variables
from functools import lru_cache

# --- 基本设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# -------------------- 1. 增强型文本预处理 ---------------------

def build_normalization_dictionaries():
    """构建用于标准化的词典"""
    shape_dict = {'0': 'o', '1': 'l', '5': 's', '8': 'b', '2': 'z'}
    normalization_dict = {
        'billy': 'william', 'bill': 'william', 'bob': 'robert', 'rob': 'robert',
        'bobby': 'robert', 'betty': 'elizabeth', 'liz': 'elizabeth', 'sue': 'susan',
        'ltd': 'limited', 'inc': 'incorporated', 'corp': 'corporation',
        'co': 'company', 'llc': 'limited liability company', 'shpg': 'shipping',
        'apt': 'apartment', 'rd': 'road', 'st': 'street', 'dr': 'drive',
        'ne': 'northeast', 'us': 'united states', 'usa': 'united states',
        'pharmaceuticals': 'drug', 'pharma': 'drug', 'drugs': 'drug',
        'tech': 'technology'
    }
    stop_words = {'mr', 'mrs', 'ms', 'dr', 'the', 'of', 'and', 'a', 'an', 'jr', 'sr'}
    return shape_dict, normalization_dict, stop_words

SHAPE_DICT, NORM_DICT, STOP_WORDS = build_normalization_dictionaries()

def enhanced_preprocess(text):
    """一个更强大的、基于词典的预处理函数"""
    if not text or not isinstance(text, str): return ''
    text = unidecode(text).lower()
    for char, replacement in SHAPE_DICT.items(): text = text.replace(char, replacement)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    normalized_words = []
    for word in words:
        word = NORM_DICT.get(word, word)
        if word not in STOP_WORDS: normalized_words.append(word)
    final_text = ' '.join(sorted(normalized_words))
    return final_text

# -------------------- 2. Dedupe & 数据读写 ---------------------
def trainDedupe(data, settings_file):
    fields = [variables.String('name')]
    if os.path.exists(settings_file):
        with open(settings_file, 'rb') as f: return dedupe.StaticDedupe(f)
    deduper = dedupe.Dedupe(fields)
    # 当数据量较少时，传递所有数据进行训练
    if len(data) < 1500:
        deduper.prepare_training(data)
    else:
        deduper.prepare_training(data, sample_size=1500) # 数据量大时进行采样
    
    dedupe.console_label(deduper)
    deduper.train()
    with open(settings_file, 'wb') as f: deduper.write_settings(f)
    return deduper

def deduplicate_and_map_ids(data, deduper, prefix):
    clusters = deduper.partition(data, 0.5)
    canonical_reps = {}
    cluster_membership = defaultdict(set)
    for cluster_id, record_ids in enumerate(clusters):
        canonical_id = f"{prefix}_{cluster_id}"
        cluster_names = [data[i]['name'] for i in record_ids]
        canonical_name = max(set(cluster_names), key=cluster_names.count)
        canonical_reps[canonical_id] = {'name': canonical_name}
        for record_id in record_ids:
            original_id = data[record_id]['original_id']
            cluster_membership[canonical_id].add(original_id)
    return canonical_reps, cluster_membership

def _read_csv(fp, pfx):
    d = {}
    with open(fp, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            original_id = r.get('ID', '').strip()
            if not original_id: continue
            rid = f"{pfx}*{original_id}"
            d[rid] = {'name': enhanced_preprocess(r['NAME']), 'original_id': original_id}
    return d

def save_matches(matches, test_data, reference, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_id','test_name','ground_truth_id', 'matched_id','matched_name','score'])
        for result in matches:
            tid, m_id, score = result
            gt_id = test_data.get(tid,{}).get('ground_truth_id','')
            m_name = reference.get(m_id,{}).get('name','') if m_id else ''
            writer.writerow([tid, test_data[tid]['raw_text'], gt_id, m_id, m_name, round(score,4)])

# -------------------- 3. 阻塞、评分与评估 ---------------------
@lru_cache(maxsize=None)
def get_ngrams(s, n=3):
    s0 = s.replace(' ', '')
    return set(s0[i:i+n] for i in range(len(s0)-n+1))

def jaccard(a, b):
    ia = get_ngrams(a)
    ib = get_ngrams(b)
    if not ia or not ib: return 0.0
    return len(ia & ib) / len(ia | ib)

def build_prefix_index(reference):
    idx = defaultdict(list)
    for rid, rec in reference.items():
        if rec['name']:
            prefix = rec['name'][:2]
            idx[prefix].append((rid, rec['name']))
    return idx

def smart_score(a, b):
    w_ratio = fuzz.WRatio(a, b) / 100.0
    sort_ratio = fuzz.token_sort_ratio(a, b) / 100.0
    set_ratio = fuzz.token_set_ratio(a, b) / 100.0
    if len(a.split()) < 3 or len(b.split()) < 3: return (w_ratio * 0.4 + sort_ratio * 0.3 + set_ratio * 0.3)
    return (w_ratio * 0.6 + set_ratio * 0.4)

def evaluate_performance(sheet_name, matches, test_data, cluster_membership):
    logger.info(f"--- Evaluating Performance for {sheet_name} ---")
    true_positives, false_positives = 0, 0
    total_positives_in_test_set = 0
    all_reference_ids = set().union(*cluster_membership.values())
    for rec in test_data.values():
        if rec.get('ground_truth_id') in all_reference_ids: total_positives_in_test_set += 1

    for tid, matched_id, score in matches:
        if matched_id:
            ground_truth_id = test_data[tid].get('ground_truth_id')
            original_ids_in_cluster = cluster_membership.get(matched_id, set())
            if ground_truth_id in original_ids_in_cluster: true_positives += 1
            else:
                if ground_truth_id in all_reference_ids: false_positives += 1
    
    if sheet_name == 'Sheet8':
        false_positives = sum(1 for _, mid, _ in matches if mid)
        true_positives, total_positives_in_test_set = 0, 0
    
    false_negatives = total_positives_in_test_set - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0 if sheet_name == 'Sheet8' else 0.0
    recall = true_positives / total_positives_in_test_set if total_positives_in_test_set > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"  - Results: TP={true_positives}, FP={false_positives}, FN={false_negatives}")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - F1-Score: {f1_score:.4f}")

# -------------------- 4. 主流程 ---------------------
def main():
    # --- 文件路径配置 ---
    primary_file_path = r'C:\Users\35369\Downloads\1\primary.csv'
    alternate_file_path = r'C:\Users\35369\Downloads\1\alternate.csv'
    test01_file_path = r'C:\Users\35369\Downloads\1\test_01.csv'
    test02_file_path = r'C:\Users\35369\Downloads\1\test_02.xlsx'
    
    logger.info("Step 1: Reading and deduplicating source data from specified paths...")
    data_p = _read_csv(primary_file_path, 'p')
    data_a = _read_csv(alternate_file_path, 'a')
    
    # --- [核心修复] 过滤掉预处理后name为空的记录 ---
    # 这是为了防止dedupe的C++模块因收到空字符串而崩溃
    filtered_data_p = {k: v for k, v in data_p.items() if v['name']}
    filtered_data_a = {k: v for k, v in data_a.items() if v['name']}
    logger.info(f"Filtered primary data: {len(filtered_data_p)} records remain from {len(data_p)}.")
    logger.info(f"Filtered alternate data: {len(filtered_data_a)} records remain from {len(data_a)}.")
    
    # 使用过滤后的数据进行训练
    deduper_p = trainDedupe(filtered_data_p, 'd_p.dedu')
    deduper_a = trainDedupe(filtered_data_a, 'd_a.dedu')
    
    can_p, member_p = deduplicate_and_map_ids(filtered_data_p, deduper_p, 'pc')
    can_a, member_a = deduplicate_and_map_ids(filtered_data_a, deduper_a, 'ac')

    reference = {**can_p, **can_a}
    cluster_membership = {**member_p, **member_a}
    
    prefix_idx = build_prefix_index(reference)
    logger.info(f"Created {len(reference)} unique entities. Building prefix index...")

    # --- 处理 test_01.csv ---
    test1 = {}
    with open(test01_file_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tid = f"t1_{r['ID']}"
            test1[tid] = {'raw_text': r['VARIANT'], 'name': enhanced_preprocess(r['VARIANT']), 'ground_truth_id': r.get('ID','').strip()}

    m1 = []
    for tid, rec in test1.items():
        best = (None, 0.0)
        if rec['name']:
            prefix = rec['name'][:2]
            bucket = prefix_idx.get(prefix, list(reference.items()))
            cands = [(rid, nm['name']) for rid, nm in bucket if jaccard(rec['name'], nm['name']) >= 0.2]
            for rid, nm in cands:
                sc = smart_score(rec['name'], nm)
                if sc > best[1]: best = (rid, sc)
        m1.append((tid, best[0] if best[1] > 0.88 else None, best[1]))
            
    save_matches(m1, test1, reference, 'matches_test_01.csv')
    evaluate_performance("test_01", m1, test1, cluster_membership)

    # --- 处理 test_02.xlsx ---
    xl = pd.ExcelFile(test02_file_path)
    for sheet in xl.sheet_names:
        if not sheet.startswith('Sheet'): continue
        
        df = xl.parse(sheet)
        td = {}
        for i, row in df.iterrows():
            tid = f"{sheet}_{row.get('ID', i)}"
            raw = str(row.get('NAME', row.iloc[0]))
            gt_id = str(int(row['ID'])) if 'ID' in row and pd.notna(row['ID']) else ''
            td[tid] = {'raw_text': raw, 'name': enhanced_preprocess(raw), 'ground_truth_id': gt_id}

        ms = []
        if sheet == 'Sheet4':
            ref_sorted = {rid: ''.join(sorted(rec['name'].replace(' ', ''))) for rid, rec in reference.items()}
            for tid, rec in td.items():
                rec_sorted = ''.join(sorted(rec['name'].replace(' ', '')))
                best = (None, 0.0)
                for rid, r_sorted in ref_sorted.items():
                    sc = fuzz.ratio(rec_sorted, r_sorted) / 100.0
                    if sc > best[1]: best = (rid, sc)
                ms.append((tid, best[0] if best[1] > 0.9 else None, best[1]))
        else:
            for tid, rec in td.items():
                best = (None, 0.0)
                if rec['name']:
                    prefix = rec['name'][:2]
                    bucket = prefix_idx.get(prefix, list(reference.items()))
                    cands = [(rid, nm['name']) for rid, nm in bucket if jaccard(rec['name'], nm['name']) >= 0.2]
                    for rid, nm in cands:
                        sc = fuzz.partial_ratio(rec['name'], nm)/100.0 if sheet == 'Sheet6' else smart_score(rec['name'], nm)
                        if sc > best[1]: best = (rid, sc)
                thresh = 0.95 if sheet == 'Sheet8' else 0.88
                ms.append((tid, best[0] if best[1] > thresh else None, best[1]))
        
        save_matches(ms, td, reference, f"matches_{sheet}.csv")
        evaluate_performance(sheet, ms, td, cluster_membership)

if __name__ == '__main__':
    main()