# Record Linkage Implementation (Task 1 & 2 Focus)
import os
import csv
import re
import logging
import dedupe
from dedupe import variables, RecordLink
from unidecode import unidecode
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# ---------- Core Functions ----------
def preProcess(column):
    """Standardize text data for consistent comparison"""
    
    if not column:
        return ''  # Return empty string if value is None, null, or empty
    
    text = unidecode(str(column))  
    # Convert Unicode characters to closest ASCII representation (e.g., "Café" -> "Cafe")
    
    text = re.sub(r'[^\w\s-]', '', text)  
    # Remove all characters except letters, digits, underscores, whitespace, and hyphens
    
    text = re.sub(r'\s+', ' ', text).strip().lower()  
    # Replace multiple spaces with a single space, trim leading/trailing spaces, and convert to lowercase
    
    return text

# ---------- Read CSVs ----------
def _read_csv(file_path, id_prefix, has_type=False):
    data = {}
    original_id_map = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row['ID']
            key = f"{id_prefix}_{orig}"
            original_id_map[orig] = key
            rec = {'name': preProcess(row['NAME'])}
            if has_type:
                rec['type'] = preProcess(row['TYPE']).upper()
            data[key] = rec
    return data, original_id_map


def _read_test_csv(file_path):
    """
    Read simulated test names (Task 2), assume test ID == primary ID,
    and set true_id = 'p_<ID>' for evaluation.
    """
    data = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        fields = reader.fieldnames or []

        # 1) 自动检测变体名称字段
        if 'NAME' in fields:
            name_field = 'NAME'
        elif 'VARIANT' in fields:
            name_field = 'VARIANT'
        else:
            # 排除 'ID' 和 'TRUE_ID' 后，取第一个剩余字段
            name_field = [fn for fn in fields if fn.upper() not in ('ID', 'TRUE_ID')][0]

        # 2) 读取每一行，构造 test key 和真值标签
        for row in reader:
            tid = f"t_{row['ID']}"
            rec = {
                'name': preProcess(row[name_field]),
                # 假设 row['ID'] 就是 primary.csv 的 ID
                'true_id': f"p_{row['ID']}"
            }
            data[tid] = rec

    return data


# ---------- Canonical Representation ----------
def getCanonicalRep(records):
    if not records:
        return {}
    canonical = {}
    for key in records[0]:
        vals = [r[key] for r in records if r.get(key)]
        canonical[key] = max(set(vals), key=vals.count) if vals else ''
    return canonical

# ---------- Dedupe Training (Task 1) ----------
def trainDedupe(data, settings_file):
    fields = [variables.String('name')]
    if any('type' in rec for rec in data.values()):
        fields.append(variables.Categorical('type', categories=['I', 'C']))
    if os.path.exists(settings_file):
        logger.info("Loading pre-trained model %s", settings_file)
        with open(settings_file, 'rb') as sf:
            return dedupe.StaticDedupe(sf)
    deduper = dedupe.Dedupe(fields)
    deduper.prepare_training(data)
    logger.info("Start active learning...")
    dedupe.console_label(deduper)
    logger.info("Training model...")
    deduper.train()
    with open(settings_file, 'wb') as sf:
        deduper.write_settings(sf)
    return deduper

# ---------- Deduplication Core (Task 1) ----------
def deduplicateRecords(data, deduper, threshold=0.5):
    clustered = deduper.partition(data, threshold)
    membership, canon_reps = {}, {}
    for cid, block in enumerate(clustered):
        ids = block[0] if isinstance(block, tuple) else block
        recs = [data[i] for i in ids]
        for i in ids:
            membership[i] = cid
        canon_reps[cid] = getCanonicalRep(recs)
    return membership, canon_reps

# ---------- Record Linkage Training (Task 2) ----------
def trainLinkage(reference, test, settings_file):
    fields = [variables.String('name')]
    if os.path.exists(settings_file):
        logger.info("Loading pre-trained linkage model %s", settings_file)
        with open(settings_file, 'rb') as sf:
            return dedupe.StaticRecordLink(sf)
    linker = RecordLink(fields)
    linker.prepare_training(reference, test)
    logger.info("Start linkage active learning...")
    dedupe.console_label(linker)
    logger.info("Training linkage model...")
    linker.train()
    with open(settings_file, 'wb') as sf:
        linker.write_settings(sf)
    return linker

# ---------- Record Linking Core (Task 2) ----------
"""
def linkRecords(reference, test, linker, threshold=0.5):
    linked = linker.join(reference, test, threshold)
    matches = []
    for cluster, score in linked:
        # cluster: iterable of keys (refs and test)
        for rid in cluster:
            for tid in cluster:
                if not tid.startswith('t_') or not rid.startswith(('pc_','ac_')):
                    continue
                # swap rid<->tid to match (ref_id, test_id)
                matches.append((tid, rid, score))
    # After swap, matches tuples: (ref_id, test_id, score)
    return [(rid, tid, score) for (rid, tid, score) in matches]
"""
# ---------- Record Linking Core (Task 2) ----------
def linkRecords(reference, test, linker, threshold=0.5):
    """
    使用 join() 返回 cluster 和其对应 score，
    但对每个 cluster 内部，手动拆出 (ref_id, test_id, score) 三元组。
    """
    linked = linker.join(reference, test, threshold)
    matches = []

    for cluster, score in linked:
        ref_ids = [rid for rid in cluster if rid in reference]
        test_ids = [tid for tid in cluster if tid in test]

        for rid in ref_ids:
            for tid in test_ids:
                matches.append((rid, tid, score))  # 每一对配上这个簇的score

    return matches



# ---------- Main Workflow ----------
def main():
    # Task 1: Primary
    data_p, map_p = _read_csv('primary.csv', 'p', has_type=True)
    deduper_p = trainDedupe(data_p, 'primary_settings.dedupe')
    mem_p, canon_p = deduplicateRecords(data_p, deduper_p)
    # Task 1: Alternate
    data_a, map_a = _read_csv('alternate.csv', 'a', has_type=False)
    deduper_a = trainDedupe(data_a, 'alternate_settings.dedupe')
    mem_a, canon_a = deduplicateRecords(data_a, deduper_a)

    # Task 2: Test link
    data_t = _read_test_csv('test_01.csv')
    reference = {f"pc_{cid}": {'name': canon['name']} for cid, canon in canon_p.items()}
    reference.update({f"ac_{cid}": {'name': canon['name']} for cid, canon in canon_a.items()})

    linker = trainLinkage(reference, data_t, 'linkage_settings.dedupe')
    raw_matches = linkRecords(reference, data_t, linker)

    # Best match per test
    grouped = defaultdict(list)
    for ref, tid, score in raw_matches:
        grouped[tid].append((ref, score))
    best = [(max(lst, key=lambda x: x[1])[0], tid, max(lst, key=lambda x: x[1])[1]) for tid, lst in grouped.items()]

    # Build true map
    true_map = {}
    for tid, rec in data_t.items():
        if 'true_id' not in rec:
            continue
        orig = rec['true_id']
        prefix, oid = (orig.split('_',1) if '_' in orig else (None, orig))
        key = None
        if prefix == 'p': key = map_p.get(oid)
        if prefix == 'a': key = map_a.get(oid)
        if not key:
            key = map_p.get(orig) or map_a.get(orig)
        if key:
            if key.startswith('p_'):
                true_map[tid] = f"pc_{mem_p[key]}"
            else:
                true_map[tid] = f"ac_{mem_a[key]}"

    # Evaluate
    y_true, y_pred = [], []
    for ref, tid, _ in best:
        if tid in true_map:
            y_true.append(1)
            y_pred.append(1 if ref == true_map[tid] else 0)
    if y_true:
        logger.info("Precision: %.4f, Recall: %.4f, F1: %.4f", \
                    precision_score(y_true, y_pred), \
                    recall_score(y_true, y_pred), \
                    f1_score(y_true, y_pred))

    # Write output
    with open('test_mapping.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['TestID','RefClusterID','Score','IsCorrect'])
        for ref, tid, score in best:
            correct = 'N/A'
            if tid in true_map:
                correct = (ref == true_map[tid])
            w.writerow([tid, ref, score, correct])

if __name__ == '__main__':
    main()
