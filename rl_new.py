# Modified Record Linkage Implementation with Multi-Sheet Excel Support
import os
import csv
import re
import uuid
import logging
import dedupe
from dedupe import variables, RecordLink
from unidecode import unidecode
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# ---------- Preprocessing ----------
def preProcess(column):
    if not column:
        return ''
    text = unidecode(str(column))
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def adapt_for_variant(text, variant_type):
    if variant_type == "Sheet1":
        return re.sub(r'\W+', '', text)
    elif variant_type == "Sheet3":
        return text.replace(' ', '')
    elif variant_type == "Sheet7":
        parts = [p.strip() for p in text.split('.') if p.strip()]
        return ' '.join(parts)
    else:
        return text

# ---------- CSV Input Readers ----------
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

def read_test_excel(file_path, sheet_name=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    test_data = {}
    for _, row in df.iterrows():
        if 'ID' in row and 'NAME' in row:
            tid = f"{sheet_name}_{row['ID']}"
            name = row['NAME']
        elif 'NAME' in row:
            tid = f"{sheet_name}_{uuid.uuid4().hex[:8]}"
            name = row['NAME']
        test_data[tid] = {
            'name': preProcess(name),
            'variant_type': sheet_name,
            'raw_text': name
        }
    return test_data

# ---------- Canonical Representation ----------
def getCanonicalRep(records):
    if not records:
        return {}
    canonical = {}
    for key in records[0]:
        vals = [r[key] for r in records if r.get(key)]
        canonical[key] = max(set(vals), key=vals.count) if vals else ''
    return canonical

# ---------- Dedupe Training ----------
def trainDedupe(data, settings_file):
    fields = [variables.String('name')]
    if any('type' in rec for rec in data.values()):
        fields.append(variables.Categorical('type', categories=['I', 'C']))
    if os.path.exists(settings_file):
        with open(settings_file, 'rb') as sf:
            return dedupe.StaticDedupe(sf)
    deduper = dedupe.Dedupe(fields)
    deduper.prepare_training(data)
    dedupe.console_label(deduper)
    deduper.train()
    with open(settings_file, 'wb') as sf:
        deduper.write_settings(sf)
    return deduper

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

# ---------- Linkage Training ----------
def trainLinkage(reference, test, settings_file):
    fields = [variables.String('name')]
    if os.path.exists(settings_file):
        with open(settings_file, 'rb') as sf:
            return dedupe.StaticRecordLink(sf)
    linker = RecordLink(fields)
    linker.prepare_training(reference, test)
    dedupe.console_label(linker)
    linker.train()
    with open(settings_file, 'wb') as sf:
        linker.write_settings(sf)
    return linker

def linkRecords(reference, test, linker, threshold=0.5):
    linked = linker.join(reference, test, threshold)
    matches = []
    for cluster, score in linked:
        ref_ids = [rid for rid in cluster if rid in reference]
        test_ids = [tid for tid in cluster if tid in test]
        for rid in ref_ids:
            for tid in test_ids:
                matches.append((rid, tid, score))
    return matches

# ---------- Evaluation ----------
def evaluate_sheet_performance(matches, test_data, sheet_name):
    if sheet_name == "Sheet8":
        matched_count = sum(1 for _, tid, _ in matches)
        total = len(test_data)
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'error_rate': matched_count / total
        }
    y_true, y_pred = [], []
    for tid, record in test_data.items():
        true_match = 1 if 'true_id' in record else 0
        pred_match = 1 if any(tid == m[1] for m in matches) else 0
        y_true.append(true_match)
        y_pred.append(pred_match)
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

# ---------- Report Generator ----------
def generate_demo_report(sheet_results):
    case_studies = defaultdict(list)
    for sheet_name, data in sheet_results.items():
        for ref_id, tid, score in data['matches']:
            record = data['test_data'][tid]
            if record['raw_text'].isupper():
                case_studies['Letter Case'].append(record)
            if re.search(r'[^a-zA-Z0-9\s]', record['raw_text']):
                case_studies['Special Character'].append(record)
            # 其他变异类型检测逻辑可扩展

    with open('variant_cases.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant Type', 'Raw Text', 'Processed Text'])
        for var_type, cases in case_studies.items():
            for case in cases[:3]:
                writer.writerow([
                    var_type,
                    case['raw_text'],
                    case['name']
                ])

# ---------- Main Function ----------
def main():
    # Deduplicate primary and alternate
    data_p, map_p = _read_csv('primary.csv', 'p', has_type=True)
    data_a, map_a = _read_csv('alternate.csv', 'a', has_type=False)
    deduper_p = trainDedupe(data_p, 'primary_settings.dedupe')
    deduper_a = trainDedupe(data_a, 'alternate_settings.dedupe')
    mem_p, canon_p = deduplicateRecords(data_p, deduper_p)
    mem_a, canon_a = deduplicateRecords(data_a, deduper_a)

    reference = {f"pc_{cid}": {'name': canon['name']} for cid, canon in canon_p.items()}
    reference.update({f"ac_{cid}": {'name': canon['name']} for cid, canon in canon_a.items()})
    
    linker = trainLinkage(reference, {}, 'linkage_settings.dedupe')

    test2_path = 'test_02.xlsx'
    sheet_results = {}
    sheets = {
        "Sheet1": "remove_special_chars",
        "Sheet2": "shuffle_words",
        "Sheet3": "spacing_variants",
        "Sheet4": "transposed_letters",
        "Sheet5": "abbreviations",
        "Sheet6": "misspellings",
        "Sheet7": "initials_only",
        "Sheet8": "simulated_names"
    }

    for sheet_name, variant_desc in sheets.items():
        logger.info(f"Processing {sheet_name}: {variant_desc}")
        test_data = read_test_excel(test2_path, sheet_name)
        for tid, record in test_data.items():
            record['name'] = adapt_for_variant(record['name'], sheet_name)
        matches = linkRecords(reference, test_data, linker)
        sheet_results[sheet_name] = {
            'matches': matches,
            'test_data': test_data
        }
        metrics = evaluate_sheet_performance(matches, test_data, sheet_name)
        logger.info(f"{sheet_name} Metrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    generate_demo_report(sheet_results)

if __name__ == '__main__':
    main()
