# Record Linkage Implementation
# Based on the dedupe library for matching primary and alternate name files

import os
import csv
import re
import logging
import optparse
import numpy as np
import pandas as pd
from unidecode import unidecode

import dedupe
import dedupe.variables


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 自定义实现 getCanonicalRep 函数，使用多数投票法生成规范表示
def getCanonicalRep(records):
    """
    使用多数投票法生成一组记录的规范化表示
    Args:
        records: list of dict，每个 dict 表示一条记录
    Returns:
        canonical: dict，包含聚类中每个字段的标准值
    """
    canonical = {}
    if not records:
        return canonical

    keys = records[0].keys()
    for key in keys:
        values = []
        for record in records:
            # 确保获取的值不是None，并转换为空字符串
            value = record.get(key, '') or ''
            # 仅当处理后的值非空时加入列表
            if value.strip():
                values.append(value)
        # 如果存在有效值，取众数；否则设为空字符串
        if values:
            canonical[key] = max(set(values), key=values.count)
        else:
            canonical[key] = ''
    return canonical

# Helper functions for data preprocessing
def preProcess(column):
    """
    Clean and standardize string data for matching
    """
    if not column:
        return ''
    
    column = unidecode(column)  # Convert to ASCII
    column = re.sub('\n', ' ', column)
    column = re.sub('-', '', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(",", '', column)
    column = re.sub(":", ' ', column)
    column = re.sub('  +', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    
    if not column:
        return ''
    return column

def readData(primary_file, alternate_file):
    """
    Read data from primary and alternate files and prepare for processing
    
    Args:
        primary_file: Path to primary names file
        alternate_file: Path to alternate names file
    
    Returns:
        Dictionary of records for deduplication and linkage
    """
    # Read primary file
    primary_data = {}
    with open(primary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: preProcess(v) for k, v in row.items()}
            primary_data[row['ID']] = dict(
                name=clean_row['NAME'], 
                type=clean_row['TYPE'],
                source='primary'
            )
    
    # Read alternate file
    alternate_data = {}
    with open(alternate_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: preProcess(v) for k, v in row.items()}
            alternate_data[row['ID']] = dict(
                name=clean_row['NAME'],
                source='alternate'
            )
    
    # Combine into a single dictionary with unique IDs
    combined_data = {}
    for id_val, record in primary_data.items():
        combined_data[f"p_{id_val}"] = record
    
    for id_val, record in alternate_data.items():
        combined_data[f"a_{id_val}"] = record
    
    return combined_data

def readTestData(test_file):
    """
    Read simulated test data from test_01.csv, which uses the 'VARIANT' column.
    """
    test_data = {}
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 使用 'VARIANT' 列而非 'NAME'
            clean_value = preProcess(row['VARIANT'])
            test_data[f"t_{i}"] = dict(
                name=clean_value,
                source='test'
            )
    
    return test_data

def trainDedupe(data, settings_file, training_file):
    """
    Train dedupe with active learning
    """
    
    # 数据预处理：将 type 字段的所有值转换为大写，并移除无效值
    for record_id, record in data.items():
        if 'type' in record:
            # 确保 'type' 字段为大写
            record['type'] = record['type'].upper() if record['type'] else None
            # 只保留有效值，其他值可以选择设置为缺失或默认
            if record['type'] not in ['I', 'C']:
                record['type'] = None
    
    fields = [
        dedupe.variables.String('name'),
        dedupe.variables.Categorical('type', categories=['I', 'C'], has_missing=True)
    ]
    
    if os.path.exists(settings_file):
        logger.info(f"Loading settings from {settings_file}")
        with open(settings_file, 'rb') as sf:
            linker = dedupe.StaticDedupe(sf)
    else:
        logger.info("Creating new deduper")
        linker = dedupe.Dedupe(fields)
        
        # 准备训练数据
        logger.info("Preparing training data...")
        
        # 如果存在训练文件，加载它
        if os.path.exists(training_file):
            logger.info(f"Loading training from {training_file}")
            with open(training_file, 'rb') as tf:
                linker.prepare_training(data, training_file=tf)
        else:
            # 准备训练，但不标记任何例子
            linker.prepare_training(data)
            
            # 使用控制台标记功能（在本地终端使用）
            # 这将允许用户手动标记一些例子
            logger.info("Starting console labeling...")
            dedupe.console_label(linker)
        
        # 训练模型
        logger.info("Training model...")
        linker.train(recall=0.9)
        
        # 保存设置和训练数据
        with open(settings_file, 'wb') as sf:
            linker.write_settings(sf)
        
        with open(training_file, 'w', newline='', encoding='utf-8') as tf:
            linker.write_training(tf)
    
    return linker



import numpy as np

import numpy as np

def deduplicateRecords(data, linker):
    """
    Deduplicate records within the same dataset
    """
    logger.info("Finding clusters...")
    clustered_dupes = linker.partition(data, threshold=0.5)

    cluster_membership = {}
    canonical_reps = {}

    for cluster_id, cluster in enumerate(clustered_dupes):
        # Detect if cluster is a tuple of (ids, something)
        if isinstance(cluster, tuple) and len(cluster) >= 1 and isinstance(cluster[0], (list, tuple, set, np.ndarray)):
            record_ids = cluster[0]
        else:
            record_ids = cluster

        records = []
        for rid in record_ids:
            # Convert numpy types to native Python
            if isinstance(rid, np.generic):
                rid_key = rid.item()
            else:
                rid_key = rid
            # rid_key should now be a string matching data keys
            cluster_membership[rid_key] = cluster_id
            # append actual record
            records.append(data[rid_key])

        # Compute canonical representation
        canonical_reps[cluster_id] = getCanonicalRep(records)

    return clustered_dupes, cluster_membership, canonical_reps


def linkTestRecords(test_data, canonical_data, linker):
    """
    Link test records to existing canonical records
    """
    # Create a dictionary for results
    results = {}
    
    # For each test record, find matches in canonical data
    for test_id, test_record in test_data.items():
        # Get potential matches
        potential_matches = linker.match(test_record, canonical_data, threshold=0.5)
        
        # Store results
        if potential_matches:
            # Get the best match and its score
            best_match_id, score = potential_matches[0]
            results[test_id] = {
                'test_name': test_record['name'],
                'best_match_id': best_match_id,
                'best_match_name': canonical_data[best_match_id]['name'],
                'match': 'Y' if score > 0.7 else 'N',  # Adjustable threshold
                'score': score
            }
        else:
            results[test_id] = {
                'test_name': test_record['name'],
                'best_match_id': None,
                'best_match_name': None,
                'match': 'N',
                'score': 0.0
            }
    
    return results

def evaluatePerformance(results, ground_truth=None):
    """
    Evaluate model performance using data mining metrics
    
    Args:
        results: Dictionary with matching results
        ground_truth: Optional dictionary with known correct matches
    
    Returns:
        Dictionary with performance metrics
    """
    if ground_truth is None:
        logger.warning("No ground truth provided for evaluation")
        return {
            "precision": None,
            "recall": None,
            "f1_score": None
        }
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for test_id, result in results.items():
        actual_match = ground_truth.get(test_id, {'match': 'N'})['match']
        predicted_match = result['match']
        
        if predicted_match == 'Y' and actual_match == 'Y':
            true_positives += 1
        elif predicted_match == 'Y' and actual_match == 'N':
            false_positives += 1
        elif predicted_match == 'N' and actual_match == 'Y':
            false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def main():
    # File paths
    primary_file = 'primary.csv'
    alternate_file = 'alternate.csv'
    test_file = 'test_01.csv'
    output_file = 'results.csv'
    settings_file = 'settings.dedupe'
    training_file = 'training.json'
    
    # Read data
    logger.info("Reading data files...")
    data = readData(primary_file, alternate_file)
    test_data = readTestData(test_file)
    # 确保每条记录都有 'type' 字段
# 确保每条记录都有 'type' 字段且值为大写或None
    for record in data.values():
        if 'type' not in record:
            record['type'] = None
        elif record['type']:
            record['type'] = record['type'].upper()
            # 如果不是有效值，则设为None
            if record['type'] not in ['I', 'C']:
                record['type'] = None

    
    # Train model or load saved settings
    linker = trainDedupe(data, settings_file, training_file)
    
    # Task 1: Deduplicate the records in primary and alternate name files
    logger.info("Deduplicating records...")
    clustered_dupes, cluster_membership, canonical_reps = deduplicateRecords(data, linker)
    
    # Create a canonical representation for each cluster
    canonical_data = {}
    for cluster_id, canonical_rep in canonical_reps.items():
        canonical_data[f"c_{cluster_id}"] = canonical_rep
    
    # Print deduplication results
    logger.info(f"Found {len(canonical_reps)} distinct entities from {len(data)} input records")
    
    # Task 2: Map test names to records in original files
    logger.info("Mapping test records to canonical records...")
    mapping_results = linkTestRecords(test_data, canonical_data, linker)
    
    # Print match results
    for test_id, result in mapping_results.items():
        if result['match'] == 'Y':
            logger.info(f"Matched '{result['test_name']}' to '{result['best_match_name']}' with score {result['score']:.4f}")
    
    # Save results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['test_id', 'test_name', 'match', 'match_id', 'match_name', 'score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for test_id, result in mapping_results.items():
            writer.writerow({
                'test_id': test_id,
                'test_name': result['test_name'],
                'match': result['match'],
                'match_id': result['best_match_id'],
                'match_name': result['best_match_name'],
                'score': result['score']
            })
    
    # Example evaluation - in real scenario you'd need ground truth data
    # This is a placeholder for demo purposes
    mock_ground_truth = {test_id: {'match': result['match']} for test_id, result in mapping_results.items()}
    metrics = evaluatePerformance(mapping_results, mock_ground_truth)
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")

if __name__ == '__main__':
    main()
