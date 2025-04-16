# Record Linkage Implementation (Task 1 Focus)
import os
import csv
import re
import logging
import dedupe
from unidecode import unidecode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# ---------- Core Functions ----------
def preProcess(column):
    """统一预处理逻辑"""
    if not column:
        return ''
    column = unidecode(str(column))
    column = re.sub(r'[^\w\s-]', '', column)  # 移除非字母数字字符
    column = re.sub(r'\s+', ' ', column).strip().lower()
    return column

def readData(primary_file, alternate_file):
    """读取并合并数据"""
    def _read_csv(file_path, id_prefix):
        data = {}
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {
                    'name': preProcess(row['NAME']),
                    'type': preProcess(row.get('TYPE', '')).upper() if 'TYPE' in row else None
                }
                data[f"{id_prefix}_{row['ID']}"] = clean_row
        return data
    
    return {
        **_read_csv(primary_file, 'p'),
        **_read_csv(alternate_file, 'a')
    }

def getCanonicalRep(records):
    """安全生成规范表示"""
    canonical = {}
    if not records:
        return canonical
    
    keys = records[0].keys()
    for key in keys:
        values = []
        for record in records:
            value = record.get(key)
            # 处理所有可能的空值情况
            if isinstance(value, str):
                clean_value = value.strip()
                if clean_value:
                    values.append(clean_value)
            elif value is not None:  # 处理非字符串类型（如数字）
                values.append(str(value))
        
        if values:
            try:
                canonical[key] = max(set(values), key=values.count)
            except:
                canonical[key] = values[0]
        else:
            canonical[key] = ''
    return canonical

# ---------- Dedupe Training ----------
def trainDedupe(data, settings_file, training_file):
    """训练去重模型"""
    fields = [
        {'field': 'name', 'type': 'String'},
        {'field': 'type', 'type': 'Categorical', 'categories': ['I', 'C']}
    ]
    
    if os.path.exists(settings_file):
        logger.info("Loading pre-trained model")
        with open(settings_file, 'rb') as sf:
            return dedupe.StaticDedupe(sf)
    else:
        deduper = dedupe.Dedupe(fields)
        deduper.prepare_training(data)
        
        logger.info("Starting active learning...")
        dedupe.console_label(deduper)
        
        logger.info("Training model...")
        deduper.train()
        
        with open(settings_file, 'wb') as sf:
            deduper.write_settings(sf)
        return deduper

# ---------- Deduplication Core ----------
def deduplicateRecords(data, deduper):
    """执行去重聚类"""
    logger.info("Clustering records...")
    clustered_dupes = deduper.partition(data, threshold=0.5)
    
    # 构建结果
    cluster_membership = {}
    canonical_reps = {}
    
    for cluster_id, records in enumerate(clustered_dupes):
        record_ids = records[0] if isinstance(records, tuple) else records
        cluster_records = [data[rid] for rid in record_ids]
        
        # 记录成员关系
        for rid in record_ids:
            cluster_membership[rid] = cluster_id
        
        # 生成规范表示
        canonical_reps[cluster_id] = getCanonicalRep(cluster_records)
    
    return cluster_membership, canonical_reps

# ---------- Main Workflow ----------
def main():
    # 文件路径
    primary_file = 'primary.csv'
    alternate_file = 'alternate.csv'
    settings_file = 'settings.dedupe'
    
    # 读取数据
    logger.info("Reading data...")
    data = readData(primary_file, alternate_file)
    
    # 训练/加载模型
    deduper = trainDedupe(data, settings_file, training_file='training.json')
    
    # 执行去重
    logger.info("Deduplicating...")
    cluster_membership, canonical_reps = deduplicateRecords(data, deduper)
    
    # 输出结果
    logger.info(f"原始记录数: {len(data)}")
    logger.info(f"发现唯一实体: {len(canonical_reps)}")
    
    # 保存聚类结果（示例）
    with open('clusters.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['RecordID', 'ClusterID', 'Canonical_Name'])
        for record_id, cluster_id in cluster_membership.items():
            writer.writerow([
                record_id,
                cluster_id,
                canonical_reps[cluster_id]['name']
            ])

if __name__ == '__main__':
    main()