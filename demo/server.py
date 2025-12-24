#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from fairseq.models.roberta import RobertaModel

TASK_FUNCBOUND = 'funcbound'
TASK_INSTBOUND = 'instbound'
SUPPORTED_TASKS = {TASK_FUNCBOUND, TASK_INSTBOUND}

FUNC_LABEL_FALLBACK = {
    '-': 0,
    'R': 1,
    'F': 2,
}

INST_LABEL_FALLBACK = {
    'B': 0,
    'S': 1,
    '-': 2,
    'N': 2,
}

TASK_LABEL_TOKENS = {
    TASK_FUNCBOUND: {
        'start': ['F'],
        'end': ['R'],
        'none': ['-'],
    },
    TASK_INSTBOUND: {
        'start': ['S'],
        'end': [],
        'none': ['-', 'N'],
    },
}

MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
MODEL_CACHE = {}
MODEL_DEVICE = {}

LABEL_MAP_CACHE = {}
SAMPLE_CACHE = {}
SAMPLE_INDEX = {}
VALIDATION_CACHE = {}
VALIDATION_LOCK = threading.Lock()
VALIDATION_JOB = {}
SHOWCASE_CACHE = {}
SHOWCASE_LOCK = threading.Lock()
CACHE_FILE = os.path.join(os.path.dirname(__file__), '.recommend_cache.json')
VALIDATION_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.validation_cache.json')

def load_persistent_cache():
    global SHOWCASE_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                SHOWCASE_CACHE = json.load(f)
            print(f"Loaded persistent recommendation cache from {CACHE_FILE}")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            SHOWCASE_CACHE = {}

def save_persistent_cache():
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(SHOWCASE_CACHE, f, indent=2)
    except Exception as e:
        print(f"Failed to save cache: {e}")

def load_validation_cache():
    global VALIDATION_CACHE
    if os.path.exists(VALIDATION_CACHE_FILE):
        try:
            with open(VALIDATION_CACHE_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict) and 'tasks' in raw:
                VALIDATION_CACHE = raw.get('tasks', {}) or {}
            elif isinstance(raw, dict):
                VALIDATION_CACHE = raw
            else:
                VALIDATION_CACHE = {}
            print(f"Loaded persistent validation cache from {VALIDATION_CACHE_FILE}")
        except Exception as e:
            print(f"Failed to load validation cache: {e}")
            VALIDATION_CACHE = {}

def save_validation_cache():
    try:
        payload = {
            'version': 1,
            'tasks': VALIDATION_CACHE
        }
        with open(VALIDATION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Failed to save validation cache: {e}")

# Initial load
load_persistent_cache()
load_validation_cache()


def get_arch_from_name(name):
    if 'x64' in name:
        return 'x64'
    if 'x86' in name:
        return 'x86'
    return 'x64'


import re

def disassemble_tokens(tokens, arch):
    if not tokens:
        return []
    
    try:
        # Convert hex tokens to bytes
        # Pad tokens to ensure they are 2 chars (byte) e.g. '5' -> '05'
        padded_tokens = [t.zfill(2) for t in tokens]
        joined = ''.join(padded_tokens)
        
        if len(joined) % 2 != 0:
             return [{'text': f"Error: Odd length hex string ({len(joined)} chars).", 'offset': 0, 'bytes': []}]
        
        data = bytes.fromhex(joined)
    except ValueError as e:
        return [{'text': f"Error: Invalid hex tokens: {str(e)}", 'offset': 0, 'bytes': []}]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        args = ['objdump', '-D', '-b', 'binary']
        if arch == 'x64':
            args.extend(['-m', 'i386', '-M', 'x86-64'])
        else:
            args.extend(['-m', 'i386'])
        
        args.append(tmp_path)
        
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            return [{'text': f"Error running objdump: {result.stderr}", 'offset': 0, 'bytes': []}]
        
        lines = result.stdout.splitlines()
        instructions = []
        started = False
        
        # Regex to parse objdump line:  10:	48 89 e5             	mov    %rsp,%rbp
        # Group 1: Offset (hex)
        # Group 2: Bytes (hex, space separated)
        # Group 3: Instruction text
        line_re = re.compile(r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F ]+)\s+(.*)$')

        for line in lines:
            if '<.data>:' in line:
                started = True
                continue
            if not started:
                continue
            
            line = line.strip()
            if not line:
                continue

            match = line_re.match(line)
            if match:
                offset_str = match.group(1)
                bytes_str = match.group(2).strip()
                instr_text = match.group(3).strip()
                
                # Parse offset
                try:
                    offset = int(offset_str, 16)
                except ValueError:
                    offset = -1
                
                # Parse bytes count to know coverage
                byte_count = len(bytes_str.replace(' ', '')) // 2
                
                instructions.append({
                    'offset': offset,
                    'size': byte_count,
                    'bytes': bytes_str,
                    'text': instr_text
                })
            else:
                # Handle lines that might be continuations or weird formatting
                # For now, append as raw text if it looks like content
                if '...' in line: # objdump sometimes skips
                     instructions.append({'text': line, 'offset': -1, 'size': 0, 'bytes': ''})

        return instructions

    except Exception as e:
        return [{'text': f"Error: {str(e)}", 'offset': 0, 'bytes': []}]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class TaskConfig:
    def __init__(
        self,
        name,
        root_dir,
        checkpoint_dir,
        checkpoint_file,
        data_bin,
        data_root,
        user_dir,
        device,
        max_tokens,
        truth_subdir=None,
        ida_subdir=None,
    ):
        self.name = name
        self.head = name
        self.checkpoint_dir = resolve_path(root_dir, checkpoint_dir)
        self.checkpoint_file = checkpoint_file
        self.data_bin = resolve_path(root_dir, data_bin)
        self.data_root = resolve_path(root_dir, data_root)
        self.user_dir = resolve_path(root_dir, user_dir)
        self.device = device
        self.max_tokens = max_tokens
        self.truth_subdir = truth_subdir
        self.ida_subdir = ida_subdir


class Config:
    def __init__(self, root_dir, args):
        self.root_dir = root_dir
        self.tasks = {
            TASK_FUNCBOUND: TaskConfig(
                TASK_FUNCBOUND,
                root_dir,
                args.checkpoint_dir,
                args.checkpoint_file,
                args.data_bin,
                args.data_root,
                args.user_dir,
                args.device,
                args.max_tokens,
                truth_subdir='truth_labeled_code',
                ida_subdir='ida_labeled_code',
            ),
            TASK_INSTBOUND: TaskConfig(
                TASK_INSTBOUND,
                root_dir,
                args.instbound_checkpoint_dir,
                args.instbound_checkpoint_file,
                args.instbound_data_bin,
                args.instbound_data_root,
                args.instbound_user_dir,
                args.device,
                args.max_tokens,
            ),
        }


CONFIG = None


def resolve_path(root_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def normalize_path(path):
    if not path:
        return '/'
    normalized = path.rstrip('/')
    return normalized or '/'

def resolve_task(task):
    task_name = (task or TASK_FUNCBOUND).strip().lower()
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f'Unsupported task: {task_name}')
    return task_name


def get_task_config(task):
    task_name = resolve_task(task)
    return CONFIG.tasks[task_name]


def load_label_map(task):
    task_name = resolve_task(task)
    if task_name in LABEL_MAP_CACHE:
        return LABEL_MAP_CACHE[task_name]

    config = get_task_config(task_name)
    label_path = os.path.join(config.data_bin, 'label', 'dict.txt')
    fallback = FUNC_LABEL_FALLBACK if task_name == TASK_FUNCBOUND else INST_LABEL_FALLBACK
    label_map = {}

    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as handle:
            for idx, line in enumerate(handle):
                token = line.strip().split()[0] if line.strip() else ''
                if not token:
                    continue
                label_map[token] = idx

    if not label_map:
        label_map = fallback.copy()
    else:
        for token, idx in fallback.items():
            label_map.setdefault(token, idx)

    if '-' in label_map and 'N' not in label_map:
        label_map['N'] = label_map['-']
    if 'N' in label_map and '-' not in label_map:
        label_map['-'] = label_map['N']

    LABEL_MAP_CACHE[task_name] = label_map
    return label_map


def map_label_to_id(label_map, label):
    if label in label_map:
        return label_map[label]
    upper = label.upper()
    if upper in label_map:
        return label_map[upper]
    lower = label.lower()
    if lower in label_map:
        return label_map[lower]
    return label_map.get('-', 0)


def get_label_sets(task):
    task_name = resolve_task(task)
    label_map = load_label_map(task_name)
    tokens = TASK_LABEL_TOKENS[task_name]

    def tokens_to_ids(values):
        return {label_map[value] for value in values if value in label_map}

    start_labels = tokens_to_ids(tokens.get('start', []))
    end_labels = tokens_to_ids(tokens.get('end', []))
    none_labels = tokens_to_ids(tokens.get('none', []))
    boundary_labels = start_labels | end_labels
    if not boundary_labels and label_map:
        boundary_labels = set(label_map.values()) - none_labels

    return start_labels, end_labels, boundary_labels


def load_model(task):
    task_name = resolve_task(task)
    if task_name in MODEL_CACHE:
        return MODEL_CACHE[task_name]

    with MODEL_LOCK:
        if task_name in MODEL_CACHE:
            return MODEL_CACHE[task_name]

        config = get_task_config(task_name)
        checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        if not os.path.isdir(config.data_bin):
            raise FileNotFoundError(f'Data bin not found: {config.data_bin}')
        if not os.path.isdir(config.user_dir):
            raise FileNotFoundError(f'User dir not found: {config.user_dir}')

        model = RobertaModel.from_pretrained(
            config.checkpoint_dir,
            config.checkpoint_file,
            config.data_bin,
            bpe=None,
            user_dir=config.user_dir
        )

        use_cuda = config.device == 'cuda' or (
            config.device == 'auto' and torch.cuda.is_available()
        )
        if use_cuda:
            model = model.cuda()
            MODEL_DEVICE[task_name] = 'cuda'
        else:
            MODEL_DEVICE[task_name] = 'cpu'

        model.eval()
        MODEL_CACHE[task_name] = model
        return model


def count_lines(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return sum(1 for _ in handle)


def load_samples(task):
    task_name = resolve_task(task)
    if task_name in SAMPLE_CACHE:
        return SAMPLE_CACHE[task_name]

    config = get_task_config(task_name)
    truth_dir = config.data_root
    ida_dir = None

    if config.truth_subdir:
        truth_dir = os.path.join(config.data_root, config.truth_subdir)
    if config.ida_subdir:
        ida_dir = os.path.join(config.data_root, config.ida_subdir)

    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f'Missing directory: {truth_dir}')
    if ida_dir and not os.path.isdir(ida_dir):
        raise FileNotFoundError(f'Missing directory: {ida_dir}')

    if ida_dir:
        truth_files = set(os.listdir(truth_dir))
        ida_files = set(os.listdir(ida_dir))
        shared = sorted(truth_files & ida_files)
    else:
        shared = sorted(os.listdir(truth_dir))

    samples = []
    for name in shared:
        path = os.path.join(truth_dir, name)
        if not os.path.isfile(path):
            continue
        length = count_lines(path)
        samples.append({
            'id': name,
            'name': name,
            'length': length
        })

    SAMPLE_CACHE[task_name] = samples
    SAMPLE_INDEX[task_name] = {sample['id']: sample for sample in samples}
    return samples


def get_sample(task, sample_id):
    load_samples(task)
    return SAMPLE_INDEX.get(resolve_task(task), {}).get(sample_id)


def read_slice(path, start, end, label_map):
    tokens = []
    labels = []
    with open(path, 'r', encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            if idx < start:
                continue
            if idx > end:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0].lower())
            labels.append(map_label_to_id(label_map, parts[1]))
    return tokens, labels


def read_full(path, label_map):
    tokens = []
    labels = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0].lower())
            labels.append(map_label_to_id(label_map, parts[1]))
    return tokens, labels


def init_counts():
    return {
        'total': 0,
        'total_correct': 0,
        'boundary_true': 0,
        'boundary_pred': 0,
        'boundary_correct': 0
    }


def update_counts(counts, truth_labels, pred_labels, boundary_labels=None):
    boundary_set = set(boundary_labels) if boundary_labels is not None else None
    for truth, pred in zip(truth_labels, pred_labels):
        counts['total'] += 1
        if truth == pred:
            counts['total_correct'] += 1
        is_boundary_truth = truth in boundary_set if boundary_set is not None else truth != 0
        is_boundary_pred = pred in boundary_set if boundary_set is not None else pred != 0
        if is_boundary_truth:
            counts['boundary_true'] += 1
            if pred == truth:
                counts['boundary_correct'] += 1
        if is_boundary_pred:
            counts['boundary_pred'] += 1


def metrics_from_counts(counts):
    precision = safe_ratio(counts['boundary_correct'], counts['boundary_pred'])
    recall = safe_ratio(counts['boundary_correct'], counts['boundary_true'])
    f1 = safe_f1(precision, recall)
    accuracy = counts['total_correct'] / counts['total'] if counts['total'] else 0.0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def init_validation_job():
    return {
        'status': 'idle',
        'progress': 0,
        'logs': [],
        'error': None,
        'result': None,
        'startedAt': None,
        'finishedAt': None
    }

def get_validation_job(task_name):
    job = VALIDATION_JOB.get(task_name)
    if job is None:
        job = init_validation_job()
        VALIDATION_JOB[task_name] = job
    return job


def append_validation_log(task, message, level='info'):
    task_name = resolve_task(task)
    with VALIDATION_LOCK:
        job = get_validation_job(task_name)
        job['logs'].append({'message': message, 'level': level})


def set_validation_progress(task, progress):
    task_name = resolve_task(task)
    with VALIDATION_LOCK:
        job = get_validation_job(task_name)
        job['progress'] = max(0, min(100, int(progress)))


def set_validation_status(task, status, error=None, result=None):
    task_name = resolve_task(task)
    with VALIDATION_LOCK:
        job = get_validation_job(task_name)
        job['status'] = status
        job['error'] = error
        if result is not None:
            job['result'] = result
        if status in ('done', 'error'):
            job['finishedAt'] = time.time()


def serialize_validation_job(task, offset=0):
    task_name = resolve_task(task)
    with VALIDATION_LOCK:
        job = get_validation_job(task_name)
        logs = job['logs'][offset:] if offset < len(job['logs']) else []
        payload = {
            'status': job['status'],
            'progress': job['progress'],
            'logs': logs,
            'logCount': len(job['logs']),
            'error': job['error'],
            'task': task_name
        }
        if job['status'] == 'done' and job['result'] is not None:
            payload['result'] = job['result']
        return payload


def start_validation_job(task, refresh=False):
    task_name = resolve_task(task)

    with VALIDATION_LOCK:
        job = get_validation_job(task_name)

        if job['status'] == 'running':
            return

        cached = VALIDATION_CACHE.get(task_name)
        if cached is not None and not refresh:
            job = init_validation_job()
            job['status'] = 'done'
            job['progress'] = 100
            job['result'] = cached
            job['finishedAt'] = time.time()
            job['logs'].append({'message': '验证集指标已缓存', 'level': 'info'})
            VALIDATION_JOB[task_name] = job
            return

        VALIDATION_CACHE.pop(task_name, None)
        save_validation_cache()
        job = init_validation_job()
        job['status'] = 'running'
        job['startedAt'] = time.time()
        job['logs'].append({'message': '开始计算验证集指标', 'level': 'info'})
        VALIDATION_JOB[task_name] = job

    thread = threading.Thread(target=run_validation_job, args=(task_name,), daemon=True)
    thread.start()


def run_validation_job(task):
    task_name = resolve_task(task)
    try:

        samples = load_samples(task_name)
        if not samples:
            raise ValueError('No samples available for validation metrics')

        config = get_task_config(task_name)
        model = load_model(task_name)
        max_tokens = config.max_tokens or 512
        total_chunks = 0
        for sample in samples:
            if sample['length'] > 0:
                total_chunks += (sample['length'] + max_tokens - 1) // max_tokens

        append_validation_log(task_name, f'样本数: {len(samples)} | 总块数: {total_chunks}')
        append_validation_log(task_name, f'设备: {MODEL_DEVICE.get(task_name, "cpu")}')

        counts_deepbound = init_counts()
        counts_ida = init_counts() if config.ida_subdir else None
        total_io_ms = 0
        total_pred_ms = 0
        processed_chunks = 0
        last_progress = -1
        last_log_progress = -10
        _, _, boundary_labels = get_label_sets(task_name)
        label_map = load_label_map(task_name)

        for sample in samples:
            truth_path = os.path.join(
                config.data_root,
                config.truth_subdir,
                sample['id']
            ) if config.truth_subdir else os.path.join(config.data_root, sample['id'])
            ida_path = os.path.join(
                config.data_root,
                config.ida_subdir,
                sample['id']
            ) if config.ida_subdir else None
            
            num_sample_chunks = (sample['length'] + max_tokens - 1) // max_tokens
            append_validation_log(task_name, f'处理样本: {sample["id"]} ({sample["length"]} 字节)')

            io_start = time.time()
            truth_tokens, truth_labels = read_full(truth_path, label_map)
            ida_tokens, ida_labels = ([], [])
            if ida_path:
                ida_tokens, ida_labels = read_full(ida_path, label_map)
            total_io_ms += int((time.time() - io_start) * 1000)

            full_len = len(truth_tokens)
            if full_len == 0:
                processed_chunks += num_sample_chunks
                continue

            # DeepBound evaluation on FULL truth
            for offset in range(0, full_len, max_tokens):
                chunk_tokens = truth_tokens[offset: offset + max_tokens]
                chunk_truth = truth_labels[offset: offset + max_tokens]

                if not chunk_tokens:
                    processed_chunks += 1
                    continue

                pred_start = time.time()
                with INFER_LOCK:
                    with torch.no_grad():
                        encoded = model.encode(' '.join(chunk_tokens))
                        logprobs = model.predict(task_name, encoded)
                        preds = logprobs.argmax(dim=2).view(-1).detach().cpu().tolist()
                        preds = preds[: len(chunk_tokens)]
                total_pred_ms += int((time.time() - pred_start) * 1000)

                update_counts(counts_deepbound, chunk_truth, preds, boundary_labels)

                processed_chunks += 1
                progress = int((processed_chunks / total_chunks) * 100) if total_chunks else 100
                if progress != last_progress:
                    set_validation_progress(task_name, progress)
                    last_progress = progress
                    if progress - last_log_progress >= 10:
                        append_validation_log(task_name, f'进度: {progress}%')
                        last_log_progress = progress

            # IDA evaluation on intersection
            if ida_path and ida_labels:
                if ida_tokens and truth_tokens and truth_tokens != ida_tokens:
                    # Optional: log warning about mismatch
                    pass
                comp_len = min(len(truth_labels), len(ida_labels))
                if comp_len > 0:
                    update_counts(counts_ida, truth_labels[:comp_len], ida_labels[:comp_len], boundary_labels)


        result = {
            'scope': 'validation',
            'task': task_name,
            'samples': len(samples),
            'totalTokens': counts_deepbound['total'],
            'dataRoot': config.data_root,
            'metrics': {
                'deepbound': metrics_from_counts(counts_deepbound),
                'ida': metrics_from_counts(counts_ida) if counts_ida else None
            },
            'timing': {
                'ioMs': total_io_ms,
                'predictMs': total_pred_ms
            },
            'device': MODEL_DEVICE.get(task_name, 'cpu')
        }
        VALIDATION_CACHE[task_name] = result
        save_validation_cache()
        set_validation_progress(task_name, 100)
        append_validation_log(task_name, '验证集指标计算完成')
        set_validation_status(task_name, 'done', result=result)
    except Exception as exc:
        append_validation_log(task_name, f'验证集指标计算失败: {exc}', level='alert')
        set_validation_status(task_name, 'error', error=str(exc))


def clamp_range(start, end, length, max_tokens=None):
    start = safe_int(start, 0)
    end = safe_int(end, start)
    start = max(0, start)
    end = max(0, end)
    if start > end:
        start, end = end, start

    if length <= 0:
        return 0, 0, False

    if start >= length:
        start = length - 1
    if end >= length:
        end = length - 1

    clamped = False
    if max_tokens and end - start + 1 > max_tokens:
        end = start + max_tokens - 1
        if end >= length:
            end = length - 1
            start = max(0, end - max_tokens + 1)
        clamped = True

    return start, end, clamped


def safe_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def safe_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def safe_f1(precision, recall):
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return 0.0
    return (2 * precision * recall) / denom


def predict_slice(task, sample_id, start, end):
    task_name = resolve_task(task)
    config = get_task_config(task_name)
    sample = get_sample(task_name, sample_id)
    if not sample:
        raise ValueError(f'Unknown sample: {sample_id}')

    label_map = load_label_map(task_name)
    start, end, clamped = clamp_range(start, end, sample['length'], config.max_tokens)

    truth_path = config.data_root
    if config.truth_subdir:
        truth_path = os.path.join(config.data_root, config.truth_subdir, sample_id)
    else:
        truth_path = os.path.join(config.data_root, sample_id)

    ida_path = None
    if config.ida_subdir:
        ida_path = os.path.join(config.data_root, config.ida_subdir, sample_id)

    io_start = time.time()
    ida_labels = None
    if ida_path:
        tokens, ida_labels = read_slice(ida_path, start, end, label_map)
        truth_tokens, truth_labels = read_slice(truth_path, start, end, label_map)
        if truth_tokens and tokens != truth_tokens:
            tokens = truth_tokens
        min_len = min(len(tokens), len(truth_labels), len(ida_labels))
    else:
        truth_tokens, truth_labels = read_slice(truth_path, start, end, label_map)
        tokens = truth_tokens
        min_len = min(len(tokens), len(truth_labels))

    if min_len == 0:
        raise ValueError('Empty slice after loading data')
    tokens = tokens[:min_len]
    truth_labels = truth_labels[:min_len]
    if ida_labels is not None:
        ida_labels = ida_labels[:min_len]
    end = start + min_len - 1
    io_ms = int((time.time() - io_start) * 1000)

    config = get_task_config(task_name)
    model = load_model(task_name)
    encode_start = time.time()
    deepbound_labels = predict_labels_for_tokens(model, tokens, config.head)
    predict_ms = int((time.time() - encode_start) * 1000)

    disassembly = []
    try:
        arch = get_arch_from_name(sample_id)
        disassembly = disassemble_tokens(tokens, arch)
    except Exception as e:
        disassembly = [{'text': f"Disassembly failed: {e}", 'offset': 0, 'bytes': []}]

    start_labels, end_labels, boundary_labels = get_label_sets(task_name)
    metrics_deepbound = compute_metrics(truth_labels, deepbound_labels, boundary_labels)
    metrics_ida = None
    counts_deepbound = count_hits(truth_labels, deepbound_labels, start_labels, end_labels)
    counts_ida = None
    if ida_labels is not None:
        metrics_ida = compute_metrics(truth_labels, ida_labels, boundary_labels)
        counts_ida = count_hits(truth_labels, ida_labels, start_labels, end_labels)

    return {
        'task': task_name,
        'sampleId': sample_id,
        'sampleLength': sample['length'],
        'start': start,
        'end': end,
        'tokens': tokens,
        'truthLabels': truth_labels,
        'idaLabels': ida_labels,
        'deepboundLabels': deepbound_labels,
        'disassembly': disassembly,
        'metrics': {
            'deepbound': metrics_deepbound,
            'ida': metrics_ida
        },
        'counts': {
            'deepbound': counts_deepbound,
            'ida': counts_ida
        },
        'timing': {
            'ioMs': io_ms,
            'predictMs': predict_ms
        },
        'device': MODEL_DEVICE.get(task_name, 'cpu'),
        'clamped': clamped
    }


def compute_metrics(truth_labels, pred_labels, boundary_labels=None):
    total = len(truth_labels)
    total_correct = 0
    boundary_correct = 0
    boundary_true = 0
    boundary_pred = 0
    boundary_set = set(boundary_labels) if boundary_labels is not None else None

    for truth, pred in zip(truth_labels, pred_labels):
        if truth == pred:
            total_correct += 1
        is_boundary_truth = truth in boundary_set if boundary_set is not None else truth != 0
        is_boundary_pred = pred in boundary_set if boundary_set is not None else pred != 0
        if is_boundary_truth:
            boundary_true += 1
            if pred == truth:
                boundary_correct += 1
        if is_boundary_pred:
            boundary_pred += 1

    precision = safe_ratio(boundary_correct, boundary_pred)
    recall = safe_ratio(boundary_correct, boundary_true)
    f1 = safe_f1(precision, recall)
    accuracy = total_correct / total if total else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def count_hits(truth_labels, pred_labels, start_labels=None, end_labels=None):
    start_hits = 0
    end_hits = 0
    start_set = set(start_labels) if start_labels is not None else set()
    end_set = set(end_labels) if end_labels is not None else set()
    for truth, pred in zip(truth_labels, pred_labels):
        if truth in start_set and pred == truth:
            start_hits += 1
        if truth in end_set and pred == truth:
            end_hits += 1
    return {
        'startHits': start_hits,
        'endHits': end_hits
    }


def predict_labels_for_tokens(model, tokens, head_name):
    if not tokens:
        return []
    with INFER_LOCK:
        with torch.no_grad():
            encoded = model.encode(' '.join(tokens))
            logprobs = model.predict(head_name, encoded)
            labels = logprobs.argmax(dim=2).view(-1).detach().cpu().tolist()

    return labels[: len(tokens)]


def iter_windows(length, window, stride, max_windows, max_tokens=None):
    if length <= 0:
        return []
    window = safe_int(window, 0)
    max_tokens = max_tokens or 512
    if window <= 0:
        window = max_tokens
    window = min(window, max_tokens, length)
    stride = safe_int(stride, window)
    if stride <= 0:
        stride = window

    starts = list(range(0, max(length - window + 1, 1), stride))
    last_start = max(0, length - window)
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))

    max_windows = safe_int(max_windows, 0)
    if max_windows > 0 and len(starts) > max_windows:
        step = len(starts) / max_windows
        selected = [starts[int(i * step)] for i in range(max_windows)]
        selected.append(starts[-1])
        starts = sorted(set(selected))

    return [(start, min(length, start + window)) for start in starts]


def find_better_positions(truth_labels, deepbound_labels, ida_labels, start_offset, start_labels, end_labels):
    better_starts = []
    better_ends = []
    start_set = set(start_labels) if start_labels is not None else set()
    end_set = set(end_labels) if end_labels is not None else set()
    for idx, (truth, deepbound, ida) in enumerate(zip(truth_labels, deepbound_labels, ida_labels)):
        if truth in start_set and deepbound == truth and ida != truth:
            better_starts.append(start_offset + idx)
        if truth in end_set and deepbound == truth and ida != truth:
            better_ends.append(start_offset + idx)
    return better_starts, better_ends


def recommend_showcase(task, options):
    task_name = resolve_task(task)
    global SHOWCASE_CACHE
    config = get_task_config(task_name)
    refresh = options.get('refresh', False)
    cache_key_options = options.copy()
    cache_key_options.pop('refresh', None)
    cache_key = json.dumps(cache_key_options, sort_keys=True)
    if not refresh:
        with SHOWCASE_LOCK:
            task_cache = SHOWCASE_CACHE.get(task_name, {})
            cached = task_cache.get(cache_key)
            if cached:
                # Verify cache validity against current samples
                cached_payload = cached.get('payload', {})
                cached_items = cached_payload.get('items', [])
                current_ids = set(s['id'] for s in load_samples(task_name))
                if cached_items and all(item['sampleId'] in current_ids for item in cached_items):
                    result = cached_payload.copy()
                    result['cached'] = True
                    return result

    start_time = time.time()
    top = max(1, safe_int(options.get('top', 3), 3))
    max_samples = max(0, safe_int(options.get('maxSamples', 30), 30))
    min_delta_f1 = safe_float(options.get('minDeltaF1', 0.0), 0.0)
    min_better = max(0, safe_int(options.get('minBetterBoundaries', 1), 1))
    window = safe_int(options.get('window', 0), 0)
    stride = safe_int(options.get('stride', 0), 0)
    max_windows = max(0, safe_int(options.get('maxWindows', 24), 24))
    limit_positions = max(0, safe_int(options.get('limitPositions', 20), 20))
    sample_ids = options.get('sampleIds') or []

    samples = load_samples(task_name)
    if sample_ids:
        sample_ids_set = set(sample_ids)
        samples = [sample for sample in samples if sample['id'] in sample_ids_set]
    if max_samples > 0:
        samples = samples[: max_samples]

    model = load_model(task_name)
    label_map = load_label_map(task_name)
    start_labels, end_labels, boundary_labels = get_label_sets(task_name)
    results = []
    samples_checked = 0
    windows_checked = 0

    if task_name == TASK_FUNCBOUND:
        for sample in samples:
            truth_path = os.path.join(config.data_root, config.truth_subdir, sample['id'])
            ida_path = os.path.join(config.data_root, config.ida_subdir, sample['id'])
            truth_tokens, truth_labels = read_full(truth_path, label_map)
            ida_tokens, ida_labels = read_full(ida_path, label_map)
            if truth_tokens and ida_tokens and truth_tokens != ida_tokens:
                tokens = truth_tokens
            else:
                tokens = truth_tokens or ida_tokens

            length = min(len(tokens), len(truth_labels), len(ida_labels))
            if length <= 0:
                continue
            tokens = tokens[:length]
            truth_labels = truth_labels[:length]
            ida_labels = ida_labels[:length]

            samples_checked += 1
            windows = iter_windows(length, window, stride, max_windows, config.max_tokens)
            for start, end in windows:
                truth_slice = truth_labels[start:end]
                if not any(label in boundary_labels for label in truth_slice):
                    continue
                ida_slice = ida_labels[start:end]
                deepbound_labels = predict_labels_for_tokens(model, tokens[start:end], config.head)
                if len(deepbound_labels) != len(truth_slice):
                    continue

                windows_checked += 1
                deepbound_metrics = compute_metrics(truth_slice, deepbound_labels, boundary_labels)
                ida_metrics = compute_metrics(truth_slice, ida_slice, boundary_labels)
                deepbound_f1 = deepbound_metrics['f1'] or 0.0
                ida_f1 = ida_metrics['f1'] or 0.0
                delta_f1 = deepbound_f1 - ida_f1

                better_starts, better_ends = find_better_positions(
                    truth_slice, deepbound_labels, ida_slice, start, start_labels, end_labels
                )
                better_total = len(better_starts) + len(better_ends)

                if delta_f1 < min_delta_f1 or better_total < min_better:
                    continue

                counts_deepbound = count_hits(truth_slice, deepbound_labels, start_labels, end_labels)
                counts_ida = count_hits(truth_slice, ida_slice, start_labels, end_labels)

                results.append({
                    'sampleId': sample['id'],
                    'start': start,
                    'end': end - 1,
                    'length': end - start,
                    'deltaF1': delta_f1,
                    'deepbound': deepbound_metrics,
                    'ida': ida_metrics,
                    'hits': {
                        'deepbound': counts_deepbound,
                        'ida': counts_ida,
                        'deltaStart': counts_deepbound['startHits'] - counts_ida['startHits'],
                        'deltaEnd': counts_deepbound['endHits'] - counts_ida['endHits']
                    },
                    'betterStarts': better_starts[:limit_positions],
                    'betterEnds': better_ends[:limit_positions],
                    'betterStartCount': len(better_starts),
                    'betterEndCount': len(better_ends)
                })
    else:
        for sample in samples:
            truth_path = os.path.join(config.data_root, sample['id'])
            truth_tokens, truth_labels = read_full(truth_path, label_map)
            length = min(len(truth_tokens), len(truth_labels))
            if length <= 0:
                continue
            tokens = truth_tokens[:length]
            truth_labels = truth_labels[:length]

            samples_checked += 1
            windows = iter_windows(length, window, stride, max_windows, config.max_tokens)
            for start, end in windows:
                truth_slice = truth_labels[start:end]
                boundary_count = sum(1 for label in truth_slice if label in boundary_labels)
                if boundary_count < min_better:
                    continue
                deepbound_labels = predict_labels_for_tokens(model, tokens[start:end], config.head)
                if len(deepbound_labels) != len(truth_slice):
                    continue

                windows_checked += 1
                deepbound_metrics = compute_metrics(truth_slice, deepbound_labels, boundary_labels)
                counts_deepbound = count_hits(truth_slice, deepbound_labels, start_labels, end_labels)

                results.append({
                    'sampleId': sample['id'],
                    'start': start,
                    'end': end - 1,
                    'length': end - start,
                    'boundaryCount': boundary_count,
                    'deepbound': deepbound_metrics,
                    'ida': None,
                    'hits': {
                        'deepbound': counts_deepbound
                    }
                })

    if task_name == TASK_FUNCBOUND:
        results.sort(
            key=lambda item: (
                item['deltaF1'],
                item['betterStartCount'] + item['betterEndCount'],
                item['deepbound']['f1'] or 0.0
            ),
            reverse=True
        )
    else:
        results.sort(
            key=lambda item: (
                item['deepbound']['f1'] or 0.0,
                item.get('boundaryCount', 0),
                item['hits']['deepbound']['startHits']
            ),
            reverse=True
        )
    results = results[:top]

    payload = {
        'cached': False,
        'task': task_name,
        'strategy': 'delta_f1' if task_name == TASK_FUNCBOUND else 'top_f1',
        'items': results,
        'params': {
            'top': top,
            'maxSamples': max_samples,
            'minDeltaF1': min_delta_f1,
            'minBetterBoundaries': min_better,
            'minBoundaryCount': min_better,
            'window': window or (config.max_tokens or 512),
            'stride': stride or (window or config.max_tokens or 512),
            'maxWindows': max_windows,
            'limitPositions': limit_positions,
            'samples': samples_checked,
            'windowsChecked': windows_checked
        },
        'durationMs': int((time.time() - start_time) * 1000)
    }

    with SHOWCASE_LOCK:
        if task_name not in SHOWCASE_CACHE:
            SHOWCASE_CACHE[task_name] = {}
        SHOWCASE_CACHE[task_name][cache_key] = {
            'payload': payload,
            'timestamp': int(time.time())
        }
        save_persistent_cache()

    return payload


class DeepBoundHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _set_headers(self, status=200, content_type='application/json', length=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Connection', 'keep-alive')
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.end_headers()

    def _write_json(self, payload, status=200):
        try:
            body = json.dumps(payload).encode('utf-8')
            self._set_headers(status=status, length=len(body))
            self.wfile.write(body)
        except Exception as e:
            print(f"Error writing JSON response: {e}")

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        try:
            self._do_get_handler()
        except Exception as exc:
            self._write_json({'error': str(exc)}, status=500)

    def _do_get_handler(self):
        parsed = urlparse(self.path)
        path = normalize_path(parsed.path)
        if path == '/api/health':
            payload = {
                'status': 'ok',
                'modelLoaded': {name: name in MODEL_CACHE for name in SUPPORTED_TASKS},
                'device': MODEL_DEVICE
            }
            self._write_json(payload)
            return

        if path == '/api/samples':
            query = parse_qs(parsed.query)
            task = query.get('task', [TASK_FUNCBOUND])[0]
            samples = load_samples(task)
            self._write_json({'samples': samples})
            return

        if path == '/api/validation/start':
            query = parse_qs(parsed.query)
            task = query.get('task', [TASK_FUNCBOUND])[0]
            refresh = query.get('refresh', ['0'])[0] == '1'
            start_validation_job(task, refresh=refresh)
            payload = serialize_validation_job(task, offset=0)
            self._write_json(payload)
            return

        if path == '/api/validation/status':
            query = parse_qs(parsed.query)
            task = query.get('task', [TASK_FUNCBOUND])[0]
            offset = safe_int(query.get('offset', ['0'])[0], 0)
            payload = serialize_validation_job(task, offset=offset)
            self._write_json(payload)
            return

        if path == '/api/validation':
            query = parse_qs(parsed.query)
            task = query.get('task', [TASK_FUNCBOUND])[0]
            payload = serialize_validation_job(task, offset=0)
            self._write_json(payload)
            return

        if path == '/api/showcase/recommend':
            query = parse_qs(parsed.query)
            task = query.get('task', [TASK_FUNCBOUND])[0]
            options = {
                'top': safe_int(query.get('top', ['3'])[0], 3),
                'maxSamples': safe_int(query.get('maxSamples', query.get('max_samples', ['30']))[0], 30),
                'minDeltaF1': safe_float(query.get('minDeltaF1', query.get('min_delta_f1', ['0']))[0], 0.0),
                'minBetterBoundaries': safe_int(
                    query.get('minBetterBoundaries', query.get('min_better_boundaries', ['1']))[0],
                    1
                ),
                'window': safe_int(query.get('window', ['0'])[0], 0),
                'stride': safe_int(query.get('stride', ['0'])[0], 0),
                'maxWindows': safe_int(query.get('maxWindows', query.get('max_windows', ['24']))[0], 24),
                'limitPositions': safe_int(query.get('limitPositions', query.get('limit_positions', ['20']))[0], 20),
                'sampleIds': query.get('sample', []) + query.get('sampleId', []),
                'refresh': query.get('refresh', ['0'])[0] == '1'
            }
            payload = recommend_showcase(task, options)
            self._write_json(payload)
            return

        self._write_json({'error': 'Not found'}, status=404)

    def do_POST(self):
        try:
            self._do_post_handler()
        except Exception as exc:
            self._write_json({'error': str(exc)}, status=500)

    def _do_post_handler(self):
        parsed = urlparse(self.path)
        path = normalize_path(parsed.path)
        
        # Reuse GET logic for validation endpoints if they are called via POST
        if path in ('/api/validation/start', '/api/validation/status', '/api/validation'):
             self._do_get_handler()
             return

        if path != '/api/predict':
            self._write_json({'error': 'Not found'}, status=404)
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length).decode('utf-8') if length else '{}'
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._write_json({'error': 'Invalid JSON body'}, status=400)
            return

        sample_id = data.get('sampleId')
        start = data.get('start', 0)
        end = data.get('end', 0)
        task = data.get('task', TASK_FUNCBOUND)

        if not sample_id:
            self._write_json({'error': 'sampleId is required'}, status=400)
            return

        payload = predict_slice(task, sample_id, start, end)
        self._write_json(payload)



def parse_args():
    parser = argparse.ArgumentParser(description='DeepBound demo backend server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint-dir', default='checkpoints/finetune_msvs_funcbound_64')
    parser.add_argument('--checkpoint-file', default='checkpoint_best.pt')
    parser.add_argument('--data-bin', default='data-bin/funcbound_msvs_64')
    parser.add_argument('--data-root', default='data-raw/msvs_funcbound_64_bap_test')
    parser.add_argument('--user-dir', default='finetune_tasks')
    parser.add_argument('--instbound-checkpoint-dir', default='checkpoints/finetune_instbound_elf')
    parser.add_argument('--instbound-checkpoint-file', default='checkpoint_best.pt')
    parser.add_argument('--instbound-data-bin', default='data-bin/finetune_instbound_elf')
    parser.add_argument('--instbound-data-root', default='data-raw/inst_bound')
    parser.add_argument('--instbound-user-dir', default='finetune_tasks')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-tokens', type=int, default=512)
    return parser.parse_args()


def main():
    global CONFIG
    args = parse_args()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CONFIG = Config(root_dir, args)

    server = ThreadingHTTPServer((args.host, args.port), DeepBoundHandler)
    print(f'DeepBound backend running on http://{args.host}:{args.port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
