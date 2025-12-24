#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
from fairseq.models.roberta import RobertaModel


LABEL_TO_ID = {
    '-': 0,
    'R': 1,
    'F': 2
}


def resolve_path(root_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def load_labels(path):
    tokens = []
    labels = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0].lower())
            labels.append(LABEL_TO_ID.get(parts[1].upper(), 0))
    return tokens, labels


def iter_windows(length, window, stride):
    start = 0
    while start < length:
        end = min(length, start + window)
        yield start, end
        if end == length:
            break
        start += stride


def compute_metrics(truth_labels, pred_labels):
    total = len(truth_labels)
    total_correct = 0
    boundary_correct = 0
    boundary_true = 0
    boundary_pred = 0

    for truth, pred in zip(truth_labels, pred_labels):
        if truth == pred:
            total_correct += 1
        if truth != 0:
            boundary_true += 1
            if pred == truth:
                boundary_correct += 1
        if pred != 0:
            boundary_pred += 1

    precision = None if boundary_pred == 0 else boundary_correct / boundary_pred
    recall = None if boundary_true == 0 else boundary_correct / boundary_true
    f1 = None
    if precision is not None and recall is not None:
        denom = precision + recall
        f1 = 0 if denom == 0 else (2 * precision * recall) / denom
    accuracy = total_correct / total if total else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'boundaryTrue': boundary_true
    }


def count_hits(truth_labels, pred_labels):
    start_hits = 0
    end_hits = 0
    for truth, pred in zip(truth_labels, pred_labels):
        if truth == 2 and pred == 2:
            start_hits += 1
        if truth == 1 and pred == 1:
            end_hits += 1
    return {
        'startHits': start_hits,
        'endHits': end_hits
    }


def find_better_positions(truth_labels, deepbound_labels, ida_labels, start_offset):
    better_starts = []
    better_ends = []
    for idx, (truth, deepbound, ida) in enumerate(zip(truth_labels, deepbound_labels, ida_labels)):
        if truth == 2 and deepbound == 2 and ida != 2:
            better_starts.append(start_offset + idx)
        if truth == 1 and deepbound == 1 and ida != 1:
            better_ends.append(start_offset + idx)
    return better_starts, better_ends


def load_model(checkpoint_dir, checkpoint_file, data_bin, user_dir, device):
    model = RobertaModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file,
        data_bin,
        bpe=None,
        user_dir=user_dir
    )

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    if use_cuda:
        model = model.cuda()
    model.eval()
    return model


def predict_labels(model, tokens):
    with torch.no_grad():
        encoded = model.encode(' '.join(tokens))
        logprobs = model.predict('funcbound', encoded)
        labels = logprobs.argmax(dim=2).view(-1).detach().cpu().tolist()
        return labels[: len(tokens)]


def main():
    parser = argparse.ArgumentParser(description='Find DeepBound showcase slices with better boundaries')
    parser.add_argument('--checkpoint-dir', default='checkpoints/finetune_msvs_funcbound_64')
    parser.add_argument('--checkpoint-file', default='checkpoint_best.pt')
    parser.add_argument('--data-bin', default='data-bin/funcbound_msvs_64')
    parser.add_argument('--data-root', default='data-raw/msvs_funcbound_64_bap_test')
    parser.add_argument('--user-dir', default='finetune_tasks')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--window', type=int, default=0)
    parser.add_argument('--stride', type=int, default=0)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--min-delta-f1', type=float, default=0.0)
    parser.add_argument('--min-better-boundaries', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--sample', action='append', default=[])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = resolve_path(root_dir, args.checkpoint_dir)
    data_bin = resolve_path(root_dir, args.data_bin)
    data_root = resolve_path(root_dir, args.data_root)
    user_dir = resolve_path(root_dir, args.user_dir)

    window = args.window if args.window > 0 else args.max_tokens
    window = min(window, args.max_tokens)
    stride = args.stride if args.stride > 0 else window

    truth_dir = os.path.join(data_root, 'truth_labeled_code')
    ida_dir = os.path.join(data_root, 'ida_labeled_code')
    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f'Missing directory: {truth_dir}')
    if not os.path.isdir(ida_dir):
        raise FileNotFoundError(f'Missing directory: {ida_dir}')

    truth_files = set(os.listdir(truth_dir))
    ida_files = set(os.listdir(ida_dir))
    samples = sorted(truth_files & ida_files)

    if args.sample:
        selected = set(args.sample)
        samples = [name for name in samples if name in selected]

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        print('No samples found.')
        return

    model = load_model(checkpoint_dir, args.checkpoint_file, data_bin, user_dir, args.device)

    results = []
    start_time = time.time()
    for idx, name in enumerate(samples, start=1):
        truth_path = os.path.join(truth_dir, name)
        ida_path = os.path.join(ida_dir, name)
        tokens, truth_labels = load_labels(truth_path)
        ida_tokens, ida_labels = load_labels(ida_path)
        length = min(len(tokens), len(truth_labels), len(ida_labels))
        if length == 0:
            continue
        tokens = tokens[:length]
        truth_labels = truth_labels[:length]
        ida_labels = ida_labels[:length]

        if idx % 10 == 0 or idx == 1:
            elapsed = int(time.time() - start_time)
            print(f'[{idx}/{len(samples)}] {name} ({length} tokens) elapsed {elapsed}s')

        for start, end in iter_windows(length, window, stride):
            truth_slice = truth_labels[start:end]
            ida_slice = ida_labels[start:end]
            if not any(label != 0 for label in truth_slice):
                continue

            ida_metrics = compute_metrics(truth_slice, ida_slice)
            if ida_metrics['boundaryTrue'] == 0:
                continue

            deepbound_labels = predict_labels(model, tokens[start:end])
            if len(deepbound_labels) != len(truth_slice):
                continue

            deepbound_metrics = compute_metrics(truth_slice, deepbound_labels)
            deepbound_f1 = deepbound_metrics['f1'] or 0.0
            ida_f1 = ida_metrics['f1'] or 0.0
            delta_f1 = deepbound_f1 - ida_f1

            start_hits_deepbound = count_hits(truth_slice, deepbound_labels)
            start_hits_ida = count_hits(truth_slice, ida_slice)
            delta_start = start_hits_deepbound['startHits'] - start_hits_ida['startHits']
            delta_end = start_hits_deepbound['endHits'] - start_hits_ida['endHits']
            better_starts, better_ends = find_better_positions(
                truth_slice, deepbound_labels, ida_slice, start
            )
            better_total = len(better_starts) + len(better_ends)

            if delta_f1 < args.min_delta_f1 and better_total < args.min_better_boundaries:
                continue

            results.append({
                'sampleId': name,
                'start': start,
                'end': end - 1,
                'length': end - start,
                'deltaF1': delta_f1,
                'deepbound': deepbound_metrics,
                'ida': ida_metrics,
                'hits': {
                    'deepbound': start_hits_deepbound,
                    'ida': start_hits_ida,
                    'deltaStart': delta_start,
                    'deltaEnd': delta_end
                },
                'betterStarts': better_starts,
                'betterEnds': better_ends
            })

    results.sort(
        key=lambda item: (
            item['deltaF1'],
            item['hits']['deltaStart'] + item['hits']['deltaEnd'],
            item['deepbound']['f1'] or 0.0
        ),
        reverse=True
    )

    results = results[: args.top]

    if args.json:
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return

    if not results:
        print('No showcase slices found with current filters.')
        return

    for rank, item in enumerate(results, start=1):
        deepbound_f1 = item['deepbound']['f1'] or 0.0
        ida_f1 = item['ida']['f1'] or 0.0
        print(f'#{rank} {item["sampleId"]} {item["start"]}-{item["end"]} len={item["length"]}')
        print(f'  deltaF1={item["deltaF1"]:.4f} deepboundF1={deepbound_f1:.4f} idaF1={ida_f1:.4f}')
        print(
            f'  startHits deepbound={item["hits"]["deepbound"]["startHits"]} '
            f'ida={item["hits"]["ida"]["startHits"]} '
            f'delta={item["hits"]["deltaStart"]}'
        )
        print(
            f'  endHits deepbound={item["hits"]["deepbound"]["endHits"]} '
            f'ida={item["hits"]["ida"]["endHits"]} '
            f'delta={item["hits"]["deltaEnd"]}'
        )
        if item['betterStarts']:
            print(f'  betterStarts: {item["betterStarts"][:8]}')
        if item['betterEnds']:
            print(f'  betterEnds: {item["betterEnds"][:8]}')


if __name__ == '__main__':
    main()
