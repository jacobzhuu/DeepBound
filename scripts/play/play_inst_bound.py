from fairseq.models.roberta import RobertaModel
import os
import sys
import argparse
from colorama import Fore, Back, Style
import torch

def int2hex(s):
    return s

def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return precision, recall, f1, accuracy

def evaluate_ida(ida_dir, truth_dir):
    print("\n" + "="*30)
    print("Evaluating IDA Pro Results")
    print("="*30)
    
    total_tp = total_fp = total_fn = total_tn = 0
    
    # We care about 'S' (Start) and 'B' (Body) vs '-' (None)
    # Usually for instruction boundary, 'S' is the most critical one to get right.
    # Or maybe we treat S and B as positive, - as negative?
    # Let's track metrics for 'S' specifically as it marks instruction start.
    
    tp_s = fp_s = fn_s = tn_s = 0
    
    try:
        filenames = os.listdir(ida_dir)
    except FileNotFoundError:
        print(f"Error: IDA directory '{ida_dir}' not found.")
        return

    if not filenames:
        print(f"No files found in '{ida_dir}'.")
        return

    print(f"{ 'Filename':<40} | { 'Prec (S)':<10} | { 'Recall (S)':<10} | { 'F1 (S)':<10}")
    print("-" * 80)

    for filename in filenames:
        try:
            f_ida = open(os.path.join(ida_dir, filename), 'r')
            f_truth = open(os.path.join(truth_dir, filename), 'r')
        except FileNotFoundError:
            print(f"Skipping {filename}: corresponding truth file not found.")
            continue
            
        file_tp = file_fp = file_fn = file_tn = 0
        
        for line_ida, line_truth in zip(f_ida, f_truth):
            parts_ida = line_ida.strip().split()
            parts_truth = line_truth.strip().split()
            
            if len(parts_ida) < 2 or len(parts_truth) < 2:
                continue
                
            label_ida = parts_ida[1]
            label_truth = parts_truth[1]
            
            # Evaluate 'S' (Start of Instruction)
            if label_truth == 'S':
                if label_ida == 'S':
                    file_tp += 1
                    tp_s += 1
                else:
                    file_fn += 1
                    fn_s += 1
            else: # truth is not S
                if label_ida == 'S':
                    file_fp += 1
                    fp_s += 1
                else:
                    file_tn += 1
                    tn_s += 1

        p, r, f1, _ = calculate_metrics(file_tp, file_fp, file_fn, file_tn)
        print(f"{filename[:40]:<40} | {p:.4f}     | {r:.4f}     | {f1:.4f}")
        
        f_ida.close()
        f_truth.close()
        
    print("-" * 80)
    avg_p, avg_r, avg_f1, _ = calculate_metrics(tp_s, fp_s, fn_s, tn_s)
    print(f"{ 'AVERAGE (Micro)':<40} | {avg_p:.4f}     | {avg_r:.4f}     | {avg_f1:.4f}")


def predict_and_evaluate(filename, model, truth_dir, ida_dir=None, start_idx=0, end_idx=512):
    truth_path = os.path.join(truth_dir, filename)
    if not os.path.exists(truth_path):
        print(f"Error: Truth file {truth_path} not found.")
        return

    print(f"\nAnalyzing: {filename}")
    
    # Read Ground Truth
    tokens = []
    truth_labels = []
    with open(truth_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                truth_labels.append(parts[1])
    
    # Read IDA (if provided)
    ida_labels = []
    if ida_dir:
        ida_path = os.path.join(ida_dir, filename)
        if os.path.exists(ida_path):
            with open(ida_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ida_labels.append(parts[1])
            # Align length if necessary
            min_len = min(len(truth_labels), len(ida_labels))
            if min_len < len(truth_labels):
                print(f"Warning: IDA file is shorter than truth file ({min_len} vs {len(truth_labels)}). Truncating.")
                tokens = tokens[:min_len]
                truth_labels = truth_labels[:min_len]
                ida_labels = ida_labels[:min_len]
        else:
            print(f"Warning: IDA file {ida_path} not found. Skipping IDA comparison for this file.")

    # Visualize Section
    print(f"\nVisualization (Indices {start_idx} to {end_idx}):")
    print("Legend: " + Fore.RED + "S (Start)" + Fore.RESET + ", " + Fore.GREEN + "B (Body)" + Fore.RESET + ", - (None)")
    
    print("\nGround Truth:")
    for i in range(start_idx, min(end_idx, len(tokens))):
        token = tokens[i]
        label = truth_labels[i]
        if label == 'S':
            print(f'{Fore.RED}{token}{Fore.RESET}', end=" ")
        elif label == 'B':
            print(f'{Fore.GREEN}{token}{Fore.RESET}', end=" ")
        else:
            print(f'{token}', end=" ")
    print(Style.RESET_ALL + '\n')

    if ida_labels:
        print("IDA Pro:")
        for i in range(start_idx, min(end_idx, len(ida_labels))):
            token = tokens[i] # reuse token from truth
            label = ida_labels[i]
            if label == 'S':
                print(f'{Fore.RED}{token}{Fore.RESET}', end=" ")
            elif label == 'B':
                print(f'{Fore.GREEN}{token}{Fore.RESET}', end=" ")
            else:
                print(f'{token}', end=" ")
        print(Style.RESET_ALL + '\n')

    # DeepBound Prediction
    # Process in chunks of 512
    deepbound_labels = []
    chunk_size = 510 # Leave room for start/end tokens if needed, though roberta.predict handles slicing often?
    # model.predict usually takes truncated input. Let's be safe.
    
    print("Running DeepBound Prediction...")
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        if not chunk: break
        
        encoded_tokens = model.encode(' '.join(chunk))
        # predict returns logprobs: [batch, len, classes]
        # We need to handle potential length mismatch due to BPE or truncation?
        # DeepBound usually maps 1-to-1 for this task if setup correctly.
        logprobs = model.predict('instbound', encoded_tokens) 
        preds = logprobs.argmax(dim=2).view(-1).cpu().numpy()
        
        # Map IDs to labels. 
        # Usually: 0 -> ?, 1 -> ?, 2 -> ?
        # Need to check dictionary. Usually based on `finetune_tasks/instbound.py`
        # But we don't have easy access to the exact dict mapping here without loading it.
        # However, looking at play_func_bound.py:
        # label 0: token
        # label 2: RED (Start)
        # label 1: GREEN (End)
        # This was for FUNCBOUND (F, R, -).
        
        # For INSTBOUND (S, B, -):
        # We should infer or check.
        # Let's assume standard mapping or print raw first.
        # Actually, let's use the model's task dictionary if possible.
        # model.task.label_dictionary
        
        for idx in preds:
            label_str = model.task.label_dictionary.symbols[idx + model.task.label_dictionary.nspecial]
            deepbound_labels.append(label_str)

    print("DeepBound Prediction:")
    for i in range(start_idx, min(end_idx, len(deepbound_labels))):
        token = tokens[i]
        label = deepbound_labels[i]
        if label == 'S':
            print(f'{Fore.RED}{token}{Fore.RESET}', end=" ")
        elif label == 'B':
            print(f'{Fore.GREEN}{token}{Fore.RESET}', end=" ")
        else:
            print(f'{token}', end=" ")
    print(Style.RESET_ALL + '\n')

    # Calculate DeepBound Metrics (S vs Not-S)
    tp = fp = fn = tn = 0
    # Also for B?
    # Let's focus on S (Start) as it's the boundary. 
    
    valid_len = min(len(truth_labels), len(deepbound_labels))
    for i in range(valid_len):
        t = truth_labels[i]
        p = deepbound_labels[i]
        
        if t == 'S':
            if p == 'S': tp += 1
            else: fn += 1
        else:
            if p == 'S': fp += 1
            else: tn += 1
            
    p, r, f1, acc = calculate_metrics(tp, fp, fn, tn)
    print(f"DeepBound Metrics (Class 'S'):")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")

    if ida_labels:
         # Calculate IDA Metrics (S vs Not-S) for this file
        tp_i = fp_i = fn_i = tn_i = 0
        valid_len_i = min(len(truth_labels), len(ida_labels))
        for i in range(valid_len_i):
            t = truth_labels[i]
            p = ida_labels[i]
            
            if t == 'S':
                if p == 'S': tp_i += 1
                else: fn_i += 1
            else:
                if p == 'S': fp_i += 1
                else: tn_i += 1
        p_i, r_i, f1_i, acc_i = calculate_metrics(tp_i, fp_i, fn_i, tn_i)
        print(f"\nIDA Metrics (Class 'S'):")
        print(f"Precision: {p_i:.4f}")
        print(f"Recall:    {r_i:.4f}")
        print(f"F1 Score:  {f1_i:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Play with Instruction Boundary Prediction')
    parser.add_argument('file', help='Filename in data-raw/inst_bound to analyze')
    parser.add_argument('--start', type=int, default=0, help='Start index for visualization')
    parser.add_argument('--end', type=int, default=100, help='End index for visualization')
    parser.add_argument('--ida-dir', help='Directory containing IDA Pro labeled files (optional)')
    parser.add_argument('--checkpoint-dir', default='checkpoints/finetune_instbound_elf', help='Checkpoint directory')
    parser.add_argument('--data-bin', default='data-bin/finetune_instbound_elf', help='Data bin directory')
    
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.checkpoint_dir}...")
    try:
        roberta = RobertaModel.from_pretrained(
            args.checkpoint_dir, 
            'checkpoint_best.pt', 
            args.data_bin, 
            bpe=None, 
            user_dir='finetune_tasks'
        )
        roberta.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Assuming data-raw/inst_bound is where truth files are
    truth_dir = 'data-raw/inst_bound'
    
    predict_and_evaluate(args.file, roberta, truth_dir, args.ida_dir, args.start, args.end)

if __name__ == '__main__':
    main()
