from collections import defaultdict
import random

from .common import read_jsonl, write_jsonl
from .prompts import response_segments, student_prompt


def tokenize_record(row, tokenizer):
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": student_prompt(row)}],
                                           tokenize=True, add_generation_prompt=True)
    thought, target = response_segments(row["cot"], row["target_code"], row["target_lang"])
    thought = tokenizer.encode(thought, add_special_tokens=False)
    target = tokenizer.encode(target, add_special_tokens=False)
    if not prefix or not thought or not target:
        raise ValueError("Empty model token segment")
    return prefix, thought, target


def score_candidates(input_path, output_path, config):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = config["scoring"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype,
                                                device_map=cfg.get("device_map", "auto"))
    model.eval()
    results = []
    for row in read_jsonl(input_path):
        if row.get("status") != "ok":
            results.append(row)
            continue
        prefix, thought, target = tokenize_record(row, tokenizer)
        ids = prefix + thought + target
        if len(ids) > cfg.get("max_length", 8192):
            results.append({"id": row["id"], "candidate_id": row["candidate_id"],
                            "status": "rejected", "reason": "scorer_context_overflow"})
            continue
        inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False).logits[0, :-1]
            labels = inputs[0, 1:]
            # Compute token NLL in chunks, avoiding a second full-vocabulary float32 copy.
            nll = torch.cat([F.cross_entropy(logits[start:start + 128].float(), labels[start:start + 128], reduction="none")
                             for start in range(0, len(labels), 128)])
            begin, middle, end = len(prefix) - 1, len(prefix) + len(thought) - 1, len(ids) - 1
            plausibility = -nll[begin:middle].mean().item()
            effectiveness = -nll[middle:end].mean().item()
        results.append({**row, "score_pla": plausibility, "score_eff": effectiveness,
                        "score": plausibility + effectiveness, "rating_model": cfg["model"],
                        "cot_tokens": len(thought), "target_tokens": len(target)})
    write_jsonl(output_path, results)


def select_candidates(input_path, output_path, k=3, mode="dual", seed=42):
    if k < 1:
        raise ValueError("k must be positive")
    groups, all_ids, seen = defaultdict(list), set(), set()
    for row in read_jsonl(input_path):
        all_ids.add(row["id"])
        if row["candidate_id"] in seen:
            raise ValueError(f"Duplicate candidate ID: {row['candidate_id']}")
        seen.add(row["candidate_id"])
        if row.get("status") == "ok":
            groups[row["id"]].append(row)
    selected = []
    rng = random.Random(seed)
    for instance_id in sorted(all_ids):
        rows = sorted(groups[instance_id], key=lambda r: r["candidate_id"])
        if len(rows) < k:
            raise ValueError(f"{instance_id}: only {len(rows)} valid candidates; need {k}. Regenerate or explicitly exclude this source")
        if mode == "random":
            rng.shuffle(rows)
        else:
            key = {"dual": "score", "plausibility": "score_pla", "effectiveness": "score_eff"}[mode]
            rows.sort(key=lambda r: r[key], reverse=True)
        for rank, row in enumerate(rows[:k], 1):
            selected.append({**row, "selected_rank": rank, "selected_k": k, "selection_mode": mode})
    if not selected:
        raise ValueError("No selected candidates")
    write_jsonl(output_path, selected)
    return {"sources": len(all_ids), "responses": len(selected), "k": k, "mode": mode}
