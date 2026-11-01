from collections import defaultdict
import random

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Name, Text

from .common import digest, read_jsonl, validate_instance, write_jsonl


def normalized_tokens(code, lang):
    tokens = []
    for kind, value in lex(code, get_lexer_by_name(lang)):
        if kind in Comment or (kind in Text and value.isspace()):
            continue
        if kind in Name:
            tokens.append("IDENTIFIER")
        elif value.strip():
            tokens.append(value)
    return tokens


def shingles(tokens, size=5):
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


class DuplicateIndex:
    """Exact normalized hashes plus explicit approximate token-shingle Jaccard matching."""

    def __init__(self, threshold=0.9):
        self.threshold, self.exact = threshold, set()
        self.sets, self.postings = [], defaultdict(set)

    def duplicate(self, code, lang):
        tokens = normalized_tokens(code, lang)
        if (lang, digest(tokens)) in self.exact:
            return True
        query = shingles(tokens)
        candidates = set()
        for item in query:
            candidates.update(self.postings.get((lang, item), set()))
        for index in candidates:
            reference = self.sets[index]
            if min(len(query), len(reference)) / max(len(query), len(reference)) < self.threshold:
                continue
            if len(query & reference) / len(query | reference) >= self.threshold:
                return True
        return False

    def add(self, code, lang):
        tokens = normalized_tokens(code, lang)
        self.exact.add((lang, digest(tokens)))
        values = shingles(tokens)
        index = len(self.sets)
        self.sets.append(values)
        for item in values:
            self.postings[(lang, item)].add(index)


def prepare(train_path, benchmark_paths, output, config):
    benchmarks = [validate_instance(row) for path in benchmark_paths for row in read_jsonl(path)]
    if not benchmarks:
        raise ValueError("Evaluation records are required for leakage checks")
    problem_ids = {row["problem_id"] for row in benchmarks}
    benchmark_index = DuplicateIndex(config.get("near_duplicate_jaccard", 0.9))
    for row in benchmarks:
        benchmark_index.add(row["source_code"], row["source_lang"])
        if row.get("target_code"):
            benchmark_index.add(row["target_code"], row["target_lang"])
    kept, rejected, seen = [], [], set()
    training_indexes = defaultdict(lambda: DuplicateIndex(config.get("near_duplicate_jaccard", 0.9)))
    for original in read_jsonl(train_path):
        row = validate_instance(original)
        if row["id"] in seen:
            raise ValueError(f"Duplicate instance ID: {row['id']}")
        seen.add(row["id"])
        direction = (row["source_lang"], row["target_lang"])
        reason = ("benchmark_problem_overlap" if row["problem_id"] in problem_ids else
                  "benchmark_code_overlap" if benchmark_index.duplicate(row["source_code"], row["source_lang"]) else
                  "duplicate_training_code" if training_indexes[direction].duplicate(row["source_code"], row["source_lang"]) else None)
        if reason:
            rejected.append({"id": row["id"], "reason": reason})
        else:
            training_indexes[direction].add(row["source_code"], row["source_lang"])
            kept.append(row)
    rng = random.Random(config.get("seed", 42))
    problems = sorted({row["problem_id"] for row in kept})
    rng.shuffle(problems)
    sft_count, rl_count = config.get("sft_per_direction", 10000), config.get("rl_per_direction", 2000)
    cut = int(len(problems) * sft_count / (sft_count + rl_count))
    sft_problems = set(problems[:cut])
    buckets = {"sft": defaultdict(list), "rl": defaultdict(list)}
    for row in kept:
        split = "sft" if row["problem_id"] in sft_problems else "rl"
        buckets[split][(row["source_lang"], row["target_lang"])].append(row)
    directions = {(r["source_lang"], r["target_lang"]) for r in kept}
    selected = {}
    for split, count in (("sft", sft_count), ("rl", rl_count)):
        selected[split] = []
        for direction in sorted(directions):
            rows = buckets[split][direction]
            rng.shuffle(rows)
            if len(rows) < count:
                raise ValueError(f"{split} {direction}: need {count}, have {len(rows)} after leakage checks; "
                                 "supply more data or explicitly change the requested scale")
            selected[split].extend(rows[:count])
    if not selected["sft"] or not selected["rl"]:
        raise ValueError("Preparation produced an empty split")
    from pathlib import Path
    for split, rows in selected.items():
        write_jsonl(Path(output) / f"{split}.jsonl", rows)
    write_jsonl(Path(output) / "rejected.jsonl", rejected)
    return {"input": len(seen), "kept_before_sampling": len(kept), "rejected": len(rejected),
            "sft": len(selected["sft"]), "rl": len(selected["rl"]), "directions": sorted(directions)}
