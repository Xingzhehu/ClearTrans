from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path

from .common import append_jsonl, guard_output, read_jsonl, record_run, validate_instance
from .execution import DockerExecutor
from .prompts import student_prompt
from .rewards import batched_group_rewards, observe
from .scoring import tokenize_record


def train_sft(input_path, config, resume=None):
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    cfg = config["sft"]
    set_seed(cfg.get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    counts = Counter()
    examples = []
    for row in read_jsonl(input_path):
        counts[row["id"]] += 1
        prefix, thought, target = tokenize_record(row, tokenizer)
        continuation = thought + target + [tokenizer.eos_token_id]
        if len(prefix) + len(continuation) > cfg.get("max_length", 8192):
            raise ValueError(f"SFT context overflow: {row['candidate_id']}; do not silently truncate anchors")
        examples.append({"input_ids": prefix + continuation, "labels": [-100] * len(prefix) + continuation})
    if not examples or len(set(counts.values())) != 1 or next(iter(counts.values())) != cfg.get("k", 3):
        raise ValueError("SFT requires exactly k selected responses for every source")

    def collate(rows):
        width = max(len(r["input_ids"]) for r in rows)
        return {"input_ids": torch.tensor([r["input_ids"] + [tokenizer.pad_token_id] * (width - len(r["input_ids"])) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]),
                "labels": torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows])}

    class SequenceSumTrainer(Trainer):
        # Equation L_sft sums target tokens, then averages equally over responses.
        # Fixed k makes uniform response sampling equivalent to averaging over sources.
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            losses = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            loss = losses.view(labels.shape).sum(dim=1).mean()
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32)
    model.config.use_cache = False
    args = TrainingArguments(output_dir=cfg["output_dir"], num_train_epochs=cfg.get("epochs", 3),
        learning_rate=cfg.get("learning_rate", 1e-5), seed=cfg.get("seed", 42), data_seed=cfg.get("seed", 42),
        per_device_train_batch_size=cfg.get("batch_size", 1), gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        bf16=cfg.get("bf16", True), gradient_checkpointing=True, deepspeed=cfg.get("deepspeed"),
        logging_steps=cfg.get("logging_steps", 10), save_strategy="epoch", save_total_limit=2,
        report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False)
    trainer = SequenceSumTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples),
                                 data_collator=collate, processing_class=tokenizer)
    trainer.model_accepts_loss_kwargs = False
    if trainer.is_world_process_zero():
        guard_output(Path(cfg["output_dir"]) / "training.jsonl", config, [input_path], "train-sft")
        record_run(cfg["output_dir"], config)
    trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(cfg["output_dir"])


def train_grpo(input_path, config, resume=None):
    import torch
    import trl
    from accelerate.utils import gather_object
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from .detectors import SonarDetector
    if trl.__version__ != "0.22.1":
        raise RuntimeError("The group reward adapter targets TRL 0.22.1; install the pinned training extras")
    cfg = config["grpo"]
    executor = DockerExecutor(config["execution"])
    detector = SonarDetector(config["sonarqube"], executor)
    rows = [validate_instance(row) for row in read_jsonl(input_path)]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("RL requires nonempty data with unique instance IDs")
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": student_prompt(row)}],
                                  "instance_json": json.dumps(row)} for row in rows])

    def cleartrans_reward(**kwargs):
        raise RuntimeError("Group reward must be computed by ClearTransGRPOTrainer, not a per-response callback")

    class ClearTransGRPOTrainer(GRPOTrainer):
        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            local, error = [], None
            try:
                if not hasattr(self, "_verified_sources"):
                    self._verified_sources = set()
                for example, completion in zip(inputs, completions):
                    row = json.loads(example["instance_json"])
                    if row["id"] not in self._verified_sources:
                        if not executor.run(row["source_code"], row["source_lang"], row["tests"])["correct"]:
                            raise ValueError(f"RL source fails reference tests: {row['id']}")
                        self._verified_sources.add(row["id"])
                    text = completion if isinstance(completion, str) else completion[0]["content"]
                    local.append(observe(row, text, executor, detector))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            # Every rank participates even if a local analyzer failed. No partial-group normalization.
            bundles = gather_object([{"observations": local, "error": error}])
            errors = [b["error"] for b in bundles if b["error"]]
            if errors:
                raise RuntimeError("Reward evaluation failed: " + "; ".join(errors))
            global_observations = [r for bundle in bundles for r in bundle["observations"]]
            scores = batched_group_rewards(global_observations, group_size=self.num_generations,
                alpha=cfg.get("alpha", 0.2), epsilon=cfg.get("epsilon", 1e-6), mode=cfg.get("reward_mode", "dual"))
            if self.accelerator.is_main_process:
                append_jsonl(Path(cfg["output_dir"]) / "reward_audit.jsonl", {
                    "step": self.state.global_step, "observations": [asdict(r) for r in global_observations], "rewards": scores})
            # TRL expects this method to return a GLOBAL tensor, not a per-rank tensor.
            return torch.tensor(scores, dtype=torch.float32, device=self.accelerator.device).unsqueeze(1)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Reject long prompts before TRL can truncate the source program.
    for row in rows:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": student_prompt(row)}],
                                            tokenize=True, add_generation_prompt=True)
        if len(ids) > cfg.get("max_prompt_length", 4096):
            raise ValueError(f"GRPO prompt overflow: {row['id']}")
    args = GRPOConfig(output_dir=cfg["output_dir"], num_train_epochs=cfg.get("epochs", 10),
        learning_rate=cfg.get("learning_rate", 1e-6), num_generations=cfg.get("group_size", 12),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 3),
        steps_per_generation=cfg.get("steps_per_generation", 3),
        max_prompt_length=cfg.get("max_prompt_length", 4096), max_completion_length=cfg.get("max_completion_length", 4096),
        temperature=cfg.get("temperature", 0.7), beta=cfg.get("beta", 0.04),
        loss_type="grpo", scale_rewards="group", num_iterations=1,
        bf16=cfg.get("bf16", True), gradient_checkpointing=True, deepspeed=cfg.get("deepspeed"),
        seed=cfg.get("seed", 42), data_seed=cfg.get("seed", 42), remove_unused_columns=False,
        report_to="none", logging_steps=cfg.get("logging_steps", 1), save_strategy="epoch", save_total_limit=2,
        use_vllm=False, model_init_kwargs={"torch_dtype": torch.bfloat16 if cfg.get("bf16", True) else torch.float32})
    trainer = ClearTransGRPOTrainer(model=cfg["model"], args=args, train_dataset=dataset,
                                    reward_funcs=cleartrans_reward, processing_class=tokenizer)
    if trainer.accelerator.is_main_process:
        guard_output(Path(cfg["output_dir"]) / "training.jsonl", config, [input_path], "train-grpo")
        record_run(cfg["output_dir"], config)
    trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg["output_dir"])
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(cfg["output_dir"])
