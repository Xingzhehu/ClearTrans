from dataclasses import asdict, dataclass
import json

from .detectors import InvalidProgram
from .prompts import parse_response


@dataclass
class Observation:
    instance_id: str
    format_ok: bool = False
    source_count: int = 0
    clean_count: int = 0
    target_count: int = 0
    clean_correct: bool = False
    target_correct: bool = False
    clean_analyzable: bool = False
    target_analyzable: bool = False
    failure: str = ""
    response: str = ""


def observe(row, response, executor, detector):
    result = Observation(instance_id=row["id"], response=response)
    try:
        parsed = parse_response(response)
    except ValueError as exc:
        result.failure = str(exc)
        return result
    result.format_ok = True
    result.source_count = detector.analyze(row["source_code"], row["source_lang"])["count"]
    for prefix, code, lang in (("clean", parsed["clean_source"], row["source_lang"]),
                               ("target", parsed["target_code"], row["target_lang"])):
        execution = executor.run(code, lang, row["tests"])
        setattr(result, prefix + "_correct", execution["correct"])
        try:
            report = detector.analyze(code, lang)
        except InvalidProgram:
            setattr(result, prefix + "_correct", False)
            continue
        setattr(result, prefix + "_analyzable", True)
        setattr(result, prefix + "_count", report["count"])
    return result


def group_rewards(observations, alpha=0.2, epsilon=1e-6, mode="dual"):
    if not 0 <= alpha <= 1 or epsilon <= 0:
        raise ValueError("Invalid reward hyperparameters")
    if len({r.instance_id for r in observations}) != 1:
        raise ValueError("A reward group must contain one source instance")
    maximum = max((r.target_count for r in observations if r.format_ok and r.target_analyzable), default=0)
    scores = []
    for r in observations:
        cot = (2 - min(1, r.clean_count / (r.source_count + epsilon))) if r.clean_correct and r.clean_analyzable else 0.0
        target = (2 - r.target_count / (maximum + epsilon)) if r.target_correct and r.target_analyzable else 0.0
        score = alpha * cot + (1 - alpha) * target
        if mode == "reasoning_only":
            score = cot
        elif mode == "translation_only":
            score = target
        elif mode != "dual":
            raise ValueError(f"Unknown reward mode: {mode}")
        scores.append(score if r.format_ok else 0.0)
    return scores


def batched_group_rewards(observations, group_size=12, **kwargs):
    if not observations or len(observations) % group_size:
        raise ValueError("Global rollout batch must contain complete contiguous groups")
    scores = []
    for start in range(0, len(observations), group_size):
        scores.extend(group_rewards(observations[start:start + group_size], **kwargs))
    return scores
