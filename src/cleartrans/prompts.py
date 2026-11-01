import re

CLEAN_BEGIN = "[Begin Clean Source Code]"
CLEAN_END = "[End Clean Source Code]"


def student_prompt(row):
    return (
        f"Remove inherited code smells from this {row['source_lang']} program and translate it "
        f"into correct, maintainable, smell-free {row['target_lang']} code. Preserve stdin/stdout behavior.\n"
        "Return exactly <think>reasoning</think> followed by one fenced target-code block. "
        f"Inside <think>, explicitly include the complete refactored source between {CLEAN_BEGIN} "
        f"and {CLEAN_END}. Do not include the full target program inside <think>.\n\n"
        f"Source code:\n```{row['source_lang']}\n{row['source_code']}\n```"
    )


def direct_prompt(row):
    return (f"Translate the {row['source_lang']} code to {row['target_lang']} code, ensuring functional "
            f"correctness and avoiding code smells. Preserve stdin/stdout behavior. "
            f"Output only one fenced code block.\n\n{row['source_code']}")


def refactor_prompt(row, code, report, feedback=None):
    return (f"You are a senior {row['source_lang']} engineer. Refactor this code without changing "
            "its functionality, input/output formats, or edge-case behavior. Fix all reported smells. "
            "Moderate refactoring is allowed. Return only [BEGIN] followed by the complete code and [END].\n"
            f"Code:\n{code}\n\nStatic analysis:\n{report}\nExecution feedback:\n{feedback or ''}")


def translation_prompt(row, clean_source, code=None, feedback=None):
    return (f"You are a senior code translation engineer. Translate the {row['source_lang']} program "
            f"to {row['target_lang']}. Preserve exact functionality and stdin/stdout behavior. "
            "Avoid code smells and use target-language idioms. "
            "Return only [BEGIN] followed by the complete target code and [END].\n"
            f"Clean source:\n{clean_source}\nPrevious draft:\n{code or ''}\nFeedback:\n{feedback or ''}")


def synthesis_prompt(row, clean_source, target):
    return (student_prompt(row) + "\n\nSynthesis-only reference anchors (not included in the student's input):\n"
            f"Refactored source:\n{clean_source}\nFinal target:\n{target}\n\n"
            "Write only a reasoning trace connecting the original source, the refactored source, and "
            "the translation plan. Start with 'Let's think step by step'. Analyze source behavior, "
            "identify inherited smells or explain why no changes are required, present the complete "
            f"refactored source verbatim between {CLEAN_BEGIN} and {CLEAN_END}, and discuss target-language "
            "idioms and translation choices. Do not output the complete target program. "
            "Describe the derivation naturally, without referring to the reference answers. "
            "Return only the trace, without outer <think> tags.")


def judge_prompt(code):
    return ("You are an experienced software engineer performing a qualitative code review. "
            "Identify meaningful design, maintainability, or readability problems using only the "
            "provided code. Do not assume missing files or force predefined labels. "
            "Start with YES if at least one issue is found, otherwise NO. For YES, write Findings: "
            "followed by numbered issues, each with a short name, code evidence, and why it matters. "
            f"For NO, briefly explain why the code is clean.\n\nCode:\n{code}")


def unwrap_code(text):
    text = text.strip()
    match = re.fullmatch(r"\[BEGIN\]\s*(.*?)\s*\[END\]", text, re.S)
    if match:
        text = match.group(1).strip()
    match = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", text, re.S)
    return match.group(1).strip() if match else text


def parse_response(text):
    match = re.fullmatch(r"\s*<think>\s*(.*?)\s*</think>\s*```[^\n]*\n(.*?)\n?```\s*", text, re.S)
    if not match or text.count("<think>") != 1 or text.count("</think>") != 1:
        raise ValueError("Expected a single think block and a single final fenced code block")
    cot, target = match.groups()
    tail = text.split("</think>", 1)[1]
    if len(re.findall(r"^\s*```", tail, re.M)) != 2:
        raise ValueError("Expected exactly one final fenced code block")
    if cot.count(CLEAN_BEGIN) != 1 or cot.count(CLEAN_END) != 1:
        raise ValueError("Missing or repeated clean-source markers")
    clean = re.search(re.escape(CLEAN_BEGIN) + r"(.*?)" + re.escape(CLEAN_END), cot, re.S)
    if not clean or not target.strip():
        raise ValueError("Empty or malformed anchors")
    clean_source = unwrap_code(clean.group(1))
    if not clean_source:
        raise ValueError("Empty clean-source anchor")
    return {"cot": cot.strip(), "clean_source": clean_source, "target_code": target.strip()}


def response_segments(cot, target, target_lang):
    return f"<think>\n{cot.strip()}\n</think>\n", f"```{target_lang}\n{target.strip()}\n```"
