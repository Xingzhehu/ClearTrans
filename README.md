# ClearTrans replication implementation

对应论文：**Eliminating Code Smells in Code Translation with Anchor-to-Thought Chain-of-Thought Synthesis**。

本仓库依据论文和补充材料实现 ClearTrans 的训练与评测流程，提供可配置的源码和脚本。当前版本为按论文重新整理的实现，不含预填的实验结果；原始数据、规则快照和训练权重需要另行接入。

## 实现内容

| 论文环节 | 入口 | 实现文件 |
|---|---|---|
| 问题级隔离、代码去重、SFT/RL 划分 | `prepare` | `src/cleartrans/data.py` |
| 源程序异味清除、目标翻译与反馈修复 | `synthesize` | `src/cleartrans/synthesis.py` |
| 三个程序锚点连接为 CoT | `synthesize` | `src/cleartrans/prompts.py` |
| 两项平均对数似然评分 | `score` | `src/cleartrans/scoring.py` |
| 每个源程序选择 top-k | `select` | `src/cleartrans/scoring.py` |
| Multi-CoT SFT | `train-sft` | `src/cleartrans/training.py` |
| 双奖励 GRPO，全局采样组归一化 | `train-grpo` | `src/cleartrans/rewards.py`, `training.py` |
| 五种语言编译与测试 | 自动调用 | `docker/runner.py`, `execution.py` |
| SonarQube、PMD、Pylint、LLM 评测 | `evaluate` | `src/cleartrans/detectors.py` |
| 单次翻译、直接提示、翻译后迭代修复 | `infer` | `src/cleartrans/evaluation.py` |
| CA、SFA、异味数量与规则分布、实测 token 用量 | `summarize` | `src/cleartrans/evaluation.py` |
| 五位评估者、每例三人、匿名材料与共识汇总 | `human-export`, `human-import` | `src/cleartrans/evaluation.py` |

训练与合成只使用 SonarQube，其他检测器只用于评测。没有用 MI、Ruff 或其他工具代替论文中的 SonarQube 奖励。

## 目录与安装

所有命令从项目根目录执行。Python 使用 `conda activate coder`；脚本会自动激活该环境。只安装基础代码：

```bash
git clone https://github.com/Xingzhehu/ClearTrans.git
cd ClearTrans
bash scripts/setup_env.sh core
bash scripts/check.sh
```

需要训练时，在 Linux + CUDA 环境安装固定的训练依赖：

```bash
bash scripts/setup_env.sh train
```

训练适配器固定使用 TRL **0.22.1**，因为自定义奖励需要覆盖其全局 reward 聚合接口。主要训练依赖固定在 `pyproject.toml`。基础测试使用标准库 `unittest`，不需要下载模型。

本次编写未安装服务器环境、下载权重、构建容器或启动 GPU 训练。默认训练启动脚本使用 8 张 GPU；论文原实验为 8×A800 80GB，不能据此保证默认长上下文配置在 L20 上的显存占用。可调整批量、累积步数和上下文长度，但须记录配置差异；源程序、锚点和答案不做静默截断。

## 第一步：准备执行和检测环境

构建一次性代码执行容器：

```bash
bash scripts/build_runner.sh
bash scripts/run.sh doctor
bash scripts/run.sh check-data --input examples/instances.jsonl --execute
```

容器支持 C11、C++17、Go、Java、Python 的 stdin/stdout 单文件程序。Java 按公共类名保存源文件；暂不处理 package 或多文件工程。每个候选在无网络、非 root、限制内存/CPU/PID 的容器中编译和执行，每个测试默认 10 秒。输出比较默认忽略空白差异；需要字节一致时，将 `execution.comparison` 改为 `exact`。自定义 checker 和浮点容差尚未实现，这类基准样本必须先接入其官方 checker，不能直接声称本实现完全覆盖。

另行提供 SonarQube Server **2025.1 LTA**、SonarScanner CLI 和对应语言分析器。C/C++ 的 CFamily 分析要求该版本支持的商业版本/许可，不能用不分析 C/C++ 的环境产生“零异味”结果。扫描主机需有与容器构建环境匹配的 GCC/G++；Java 分析所需 `.class` 文件由隔离容器编译后取回。编译数据库保留同样的标准和优化选项，并将构建路径映射到扫描工作目录；复现时应固定容器与扫描主机编译器版本。

```bash
export SONAR_HOST_URL='http://your-sonarqube-host:9000'
read -r -s -p 'Sonar token: ' SONAR_TOKEN
export SONAR_TOKEN
```

token 需要读取分析配置、创建分析项目、提交分析、关联质量配置及删除本包创建的临时项目的权限。临时项目名带 `cleartrans_repro_` 和随机 UUID；不会删除已有研究项目。

先在 `configs/paper.yaml` 填好五种语言的**实际质量配置名**，再导出规则和插件快照：

```bash
bash scripts/run.sh sonar-snapshot
```

快照写入 `artifacts/sonar_manifest.json`，包括服务器版本、插件版本和全部五种语言的配置导出 XML。之后检测会核对快照并在规则发生变化时停止。默认 `Sonar way` 只是可配置的起点，并不是恢复得到的原实验规则表。发布复现包前应附上真正使用的规则快照。

Sonar 检测等待计算任务成功，确认实际分析了指定语言，分页读取未解决的 `CODE_SMELL`；不将 BUG/漏洞混入该计数。检测失败、未覆盖指定语言、扫描超时均报错，不按无异味处理。重复代码等只有被当前规则配置报告成 `CODE_SMELL` 才进入该指标；独立 duplication 指标不会自动混算。

## 第二步：接入原始数据

统一 JSONL 格式，每行一个翻译实例：

```json
{"id":"instance-0001","problem_id":"codeforces:123:A","source_lang":"python","target_lang":"cpp","source_code":"a,b=map(int,input().split())\nprint(a+b)","tests":[{"input":"2 3\n","output":"5\n"}]}
```

- `id` 在文件中唯一；`problem_id` 必须在训练集和基准间采用一致的全局问题标识。
- 语言使用 `c`、`cpp`、`go`、`java`、`python`。`C++`、`py` 等常用别名会规范化。
- 测试输入与期望输出均为字符串，至少有一个测试；使用真实基准测试，不生成虚构标签。
- 基准可额外提供 `target_code`，供训练/评测间的跨端代码去重。
- 示例仅用于格式和环境检查，不是论文训练集或基准结果。
- 没有捏造 CodeNet、F2SBench、xCodeEval 的下载地址或假定它们具有相同的原生结构；将实际数据按此 schema 转换后再运行。

```bash
bash scripts/run.sh check-data --input data/raw_train.jsonl
bash scripts/run.sh prepare --input data/raw_train.jsonl \
  --benchmarks data/codenet.jsonl data/f2sbench.jsonl data/xcodeeval.jsonl \
  --output-dir data/prepared
bash scripts/run.sh check-data --input data/prepared/rl.jsonl --execute
```

默认每个方向抽取 SFT 10,000 例和 RL 2,000 例。问题级划分先于抽样，两个阶段不会共享问题。去重包含去注释/空白/标识符后的哈希匹配，以及显式的 token-shingle Jaccard 近似匹配；不足请求规模时直接报错。其 near-duplicate 方法是本次重建的实现选择，并不能保证发现所有语义重复。程序会输出实际排除清单，不预设“没有重叠”。

## 第三步：构造训练样本

教师通过 OpenAI-compatible HTTP 接口调用。接口地址、服务模型名和环境变量名在 YAML 的 `teacher` 中配置；可使用现有兼容服务。默认对应非推理 Qwen2.5-32B 的 Instruct 服务形式。

若使用 vLLM，需先在 `coder` 中另行安装与本机 CUDA/PyTorch 相容的版本。脚本不自动替换训练依赖。启动示例：

```bash
# 单独终端启动；也可使用其他机器上的兼容接口
bash scripts/serve_model.sh /path/to/Qwen2.5-32B-Instruct Qwen2.5-32B-Instruct 8000 8
```

```bash
bash scripts/run.sh synthesize --input data/prepared/sft.jsonl --output artifacts/candidates.jsonl
bash scripts/run.sh score --input artifacts/candidates.jsonl --output artifacts/scored.jsonl
bash scripts/run.sh select --input artifacts/scored.jsonl --output artifacts/selected.jsonl --k 3
```

每个源程序尝试 15 次完整的锚点构建与 CoT 合成。源/目标锚点只有通过所有测试且无 Sonar 异味才被接受；CoT 必须保留验证过的源锚点。失败样本保留拒绝原因。每个源程序少于 k 个有效候选时，选择阶段报错，需补充生成或明确记录剔除。

评分使用学生 backbone，分别计算 `log p(cot | src)` 与 `log p(tgt | src, cot)` 的平均 token 对数似然，然后相加。模型输入仅有源程序和固定任务提示，不包含教师持有的参考锚点。评分与 SFT 共用分段 tokenization；`<think>` 标记计入 reasoning 段，代码围栏计入目标段；EOS 参与 SFT，不参与评分。长度不足时拒绝，绝不截掉锚点后继续评分。

合成、推理、评测支持按 ID 继续追加；旁路 provenance 文件阻止以不同配置或输入误续跑旧结果。评分和选择会原子重写结果。修改规则快照或提示实现后，应使用新输出目录。合成中被拒绝的 candidate_id 被保留并视为已尝试，不自动再次付费生成；补跑可换一个输出文件并合并去重，或在明确保留审计后重新开始。

## 第四步：SFT 和 GRPO

```bash
bash scripts/train_sft.sh artifacts/selected.jsonl
bash scripts/train_grpo.sh data/prepared/rl.jsonl
```

SFT：3 epochs，学习率 `1e-5`，默认 k=3；对回复 token 的负对数似然求和，再按样本平均，源 prompt 不计损失。GRPO：10 epochs，学习率 `1e-6`，每组 12 个候选，`alpha=0.2`。

```text
R_cot = I(clean source correct) * (2 - min(1, N_clean/(N_source+epsilon)))
R_tgt = I(target correct)       * (2 - N_target/(N_group_max+epsilon))
R     = I(format correct)       * (alpha*R_cot + (1-alpha)*R_tgt)
```

程序先在各 rank 执行检测，再汇总**全局完整的候选组**计算 `N_group_max`，避免每个 GPU 各算一个分母。默认 8 GPU × 每卡 batch 1 × generation steps 3 = 24 个候选，即两个完整组。改变卡数时，应保证全局 generation batch 是 12 的整数倍。单 GPU 可将 generation steps/累积步数改为 12。

无合法格式的回复得零分。不能编译的锚点不得正确性奖励；无法分析的语法无效目标也不作为最大异味计数的候选。能编译但测试失败的目标仍参与异味分母。这个无效程序边界处理是明确记录的实现选择；Sonar 服务故障则停止整个步骤，不产生替代奖励。审计保存每个候选的原始输出、检测数量、正确性标记和最终奖励。

支持恢复：`bash scripts/train_grpo.sh data/prepared/rl.jsonl --resume artifacts/grpo/checkpoint-...`。

## 第五步：推理和评测

```bash
# 在另一个终端服务模型；YAML 的 inference.model 与服务别名相同
bash scripts/serve_model.sh artifacts/grpo cleartrans 8001 8

bash scripts/run.sh infer --input data/codenet.jsonl --output artifacts/predictions.jsonl --method cleartrans
bash scripts/run.sh evaluate --input artifacts/predictions.jsonl --output artifacts/eval_sonar.jsonl --detector sonarqube
bash scripts/run.sh evaluate --input artifacts/predictions.jsonl --output artifacts/eval_pylint.jsonl --detector pylint
bash scripts/run.sh evaluate --input artifacts/predictions.jsonl --output artifacts/eval_pmd.jsonl --detector pmd
bash scripts/run.sh evaluate --input artifacts/predictions.jsonl --output artifacts/eval_llm.jsonl --detector llm
bash scripts/run.sh summarize --inputs artifacts/eval_sonar.jsonl --output artifacts/summary.json
```

Pylint 仅评估 Python 目标，PMD 仅评估 Java 目标，输出会明确记录跳过的其他语言例数。另行安装固定版本 Pylint 3.3.8 和 PMD 7.17.0，将命令路径写入配置。规则文件已提供，但它们是重建时的可执行默认配置，必须与真实实验配置核对后再做数值比较。LLM evaluator 使用 `judge` 中的独立接口，其 YES/NO 不被伪造为可数的异味数量。

- CA：所有测试通过的程序数 / 全部输入程序数。
- SFA：所有测试通过且所选 evaluator 判为无异味的程序数 / 全部输入程序数。
- 功能失败记入两项指标的分母；检测服务失败中断，不能当作无异味。
- 数量分布只在功能正确且有数值检测报告的程序上统计；LLM 没有数量时使用 JSON `null`。
- token 来自接口返回的真实 usage；translate-and-repair 累加全部生成请求。接口不返回时为 `null`，不会模拟数据。
- `--method direct` 为单次直接翻译；`--method repair` 为先翻译再最多 3 次反馈修复。配置基线服务后使用独立输出文件。
- ExeCoder、F2STrans 应使用其真实模型生成的输出转换为统一 schema 后评估；本包没有把简单提示冒充其官方实现。
- 多 seed 实验修改配置中的 seed，并使用独立模型、输出路径。建议 42/43/44。summary 分 seed 输出，同时计算均值和样本标准差，并要求各 seed 的实例集合一致。

推理和评测完成后，应核对输出 ID 与基准 ID 集合一致，再报告结果。使用 `check-data` 检查 schema/ID，基准缺失测试不能以空测试补齐。

## 人工评测

使用各方法在同一批实例上的已评测输出：

```bash
bash scripts/run.sh human-export --inputs artifacts/method_a_eval.jsonl artifacts/method_b_eval.jsonl \
  --output-dir artifacts/human --sample-size 300
bash scripts/run.sh human-import --labels artifacts/human/independent_labels.jsonl \
  --consensus artifacts/human/consensus.jsonl --mapping artifacts/human/PRIVATE_mapping.jsonl \
  --output artifacts/human/results.json
```

每个方法输入文件的 `method` 字段必须唯一，例如 `cleartrans-7b`、`gpt5-direct`；不能让不同模型都叫 `direct` 后混在一起。该字段可在导入外部模型输出时设置。

`assessor_materials.jsonl` 只包含随机编号、代码、语言和被分配的三位评估者；**不要将 PRIVATE_mapping.jsonl 发给评估者**。独立标签格式：

```json
{"blind_id":"sample_00000","assessor":"A","smell_free":true}
```

讨论后另存共识文件，不覆盖独立标签：

```json
{"blind_id":"sample_00000","smell_free":true}
```

统计独立讨论前的三人全体一致率，并与最终共识 SFA 分开。这里没有把一致率冒称 Cohen/Fleiss kappa，也没有将人工异味评测与查看 Sonar 报告的 inherited/introduced 来源标注任务混算。来源标注需要按其不同的信息条件单独组织材料和结果。当前随机抽样覆盖共有实例集合，不保证每方向固定配额；要复现固定分层采样，应提供原始抽样清单。

## 消融与发布核对

- 评分消融：`select --mode plausibility`、`effectiveness`、`random`。
- Multi-CoT 消融：`select --k 1/3/5`，同时修改 `sft.k`，每种设置重新训练。
- 奖励消融：`grpo.reward_mode` 为 `dual`、`reasoning_only`、`translation_only`。
- teacher/backbone/alpha 消融：改独立 YAML，使用独立候选和输出目录。
- 仅 SFT：直接服务 `artifacts/sft` 并用同一评测流程。

详细论文映射和不能从论文恢复的设置见 `docs/METHOD_MAPPING.md`。验证记录见 `VALIDATION.md`。

作为公开复现包发布前，应附上实际数据清单/来源与许可、划分 ID、真实规则快照、模型 checkpoint/revision、完整环境清单和实际运行日志。这些信息不能由脚本替代；本包不声称已经复现论文表格。
