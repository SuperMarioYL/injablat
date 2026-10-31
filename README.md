[English](./README.en.md) | **简体中文**

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Noto+Sans+SC&weight=600&size=24&pause=1200&color=2F81F7&center=true&vCenter=true&width=920&lines=%E5%A4%8D%E7%8E%B0%E4%B8%8D%E7%AD%89%E4%BA%8E%E5%BD%92%E5%9B%A0;%E5%88%A0%E6%8E%89%20prefill%EF%BC%8C%E7%BB%93%E8%AE%BA%E8%BF%98%E6%88%90%E7%AB%8B%E5%90%97%EF%BC%9F;bit-exact%20manifest%20%C2%B7%20%E6%9C%80%E5%B0%8F%E5%85%85%E5%88%86%E6%B3%A8%E5%85%A5%20%C2%B7%20%E5%8F%AF%E5%A4%8D%E6%A0%B8%E8%A3%81%E5%AE%9A%E5%8D%A1" alt="复现不等于归因 · 删掉 prefill，结论还成立吗？">
</p>

<h1 align="center">Injablat</h1>

<p align="center"><b>删掉 harness 注入的每一段上下文，重跑，看这个模型行为结论还站不站得住。</b></p>

<p align="center">
  <a href="https://github.com/SuperMarioYL/injablat/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/injablat/ci.yml?branch=main&label=ci" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT">
  <img src="https://img.shields.io/badge/version-0.1.0-6f42c1" alt="v0.1.0">
  <img src="https://img.shields.io/badge/backends-vLLM%20%7C%20llama.cpp-0b7285" alt="vLLM and llama.cpp">
</p>

## 它解决什么问题

一条公开声明说「模型会补全半截 cURL」。你在自己的端点上复现了：在同一个 system prompt、prefill 和 tool schema 下，它确实会。但复现只证明这个行为在某种上下文里出现过，没有回答是谁让它出现的。

Injablat 把这个问题变成一次受控删除：固定 temperature 0、seed 0、batch 1，记下模型 revision、chat template、tokenizer 与请求字节，然后逐个删掉注入的 span 再跑。输出是一张裁定卡，上面有最小充分注入，以及那个真正想问的问题的答案——把这段删掉，结论还成立吗。

它适合三类人：

- **模型安全与合规团队**：把一条争议结论写进报告或采购评审之前，先分清模型行为和 harness 注入。
- **agent 平台与 harness 作者**：system prompt、prefill、tool schema、few-shot 层层叠加，需要知道哪一层真正负载。
- **模型厂商安全组**：回应「这是不是你们模型自己的能力」时，给出一张对方可以自己重跑的表。

## 安装

需要 Python 3.12 或更高版本。v0.1 尚未发布到 PyPI，请从源码安装；CLI 只发 HTTP 请求，不安装 CUDA，不下载权重。

```bash
git clone https://github.com/SuperMarioYL/injablat.git
cd injablat
uv venv --python 3.12
uv pip install -e .
source .venv/bin/activate
injablat --help
```

没有 uv 时用标准库的 venv：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 快速开始

先跑离线路径确认机制。它使用随包附带的确定性 fixture，不连接任何端点，也不需要 GPU：

```bash
injablat validate examples/qwen-prefill/claim.yaml --fixture
injablat replay examples/qwen-prefill/claim.yaml --runs 2 --fixture
injablat ablate examples/qwen-prefill/claim.yaml --runs 2 --fixture
injablat report examples/qwen-prefill/claim.yaml
injablat verify reports/qwen-prefill-curl --fixture
```

fixture 的答案由本地模拟端点生成，只用于演示消融机制，不代表任何已发布权重的行为。卡片会把 endpoint build 标成 `+fixture`，不会把它伪装成一次真实测量。

要测量真实端点，先起一个 OpenAI 兼容服务：

```bash
vllm serve Qwen/Qwen3-8B-Instruct --port 8000 --seed 0 --max-model-len 8192
```

然后把 `examples/qwen-prefill/claim.yaml` 里的 `model.endpoint` 和 `model.revision` 换成本次权重的值，再二选一固定模板：

- 用目标权重的模板字节覆盖 `templates/chat_template.jinja`，并把 `chat_template_sha` 设成该文件的 sha256；
- 或者删掉 `model.chat_template_path`，让端点提供模板（vLLM 需开启 `--enable-tokenizer-info-endpoint`），并把 `chat_template_sha` 设成端点返回模板的 sha256。

`validate` 会重新计算模板文件或端点模板的哈希；它与 `chat_template_sha` 不一致就退出 2，所以只改哈希、留着旧模板会被拒绝。改完后按上面的顺序去掉 `--fixture` 逐条重跑。如果 claim 没有带模板、端点也不提供，`validate` 会报错退出，而不是猜一个模板。

llama.cpp 走同样的 `/v1/chat/completions` 接口，把 `backend` 改成 `llamacpp` 并指向 `llama-server` 即可。它的 prefill 消融不在 v0.1 支持范围内，细节见「局限与已知边界」。

## 演示

实录约 70 秒，从 `validate` 到 `verify` 全部是离线 fixture 的真实输出：

[`docs/demo.cast`](./docs/demo.cast) · 本地播放：`asciinema play docs/demo.cast`

```text
claim qwen-prefill-curl
...
manifest sha256:8c5885884957544c421ea15c3a1c227b585c3f6af29d391851517d5e89b46127
runs per configuration 2
determinism DETERMINISTIC (repeat runs are byte-identical for every configuration)

  remove span                   behavior    output sha    role
  (bare prompt)                 NOT_FIRED   7d554cf3...   baseline
  (nothing removed)             FIRED       c73ca629...   reproduces
  prefill[0:2232]               NOT_FIRED   7d554cf3...   necessary
  system                        FIRED       d670c441...   redundant
  tool_schema                   NOT_FIRED   7d554cf3...   necessary
  few_shot[0:2]                 NOT_FIRED   7d554cf3...   necessary
  prefill[0:558]                NOT_FIRED   7d554cf3...   insufficient
  prefill[0:1116]               NOT_FIRED   7d554cf3...   insufficient
  prefill[0:1674]               NOT_FIRED   7d554cf3...   insufficient
  (minimal set)                 FIRED       d670c441...   verified

minimal sufficient injection  prefill[0:2232] + tool_schema + few_shot[0:2]
survives without prefill      NO
```

这张卡说的是一件很具体的事：完整注入下行为出现；删掉 2,232 token 的 prefill 后行为消失；删掉 system prompt 后行为仍在；tool schema 和 few-shot 也是必要的；prefill 截到 558、1,116、1,674 token 都还不够。最后的最小集合被重新跑了一遍，仍然成立。

同一次运行的命令、退出码和关键输出记录在 [`docs/demo-results.json`](./docs/demo-results.json)。内置示例的文本是合成的，形状取自那场公开的 prefill 争议；token 数来自随包 fixture 的 word-piece tokenizer，不是 Qwen tokenizer。

## 工作原理

### Span、Claim 与 Manifest

一次测量由三个物件定义：

```text
Span    {id, role, text, tokens, truncations?}
        role: system | prefill | tool_schema | few_shot | user
Claim   {id, model, spans, detector, sampling}
Report  {manifest, rows, minimal_sufficient, survives_without, determinism}
```

`Claim` 把一条声明连同它当时被注入的上下文固定下来：模型名、endpoint、后端类型、revision、chat template 的哈希（以及可选的模板文件）、每个 span 的文本与 token 数、声明式检测器，以及采样参数。只有注入的 span 参与消融，用户任务始终保留。

`Manifest` 是可复核性的核心，而不是一张装饰性的元数据表。它记录 model revision、endpoint build 字符串（vLLM 的 `GET /version` 或 llama.cpp 的 `GET /props` 的 `build_info`）、chat template 哈希、tokenizer 名称、tokenizer probe 的哈希、prompt token ids 的哈希、采样参数、序列化后 messages 的请求哈希，以及 output 的哈希。prompt token ids 只取自后端 `/tokenize` 的返回，不从 chat 响应里推断。

每次请求都落盘成 `reports/<claim>/runs/<config>.run<N>.json`，里面是请求字节、响应字节、耗时与 output sha256。`report` 从记录下来的 `ablation.json` 渲染卡片，该文件的行里带着请求体和 output 哈希；`verify` 读取卡片 JSON（`report.json`，或 `report.md` 中内嵌的同一份 JSON）。两者都回溯到已记录的 HTTP 交换，不重新推导证据。

### 重跑与确定性

每个配置至少跑两遍。两遍的 `output_sha256` 不一致就报 `NONDETERMINISTIC`，以非 0 退出，并且不写 manifest：没有稳定的指纹，就没有可复核的卡。宁可说判不了，也不把噪声当结论。

两个后端的推理都走 `/v1/chat/completions`，但 tokenize、chat template 和 prefill 三件事并不通用：

| 能力 | vLLM | llama.cpp |
| --- | --- | --- |
| tokenize | `POST /tokenize`，接受 prompt 或 messages | `POST /tokenize`，接受 content 与 pieces 选项 |
| chat template | `GET /tokenizer_info`（需服务端开启）或 claim 自带 | `GET /props` 或 claim 自带 |
| prefill / assistant 前缀 | `continue_final_message` 原生支持 | v0.1 不做 prefill 消融 |
| build 标识 | `GET /version` | `GET /props` 的 `build_info` |

两边 `/tokenize` 的请求体不通用，所以后端适配层按 `backend` 显式分支，而不是做一个「一份请求打通两边」的统一抽象。

### 声明式检测器

判定路径上没有模型。检测器只有三种：

| kind | 匹配方式 |
| --- | --- |
| `contains` | 输出包含给定子串 |
| `regex` | 输出匹配给定正则 |
| `json_field` | 输出按 JSON 解析后读取指定字段 |

它回答的是「声明的行为有没有出现」，不是「这条声明是否成立」。把自然语言声明算子化成一个可匹配谓词，是使用者的工作；工具不会用另一个 LLM 来当裁判。

### 三个裁定词

| 词 | 含义 |
| --- | --- |
| `NECESSARY` | 删掉该 span（prefill 截断到 0），行为不再出现 |
| `REDUNDANT` | 删掉该 span，行为仍然出现 |
| `UNDECIDED` | 该配置两次重跑不一致，不给结论 |

卡片上的每一行还带一个角色，用来说明这一行为什么被跑：

| role | 含义 |
| --- | --- |
| `baseline` | 去掉全部注入的用户任务 |
| `reproduces` | 完整注入，行为在这里出现 |
| `insufficient` | 声明的 prefill 前缀还没恢复行为 |
| `sufficient` | 声明的 prefill 前缀恢复了行为 |
| `verified` | 最终集合再跑一遍，确认它仍然成立 |
| `undecided` | 重跑不一致，这一行没有结论 |

### 最小充分注入

搜索是贪心的：每次删掉一个 span；如果行为仍然出现，就保留这次删除（`REDUNDANT`），如果行为消失，就把这个 span 放回（`NECESSARY`）；最后对整个最小集再验一遍。请求数是 O(span 数)，不是 O(2^n)。

最小充分集不唯一。贪心搜索给出的是一个解，不是全部解。报告保留每一行搜索路径和删除顺序，对方可以换一个顺序重跑；只要两组最小集的交集还在，交集中的 span 就仍然是负载的。

## 命令参考

| 命令 | 作用 | 常用参数 |
| --- | --- | --- |
| `injablat validate <claim>` | 校验 schema、解析模板、渲染 messages、按 `/tokenize` 统计每个 span | `--config` |
| `injablat replay <claim>` | 重跑并写 manifest，执行确定性门 | `--runs`、`--out` |
| `injablat ablate <claim>` | 删除扫描、prefill 截断梯子、贪心最小充分搜索 | `--runs`、`--out` |
| `injablat report <claim>` | 从已记录的消融结果渲染 `report.md`、`report.html`、`report.json` | `--out` |
| `injablat verify <report>` | 按卡里的记录重新发请求，逐行打印 `MATCH`、`DRIFT` 或 `UNDECIDED` | `--endpoint`、`--runs` |

`verify` 接受报告目录、`report.json` 或 `report.md`。`report.md` 内嵌了与 `report.json` 相同的卡片数据，所以对方只需要收到一个文件。

退出码：`0` 产生了裁定；`1` 判不了（不一致、无裁定、DRIFT，或没有任何可比对的行）；`2` claim、模板或端点错误。

`--fixture` 是离线演练用的本地模拟端点，不出现在 `--help` 里。它不会调用真实端点，卡片会把 endpoint build 标成 `+fixture`。

## 配置

运行参数从 `configs/injablat.yaml` 读取，也可以用 `--config` 指定另一个文件。命令行选项覆盖文件值；claim 本身不携带超时和运行次数。

```yaml
backend:
  request_timeout_s: 300
  tokenize_timeout_s: 30

determinism:
  runs: 2
```

未知配置键会在加载时被拒绝，避免写了一个没人读的开关还以为它生效了。

## 与评测/trace 工具的区别

| | 评测框架 | trace 回放 | Injablat |
| --- | --- | --- | --- |
| 起点 | 固定任务或数据集 | 一次已记录的执行 | 一条已发布声明及其注入上下文 |
| 单位 | 任务上的聚合分数 | 一份转录 | 每条 claim 的 span 删除搜索 |
| 回答 | 模型在这组任务上表现如何 | 这次执行能否重现 | 哪段注入是必要的，删掉结论是否成立 |
| 输出 | 可比的聚合指标 | 转录与差异 | 裁定卡、manifest、最小充分注入 |

评测框架（例如 lm-evaluation-harness、Inspect AI）把任务、评分和复现性做得很好，但它们的单位是数据集上的任务，输出是用于横向比较的聚合值。按 claim 做 span 搜索没有排行榜位置，这是抽象层不匹配，不是它们缺一个功能。

trace 发布比「相信我们」前进了一步：它是真实的输入序列，回放成本也低。但回放只证明行为能重现，不证明是谁造成的；删掉 prefill 之后那段转录，厂商没有发布过，也没有动力发布。Injablat 不替代这些工具，它处理的是它们都不覆盖的那一步：归因。

## 局限与已知边界

- **内置示例是合成的。** 文本为示例编写，token 数来自随包 fixture 的 tokenizer；它演示机制，不代表任何已发布权重。
- **离线 fixture 不是测量。** `--fixture` 只用于快速验证流程，卡片会标 `+fixture`。
- **prefill 消融只支持 vLLM。** 走 `continue_final_message`；llama.cpp 在 v0.1 只用于非 prefill 的删除扫描。
- **只认一种 span JSONL 格式。** 任意 agent 框架的 trace 需要手工转换，没有自动导入。
- **只做删除与前缀截断。** 不改写、不做同义替换、不做 token 扰动；删掉后行为消失只说明这段是必要的，不证明它单独就足以诱发行为。
- **检测器是声明式的。** 它不理解语义，只执行 `contains`、`regex` 或 `json_field`；claim 必须先被算子化成可匹配谓词。
- **最小充分集不唯一。** 贪心顺序会影响结果；报告保留搜索路径，请用另一个顺序复核。
- **没有统计显著性。** 默认前提是 temperature 0、seed 0；不做多次采样的置信区间。
- **无法固定 revision 的闭源端点不在支持范围。** 没有可复核的 manifest，就不给裁定。
- **只有单进程 CLI 和本地报告目录。** 没有 Web 面板、服务端、多用户、登录或队列。
- **没有 Windows 原生打包。** 在 Linux 和 macOS 上使用 Python 3.12 运行。
- **端点漂移会触发 DRIFT。** 换权重、换驱动、batching 或静默升级都可能改变输出哈希，这正是 `verify` 要暴露的。
- **token 计数可能不一致。** 端点的 tokenizer 与记录不一致时，`validate` 会打印差异，而不是悄悄改掉记录。

## 路线图

v0.1 已经实现：claim schema、模板固定、vLLM 与 llama.cpp 的非 prefill 重跑、manifest 与确定性门、删除扫描、vLLM 的 prefill 截断梯子、贪心最小充分搜索、`report.md` / `report.html`、`verify` 的 MATCH / DRIFT，以及离线 fixture。

后续方向（尚未实现，不承诺日期）：

- llama.cpp 的 prefill 消融，走本地渲染后的原生 `/v1/completions`。
- 从常见 agent 框架的 trace 一次性转换成 span JSONL 的导入器。
- 多 claim、多模型的批量矩阵。
- 多次采样与统计显著性检验。
- 裁定卡的签名、归档与换权重回归（团队版）。

## 团队版与授权

CLI 和消融引擎保持 MIT 开源，免费使用。付费的是团队版，卖的是可交付性，不是把命令行锁起来：

- **内网与私有化部署**：适配只在内网可达的推理端点，数据不出网络。
- **裁定卡归档与签名**：把报告、manifest 和运行记录做成可审计的证据链。
- **换权重回归**：同一条 claim 在每次发新权重后自动重跑，输出 MATCH / DRIFT 趋势。
- **采购与尽调模板**：给合规评审、备案和采购流程使用的可复核报告格式。
- **单份裁定报告**：针对一条争议 claim 交付卡片、manifest，并在对方端点完成一次 `verify`，是团队版之前的最小成交单元。

团队版按年授权，不做自助结账；咨询与采购走合同、发票与对公付款。具体席位数、交付范围和报价请通过 GitHub Issues（标题以 `[team]` 开头）联系维护者。

## LICENSE

MIT。见 [LICENSE](./LICENSE)。

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
