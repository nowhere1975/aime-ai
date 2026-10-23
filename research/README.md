# AIME AI 选字研究

目标：让 AIME（基于 RIME）选字时参考**已经输入的上下文**和**用户自己的习惯**，减少三件事（按痛苦程度）：

1. 选错候选（首选不对，要按数字键 / 方向键挑）
2. 翻页
3. 上屏后发现错了，删掉重打

做法是"只排序、不生成"：RIME 照常给出候选，本地小语言模型按 `log P(候选 | 光标前文字)` 打分，
与 RIME 自己的排名融合后重排。完全离线，模型跑在 Apple Silicon（MLX）上。

**完整经过和结论见 [`REPORT.md`](REPORT.md)。**

## 目录

| 路径 | 内容 |
|---|---|
| `bench/` | 速度基准：每次按键给一页候选打分要多久（`bench_fast.py`），小样例准确率（`bench_rerank.py`） |
| `eval/` | 离线评测：librime 驱动 `rime_cands.c`、评测主程序 `run_eval.py`、语料准备、打字日志分析 `analyze_log.py` |
| `ranker/` | 线上打分服务（`server.py`）、启动脚本 `run.sh`、登录自启 `install-agent.sh`、部署到另一台 Mac `deploy-remote.sh`、延迟测试 `bench_client.py` |
| `tinylm/` | 自训字级小模型：语料处理 `prepare.py`、训练 `train.py`、模型与打分 `model.py` |
| `clt-build/` | 不装 Xcode 用命令行工具构建 AIME（`build.sh`） |
| `results/` | 实验记录（只含汇总数字，不含个人文字） |
| `data/` | **不提交**：语料、个人文字、打字日志派生数据、评测记录、RIME 用户数据副本 |

`../Apps/AIME/TypingLog.swift` 是本分支给 AIME 加的打字过程日志（可选，默认关闭，见 `../docs/privacy.md`）。

## 快速开始

```bash
python3.12 -m venv research/.venv && research/.venv/bin/pip install -r research/requirements.txt

# 1. 构建（无需 Xcode），产物 research/clt-build/dist/AIME.app；安装命令见脚本输出
bash research/clt-build/build.sh

# 2. 离线评测：编译 librime 驱动、复制 RIME 用户数据、下载公开语料
bash research/eval/setup.sh
research/.venv/bin/python research/eval/fetch_corpus.py
research/.venv/bin/python research/eval/run_eval.py --corpus corpus.json --ctx 16 --topn 9 --max-words 4000

# 个人文字（Claude Code / Codex 提示词历史；或聊天导出 Markdown）
research/.venv/bin/python research/eval/extract_personal.py
research/.venv/bin/python research/eval/parse_chat_export.py ~/Downloads/导出.md personal_chat.json
research/.venv/bin/python research/eval/run_eval.py --corpus personal.json --ctx 16 --topn 9

# 3. 打字过程日志（装了本分支构建后）
mkdir -p ~/Library/AIME/typing-log && touch ~/Library/AIME/typing-log/ENABLED
python3 research/eval/analyze_log.py      # 首选命中率、选非首选 / 翻页 / 上屏后退格 次/百字
```

## 评测方法

逐词、真实上下文：语料用 jieba 分词，对每个纯汉字词用 pypinyin 生成全拼，交给**已安装 AIME 的 librime 和用户的真实方案**
（含 Lua / octagram 插件）取前 20 个候选；同长度候选由模型重排。融合分数 `lm − λ·log(1+rank)`，
λ 在偶数段落上调（单字词、多字词各一个），只在奇数段落上报告。

局限：语料是最终文字，看不到真实打字过程（选候选、删了重来）；全拼，没有简拼。所以有了 `TypingLog` + `analyze_log.py`，
用真实过程数据做基线和回放评测。

## 现状

见 `REPORT.md` 与 `results/`。一句话：已接入 AIME 并在真实使用中把需要纠正的词从 8.3% 降到 4.3%；
线上模型是自训的 13.7M 参数字级小模型（服务内存约 145MB），效果与微调后的 Qwen3-0.6B 打平。

相关项目：[rime-llm-ime](https://github.com/zhanghaozhecn/rime-llm-ime)（Windows，同路线，最完整）、
[azooKey Zenzai](https://zenn.dev/azookey/articles/ea15bacf81521e)（日文，神经网络直接转换）。
