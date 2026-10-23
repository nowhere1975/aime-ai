# AIME AI 选字：研究总报告（2026-10-13 至 10-17）

## 结论

给 AIME 加了一个本地运行的上下文选字功能。真实使用中，**需要纠正的词从 8.3% 降到 4.3%，约减少一半**。
现在线上用的是自己从头训练的字级小模型（52MB，服务内存约 145MB），回放效果与微调后的 Qwen3-0.6B 打平，内存只要后者约八分之一。

全程不联网：打字内容、个人语料、模型都只在本机。

## 1. 问题

用户的打法：从不整句输入，两三个字一个词地打，常用简拼（如 `sm`）。痛点排序：**选错候选 > 翻页 > 上屏后才发现错**。
目标是减少打字**过程**里的纠正动作，而不是提高最终文字的正确率。

## 2. 方案

"只排序、不生成"：RIME 照常给出候选，本地语言模型按 `log P(候选 | 光标前文字)` 打分，与 RIME 排名、个人搭配融合：

```
score = log P(cand | ctx) − λ·log(1 + RIME 排名) + μ·[（前一个词, cand）以前打过]
```

只在同长度的纯汉字候选里重排。λ 让模型只有在相当有把握时才挪动；单字词、多字词各一个 λ。

### 接入方式（`../Apps/AIME/`）

- `Ranker.swift`：AIME 通过 Unix socket（`~/Library/AIME/ranker/ranker.sock`，仅本人可读写）问本地服务该高亮哪个，**等不到 50ms 就用 RIME 原排序**。
- 只移动高亮，不改顺序：数字键对应不变，空格选高亮项。每个输入串只调整一次，只在第 1 页、高亮仍在首位时调整。开启 Secure Input（输密码）时不发送。
- 上屏时把结果发给服务，更新个人搭配，并预先算好下一次的前文。
- 开关：`~/Library/AIME/ranker/ENABLED`。服务挂掉时 AIME 照常打字。
- `TypingLog.swift`：可选的打字过程日志（`~/Library/AIME/typing-log/ENABLED`），记录按键、候选、上屏，是全部评测的数据来源。两项功能都写在 `../docs/privacy.md`。

### 服务（`ranker/`）

- `server.py`：Python + MLX，`--tiny` 用 tinylm，否则用 Qwen（mlx_lm）。
- `run.sh`：有 `data/tinylm/live` 就用 tinylm，否则用微调后的 Qwen。
- `install-agent.sh`：装成登录自启的 LaunchAgent（不需要 sudo，退出后自动重启，`ProcessType=Interactive` 避免被降速）。卸载：`install-agent.sh uninstall`。

## 3. 时间线与结果

### 10-13 离线评测（`results/2026-10-13-offline-eval.md`）

- TypeSafe Jev 只有云端 API（70–500ms），不能用；但它"只打分、不生成"的思路可以用本地小模型复现。
- 用 librime 驱动（`eval/rime_cands.c`）在语料上逐词取真实候选：公开语料上首选错误减少约一半，用户自己的文字上减少 15–35%。

### 10-14 真实打字日志（`results/2026-10-14-typing-log.md`）

- 基线：约 10% 的词需要纠正，单字错误率 11.5%，翻页几乎为 0。RIME 首选 94% 等于"上次对这个拼音的选择"。
- **否定**：按长期计数稳定排序（纠正反而 +11–12%）；改成一次打多个词（无论白霜还是完整万象方案都比逐词差）。
- 只看前文的通用模型收益小（纠正 −9%）。

### 10-15 评测口径、换方案、微调（同上文件的补充）

- 口径修正：上屏后删掉、换了拼音重打的（打错拼音或改主意）不算排序错误。
- **否定换万象**：按时间回放并让引擎同时学习，空词库下万象 90.3%、白霜 90.0%，打平；白霜目前的优势来自积累的个人词库。
- 用用户自己的 26 万字（打字日志开始前写的）对 Qwen3-0.6B 做 LoRA 微调：回放纠正减少从 14% 提高到 20%。
- 服务延迟优化（合并模型、上屏时预计算前文、固定批次形状）。

### 10-15 至 10-16 上线实测（`results/2026-10-16-live-ranker.md`）

| | 开启前（2405 词） | 开启后（2702 词） |
|---|---|---|
| 需要纠正 | 8.3% | **4.3%** |
| 单字需要纠正 | 12.4% | 5.5% |
| 每百字纠正次数 | 5.0 | 2.6 |

同期 RIME 原排序的错误率仍是 8.2%，下降来自模型。调整 165 次：帮上忙 131、帮倒忙 26（主要是 在/再）。
用户反馈：略有延迟但不影响使用，整体体验明显变好。

### 10-16 至 10-17 内存优化与 tinylm（`results/2026-10-17-tinylm.md`）

- Qwen 服务内存 4.5GB（MLX GPU 缓存无上限）→ 设上限后 1.2GB。转 4bit 降到约 760MB，但纠正减少从 36% 掉到 27%，放弃。
- 从头训练字级小模型：13.7M 参数，11 亿字公开语料（知乎、维基、新闻）预训练约 2 小时，再用个人文字微调 500 步。
- 回放：需纠正 4.9%（Qwen 5.2%），打平；服务内存约 145MB，延迟 p50 12ms，超过 40ms 只有 0.3%。10-17 起线上使用。

## 4. 学到的东西

- **个人语料最值钱。** 通用模型只看前文收益有限；用用户自己的文字微调，无论 Qwen 还是小模型都大幅提升（tinylm：6.5% → 4.9%）。
- **融合 RIME 排名是必须的。** 单独用模型排序反而比 RIME 差；RIME 已学到"最近用过"，模型只该在有把握时推翻它。
- **小而专的模型够用。** 判断"哪个候选接在前文后面更顺"不需要大模型。一字一词元省掉了大词表，13.7M 参数的模型就能和 0.6B 打平。
- **个人文字少，微调要短。** 26 万字，超过约 500 步就过拟合。
- **剩下的错误多是 在/再、的/得 这类要看后文的字。** 用户逐词输入，打这个字时后文还没出现。用户表示不在意（不影响理解），暂不处理。
- **离线回放可信。** 回放预测和真实使用方向一致；真实效果更好（个人搭配会随打字实时更新）。

## 5. 复现

```bash
# 环境
python3.12 -m venv research/.venv && research/.venv/bin/pip install -r research/requirements.txt pyarrow

# 打字日志 → 回放用例（research/data/replay_cases.json）
python3 research/eval/analyze_log.py

# tinylm：下载语料到 research/data/tinylm/raw/（Zhihu-KOL、wikipedia-cn-20230720-filtered、THUCNews，见 prepare.py）
research/.venv/bin/python research/tinylm/prepare.py
research/.venv/bin/python research/tinylm/train.py --out research/data/tinylm/base --steps 40000
research/.venv/bin/python research/tinylm/train.py --out research/data/tinylm/ft500 --init research/data/tinylm/base \
    --steps 500 --warmup 100 --lr 3e-4 --mix research/data/tinylm/personal_train.bin --mix-frac 0.3
research/.venv/bin/python research/eval/replay_log.py --tiny research/data/tinylm/ft500

# 上线
ln -sfn ft500 research/data/tinylm/live
bash research/ranker/install-agent.sh
touch ~/Library/AIME/ranker/ENABLED
```

## 6. 可能的后续

- 用一两天 tinylm 后，与 Qwen 线上的 4.3% 比较真实纠正率。
- 提升 tinylm：预训练更久、模型稍大、把打字日志里已上屏的文字加入微调、用 Qwen 作老师蒸馏。
- 把模型移进 AIME 本身（Swift），省掉约 145MB 中的 Python 部分，也不再需要单独的服务。
- 回改：看到下一个词后回头改前一个字（解决 在/再），需要先验证是否打扰用户。
