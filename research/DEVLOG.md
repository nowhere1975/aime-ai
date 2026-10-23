# AIME AI 选字 开发日志

接手时先读本文件，再看 `REPORT.md`（完整经过）和 `results/`（各次实验数字）。最新的记录在最上面。

## 当前状态（2026-10-23）

- 已合入 `main`（PR #1，`815489e`），`ai-ranking` 分支已删。下次开工从 `main` 新建分支。
- 线上模型：tinylm 微调 500 步版本（`data/tinylm/live -> ft500`，13.7M 参数），λ=(3,4)，μ=2。
- 一周实测：需要纠正 **4.0%**（无模型 8.3%，Qwen 微调 4.5%）。见 `results/2026-10-23-tinylm-live-mini.md`。

### 两台机器

| | MacBook（M1 Pro，本仓库所在） | Mac mini（M4 / 24GB） |
|---|---|---|
| 访问 | 本机 | `ssh nowhere@192.168.2.73`（密钥登录） |
| 用途 | 开发、训练、评测 | **10-23 起主要在这台打字** |
| AIME | 10-15 的研究构建（菜单图标仍是「艾」） | `clt-build/dist` 构建（书签图标），原官方 0.1.9 备份在 `~/aime-deploy/AIME-0.1.9-official.app` |
| 打分服务 | LaunchAgent，跑本仓库的 `research/ranker/run.sh` | LaunchAgent，跑 `~/mycode/aime-ai/research/ranker/run.sh`（独立副本，非 git） |
| 打字日志 | `~/Library/AIME/typing-log/` | 同路径，在 mini 上 |
| 服务日志 | `~/Library/AIME/ranker/server.log` | 同路径，在 mini 上 |

## 下次开工

```bash
# 1. 把 mini 的打字日志拉回来（存到 data/typing-log/192.168.2.73/，analyze_log 会一并统计）
bash research/eval/pull-logs.sh nowhere@192.168.2.73

# 2. 总体指标 + 生成回放用例 data/replay_cases.json
python3 research/eval/analyze_log.py

# 3. 两台机器的服务是否在跑
launchctl print gui/$(id -u)/app.zool.inputmethod.aime.ranker | grep -E '^\s*state'
ssh nowhere@192.168.2.73 'tail -3 ~/Library/AIME/ranker/server.log'
```

按模型分时段比较"需要纠正"：用 `eval/intents.py` 的 `load_intents()`，按 `t`（毫秒时间戳）切段，
高亮的是 `cands[rk or 0]`，`page > 0` 或高亮 ≠ `gold` 记为需要纠正。切换点：Qwen 10-15 12:37，tinylm 10-17 10:55。

## 待办（按优先级）

1. **用新打字数据重新微调 tinylm。** 现在的 ft500 只用了打字日志开始前的 26 万字；日志里已上屏的文字（10-23 时约 1.7 万字，
   以后主要来自 mini）还没用上。建议攒到 5 万字以上再做。
   - 注意：`eval/make_ft_data.py` 故意丢掉日志开始后的文字，避免回放评测泄漏。加入日志文字时要按时间切分：
     早期日志进训练，最近几天留作回放评测。
   - 26 万字超过约 500 步就过拟合；数据变多后重新扫步数（参考 `tinylm/pipeline.sh`）。
   - 新模型回放更好再上线：`ln -sfn <新目录> data/tinylm/live`，本机 `bash ranker/install-agent.sh`，
     mini 用 `bash ranker/deploy-remote.sh nowhere@192.168.2.73`（会拷 live 指向的模型并重启服务；不覆盖 mini 已有的 bigrams）。
2. MacBook 上的 AIME 更新到书签图标构建：`bash research/clt-build/build.sh` 后按其输出的命令安装（需本机管理员密码）。
3. 其他后续见 `REPORT.md` 第 6 节：更长预训练 / 稍大模型 / Qwen 蒸馏；模型移进 AIME（Swift）省掉 Python 服务；在/再 的回改。

## 坑

- `analyze_log.py` 每次运行都会覆盖 `data/replay_cases.json`。`REPORT.md` 里 10-17 的回放数字用的是 4 天版本，
  备份在 `data/replay_cases.4days.json`；要复现旧数字先换回去。
- 远端 `pip install` 曾卡在大文件下载十多分钟，`deploy-remote.sh` 已加 `--timeout 30 --retries 5`。
- 重启 AIME 进程时若窗口已在前台，日志前几条 `app` 为空（没收到 activate），切换一次应用就恢复。
- `bench_client.py` 读本机最新一天的日志；在没有日志的机器上测延迟，可用临时 `HOME` 放一份日志并把 `Library/AIME/ranker` 软链到真实目录。
- `data/` 不提交（语料、个人文字、日志、模型）。换机器训练要手动拷 `data/tinylm/`。

## 日志

### 2026-10-23
- 检查 tinylm 一周实测：4.0%，不输 Qwen（4.5%）。10-19 后 MacBook 输入量很少。
- 部署到 Mac mini：AIME 研究构建 + tinylm 服务 + 个人词对历史，打字日志开启。M4 延迟 p50 7.8ms / p99 13.8ms。
- 新增 `ranker/deploy-remote.sh`、`eval/pull-logs.sh`；PR #1 合入 main。

### 2026-10-13 至 10-17
见 `REPORT.md`：离线评测 → 打字日志 → Qwen 微调上线（8.3% → 4.3%）→ 内存优化 → 自训 tinylm 替换 Qwen。
