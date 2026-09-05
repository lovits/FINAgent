# 本地、AutoDL 和 GitHub 代码同步

代码基准为 `https://github.com/lovits/TradingAgents-RL` 的 `main` 分支。
本地项目和 AutoDL 项目使用同一个提交版本。

## 修改代码时

1. 开始工作前，在干净工作区运行 `git pull --ff-only origin main`。
2. 在任一端修改后，检查差异、运行相关测试，提交代码并推送到 `main`。
3. 另一端工作区没有未提交改动时，同样运行 `git pull --ff-only origin main`。

不要同时在两端修改同一批文件。存在未提交文件或分叉提交时，先合并并检查，不能强制覆盖。

Codex 的周期同步仅快进干净工作区到 GitHub 已提交版本；不会自动提交编辑中的文件。
离线、服务器关机或 Codex 未执行检查时，不保证实时同步。

## 不通过 Git 同步的内容

- `.env` 等密钥与机器配置。
- 基模、SFT/RL 权重、优化器状态。
- 生成的训练轨迹、行情缓存、报告及日志。
- CodeGraph 索引、虚拟环境以及独立的 `cockpit-tools` 仓库。

这些文件保留在各自机器上。代码版本一致，不表示 GPU、Python 环境、数据文件也自动一致。
