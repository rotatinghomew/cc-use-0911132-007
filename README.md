# 租户组合的相关收入风险与替换情景

项目使用 Python 3.11 标准库与 SQLite，现有逐案处理入口为 `market_space_twin.simulator.cli`。保留原有处理器、持久化与测试；分析业务使用本地合成资料，不连接生产系统。无需 pip 安装、下载模型或启动额外服务。

## 现有程序

在仓库根目录运行 `bash scripts/demo.sh`，依次完成现有产品回归、分析输入核对、语法构建和真实业务入口调用。所有运行文件位于临时目录，结束后清理；运行失败立即退出，不使用预存输出代替执行。

单独测试：`python3 -m unittest discover -s tests/space_scenarios -p 'test_*.py' -v`。

输入检查：`python3 -m unittest discover -s tests -p test_analysis_inputs.py -v`。

组合风险反例：`python3 -m unittest discover -s tests/analysis -p 'test_*.py' -v`。

构建检查：`python3 -m compileall -q market_space_twin tests`。

业务调用：`python3 -m market_space_twin.simulator.cli fixtures/demo.json --database /tmp/market_space_twin-demo.sqlite3`。

## 分析资料与联调

`fixtures/analysis/input.json` 保存业务演练数据，字段解释在 `fixtures/analysis/fields.md`。以下已核对的数据形态用于分析回归：new_tenant、unequal_history、matched_scenario_rents。现有输入测试检查这些形态是否真实存在，不代替分析方法或数值正确性的验证。

组合风险入口：

```bash
python3 -m market_space_twin.portfolio_risk \
  --input fixtures/analysis/input.json \
  --database /tmp/market_space_twin-analysis.sqlite3 \
  --output /tmp/portfolio-risk.json
```

同输入、同 twin 基线重放返回同一 `risk_run_id`。`scripts/feature_demo.sh` 先重跑现有产品与基线 twin 运行，再从干净数据库计算分析、同输入重放、核对输出结构并打印完整 JSON。分析 ID 为 `risk_run_id`，顶层含 `headline`（等租金假设下逐方案一行比较），其余输出包括 `exposures, dependence, scenarios, contributions, limitations`，并引用现有租约与 twin 版本（`twin_reference`）。

### 方法口径

- **合同口径 vs 可比表现**：可收租金 = 申报应收 − 已明确减免；减免不是欠款。若申报应收与租约按日折算的合同租金不一致，记入 `contract_basis_adjustments` 并采用申报应收作为当月合同口径，保证比率可比。
- **缺失与晚到**：`available_at` 晚于 `as_of` 的事实不采用；缺失月份记为缺失，不填零。无历史候选租户用同行业代理与区间，不虚构序列。
- **依赖结构**：同行业用行业共同因子估计载荷；跨行业相关在重叠不足时不识别，向结构性先验收缩并给出识别带，始终不假设相互独立。矩阵与模拟使用同一套数值。
- **尾部风险**：月度两状态压力模型（概率 × 深度）匹配观测均值与方差，保证收款率在 [0,1]；月度独立、租户间按因子相关。`VaR95/ES95` 是模型估计，贡献之和经核对等于组合 ES。
- **情景分层**：`historical_term` 来自历史估计；`management_shocks` 与 `drills` 是管理层假设/结构演练，标记 `not_a_probability=true`，不混入估计。`annual_normalisation.matched` 把各方案缩放到同一租金假设，只让业态组合与依赖改变。
- **空置**：空置是独立区间，不计入任何租户暴露或贡献；替换方案在 `starts_at` 截断旧租约，旧新暴露不重叠。

业务反例测试：`python3 -m unittest discover -s tests/analysis -p 'test_*.py' -v`。

## 离线环境

使用 Python 3.11.2 的 Linux 环境即可复现。容器环境预置解释器后，以只读方式挂载仓库，将文件复制到容器临时目录再运行脚本；运行时关闭网络。基础测试和小规模本地数据适合在 90 秒内展示，正式展示命令为 `bash scripts/feature_demo.sh`。不要在展示脚本里安装依赖或等待外部服务。
