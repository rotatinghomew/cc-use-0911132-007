# 业务演练数据口径

cashflows 为实际月度应收实收，concession 是已明确减免金额，不能当欠款。scenarios.shocks 是管理假设，非概率；replacement.starts_at 前旧租约有效，之后只计算新暴露。candidate_tenants 无历史时只能借助透明代理或给区间，不存在隐藏的真实现金流。

各 ID 均为本地合成标识，不包含真实人员或生产数据。时刻使用带时区 ISO 8601；as_of 是分析可知信息的截止时刻，不是业务发生时间上界。晚到事实允许保留在输入中，但只能在 available_at 到达之后采用。source_versions 标识输入版本；baseline_case_id 关联同库既有演示运行。空值表示未知，不应未经说明转成零。
