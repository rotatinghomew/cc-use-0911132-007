import json
import sys
from pathlib import Path

first = json.loads(Path(sys.argv[1]).read_text())
replay = json.loads(Path(sys.argv[2]).read_text())
required = ['risk_run_id', 'exposures', 'dependence', 'scenarios', 'contributions', 'limitations']
missing = [field for field in required if field not in first]
if missing:
    raise SystemExit("缺少分析字段：" + ", ".join(missing))
if first['risk_run_id'] == replay.get('risk_run_id') == "":
    raise SystemExit("分析记录缺少 risk_run_id")
if first['risk_run_id'] != replay.get('risk_run_id'):
    raise SystemExit("同一输入重放未引用同一分析记录")
if replay.get('replayed') is not True:
    raise SystemExit("重放未标记为 replayed")
if not first.get('twin_reference', {}).get('referenced'):
    raise SystemExit("分析未引用既有 twin 基线版本")
print("分析输出结构、重放标识与 twin 引用核验完成")
