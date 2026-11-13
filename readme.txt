测试：python3 -m unittest discover -s tests/space_scenarios -p 'test_*.py'
构建：python3 -m compileall -q market_space_twin tests/space_scenarios
验收：python3 -m market_space_twin.simulator.cli fixtures/demo.json --database /tmp/market_space_twin-acceptance.sqlite3
所有输入均为本地合成数据，运行不需要网络。
