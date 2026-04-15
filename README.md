# Portfolio Optimizer Dashboard

基于 Python + Dash 构建的投资组合优化仪表盘，支持有效前沿分析、资产配置可视化、相关性矩阵和多种经典组合优化策略。

## 技术栈

- Python
- Dash
- Plotly
- Pandas / NumPy

## 功能特性

- 最大夏普比率、最小方差、风险平价三类优化策略
- 有效前沿可视化
- 资产权重分布图
- 收益与风险指标展示
- 相关性矩阵分析
- 健康检查接口 `GET /healthz`

## 项目结构

```text
portfolio-optimizer/
├── app.py           # Dash 应用入口
├── data.py          # 数据获取与统计处理
├── optimizer.py     # 组合优化逻辑
├── assets/          # 样式资源
└── tests/           # 单元测试
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 测试

```bash
pytest tests -v
```
