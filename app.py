"""投资组合优化器 - 交互式 Dashboard"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import jsonify
from dash import Dash, html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from data import fetch_stock_data, compute_returns, compute_statistics
from optimizer import max_sharpe, min_variance, risk_parity, efficient_frontier, STRATEGIES


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "投资组合优化器"


@app.server.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "portfolio-optimizer"}), 200


def placeholder_figure(title: str, subtitle: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=subtitle,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#6b7280"},
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.P("Portfolio Lab", className="hero-eyebrow"),
                    html.H1("投资组合优化器", className="hero-title"),
                    html.P(
                        "多策略权重优化、有效前沿与风险结构可视化，一屏完成组合诊断。",
                        className="hero-subtitle",
                    ),
                ],
                width=12,
            ),
            className="hero",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("参数设置", className="mb-3"),
                                dbc.Label("股票代码（逗号分隔）"),
                                dbc.Input(
                                    id="tickers",
                                    value="AAPL,MSFT,GOOGL,AMZN,TSLA",
                                    placeholder="例: AAPL,MSFT,GOOGL",
                                ),
                                dbc.Label("历史数据周期", className="mt-3"),
                                dcc.Dropdown(
                                    id="period",
                                    value="2y",
                                    options=[
                                        {"label": label, "value": value}
                                        for label, value in [
                                            ("1年", "1y"),
                                            ("2年", "2y"),
                                            ("3年", "3y"),
                                            ("5年", "5y"),
                                        ]
                                    ],
                                ),
                                dbc.Label("优化策略", className="mt-3"),
                                dcc.Dropdown(
                                    id="strategy",
                                    value="最大夏普比率",
                                    options=[{"label": s, "value": s} for s in STRATEGIES],
                                ),
                                dbc.Label("无风险利率 (%)", className="mt-3"),
                                dbc.Input(id="rf", type="number", value=2, step=0.1),
                                dbc.Button(
                                    "运行优化",
                                    id="run-btn",
                                    color="primary",
                                    className="mt-3 w-100 run-btn",
                                ),
                                html.Div(id="status-message", className="mt-3"),
                            ]
                        ),
                        className="control-card",
                    ),
                    lg=3,
                    md=4,
                ),
                dbc.Col(
                    dcc.Loading(
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        dcc.Graph(
                                            id="frontier-chart",
                                            figure=placeholder_figure("有效前沿", "点击左侧按钮开始优化"),
                                            config={"displayModeBar": False},
                                        ),
                                        label="有效前沿",
                                    ),
                                    dbc.Tab(
                                        dcc.Graph(
                                            id="allocation-chart",
                                            figure=placeholder_figure("资产配置", "等待优化结果"),
                                            config={"displayModeBar": False},
                                        ),
                                        label="资产配置",
                                    ),
                                    dbc.Tab(html.Div(id="metrics-table"), label="组合指标"),
                                    dbc.Tab(
                                        dcc.Graph(
                                            id="corr-chart",
                                            figure=placeholder_figure("相关性矩阵", "等待数据加载"),
                                            config={"displayModeBar": False},
                                        ),
                                        label="相关性矩阵",
                                    ),
                                ]
                            )
                        ]
                    ),
                    lg=9,
                    md=8,
                ),
            ],
            className="mt-4 g-4",
        ),
    ],
    fluid=True,
    className="app-root py-4",
)


@callback(
    Output("frontier-chart", "figure"),
    Output("allocation-chart", "figure"),
    Output("metrics-table", "children"),
    Output("corr-chart", "figure"),
    Output("status-message", "children"),
    Input("run-btn", "n_clicks"),
    State("tickers", "value"),
    State("period", "value"),
    State("strategy", "value"),
    State("rf", "value"),
    prevent_initial_call=True,
)
def run_optimization(n_clicks, tickers_str, period, strategy, rf_pct):
    if not tickers_str:
        return no_update, no_update, no_update, no_update, dbc.Alert("请输入至少一个股票代码", color="warning")

    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers_str.split(",") if t.strip()]))
    if len(tickers) < 2:
        return no_update, no_update, no_update, no_update, dbc.Alert("请至少输入两只股票", color="warning")

    rf = (rf_pct or 2) / 100

    try:
        prices = fetch_stock_data(tickers, period)
        returns = compute_returns(prices)
        mu, cov = compute_statistics(returns)

        if strategy == "风险平价":
            weights = risk_parity(cov)
        elif strategy == "最小方差":
            weights = min_variance(mu, cov)
        else:
            weights = max_sharpe(mu, cov, rf)

        f_risk, f_ret, _ = efficient_frontier(mu, cov, n_points=40)
        if len(f_risk) == 0:
            raise ValueError("有效前沿计算失败，请检查输入数据")

        port_ret = float(mu.values @ weights)
        port_risk = float(np.sqrt(weights @ cov.values @ weights))
        sharpe = (port_ret - rf) / port_risk if port_risk > 0 else 0.0

        fig_frontier = go.Figure()
        fig_frontier.add_trace(
            go.Scatter(
                x=f_risk,
                y=f_ret,
                mode="lines",
                name="有效前沿",
                line={"color": "#0a6da6", "width": 3},
            )
        )
        fig_frontier.add_trace(
            go.Scatter(
                x=[port_risk],
                y=[port_ret],
                mode="markers",
                name=strategy,
                marker={"size": 14, "color": "#f97316", "symbol": "star"},
            )
        )
        for i, ticker in enumerate(tickers):
            fig_frontier.add_trace(
                go.Scatter(
                    x=[float(np.sqrt(cov.values[i, i]))],
                    y=[float(mu.values[i])],
                    mode="markers+text",
                    text=[ticker],
                    textposition="top center",
                    marker={"size": 8, "color": "#111827"},
                    showlegend=False,
                )
            )

        fig_frontier.update_layout(
            title="有效前沿",
            xaxis_title="年化波动率",
            yaxis_title="年化收益率",
            template="plotly_white",
            height=500,
            margin={"l": 20, "r": 20, "t": 56, "b": 35},
        )

        labels = [f"{t} ({w:.1%})" for t, w in zip(tickers, weights)]
        fig_alloc = go.Figure(go.Pie(labels=labels, values=weights, hole=0.45))
        fig_alloc.update_layout(
            title=f"{strategy} - 资产配置",
            height=500,
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 56, "b": 35},
        )

        metrics = {
            "指标": ["年化收益率", "年化波动率", "夏普比率", "最大权重资产", "最小权重资产"],
            "值": [
                f"{port_ret:.2%}",
                f"{port_risk:.2%}",
                f"{sharpe:.3f}",
                f"{tickers[np.argmax(weights)]} ({weights.max():.1%})",
                f"{tickers[np.argmin(weights)]} ({weights.min():.1%})",
            ],
        }
        alloc_rows = [{"资产": t, "权重": f"{w:.2%}"} for t, w in zip(tickers, weights)]

        table = dbc.Container(
            [
                dbc.Row(dbc.Col(html.H5("组合指标", className="mt-3"))),
                dbc.Table.from_dataframe(pd.DataFrame(metrics), striped=True, bordered=True, size="sm"),
                dbc.Row(dbc.Col(html.H5("权重明细", className="mt-3"))),
                dbc.Table.from_dataframe(pd.DataFrame(alloc_rows), striped=True, bordered=True, size="sm"),
            ]
        )

        corr = returns.corr()
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig_corr.update_layout(
            title="资产相关性矩阵",
            height=500,
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 56, "b": 35},
        )

        return (
            fig_frontier,
            fig_alloc,
            table,
            fig_corr,
            dbc.Alert(f"优化完成：{strategy}，样本点 {len(returns)}", color="success"),
        )
    except Exception as exc:
        error = f"优化失败: {exc}"
        return (
            placeholder_figure("有效前沿", "请修正参数后重试"),
            placeholder_figure("资产配置", "暂无结果"),
            dbc.Alert(error, color="danger"),
            placeholder_figure("相关性矩阵", "暂无结果"),
            dbc.Alert(error, color="danger"),
        )


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "false").lower() in {"1", "true", "yes"}
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=debug, host="0.0.0.0", port=port)
