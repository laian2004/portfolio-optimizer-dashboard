"""数据获取与预处理模块"""
import pandas as pd
import yfinance as yf


def fetch_stock_data(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """获取历史收盘价数据"""
    if not tickers:
        raise ValueError("股票列表不能为空")

    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("未获取到有效行情数据，请检查股票代码或网络")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("下载数据缺少 Close 列")
        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("下载数据缺少 Close 列")
        prices = data["Close"].to_frame(name=tickers[0])

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("可用收盘价为空，请调整周期或股票代码")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """计算日收益率"""
    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("收益率为空，无法继续优化")
    return returns


def compute_statistics(returns: pd.DataFrame, trading_days: int = 252):
    """计算年化期望收益率和协方差矩阵"""
    if returns.empty:
        raise ValueError("收益率数据为空")
    mu = returns.mean() * trading_days
    cov = returns.cov() * trading_days
    return mu, cov
