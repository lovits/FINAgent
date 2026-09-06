from types import SimpleNamespace

import pandas as pd

import tradingagents.dataflows.akshare_news as china_news
import tradingagents.dataflows.baostock as china_stock
import tradingagents.dataflows.stockstats_utils as stockstats
import tradingagents.graph.trading_graph as graph_module
from tradingagents.dataflows.interface import VENDOR_METHODS
from tradingagents.dataflows.symbol_utils import is_a_share_symbol, normalize_symbol
from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_plain_a_share_codes_gain_exchange_suffix() -> None:
    assert normalize_symbol("600519") == "600519.SS"
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("600519.SH") == "600519.SS"
    assert is_a_share_symbol("300750")
    assert not is_a_share_symbol("AAPL")


def test_baostock_symbol_conversion() -> None:
    assert china_stock.to_baostock_code("600519") == "sh.600519"
    assert china_stock.to_baostock_code("000001.SZ") == "sz.000001"


def test_a_share_ohlcv_bypasses_yahoo(monkeypatch) -> None:
    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-09-04"]),
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000],
        }
    )
    monkeypatch.delenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", raising=False)
    monkeypatch.setattr(china_stock, "load_ohlcv", lambda symbol, date: expected)
    monkeypatch.setattr(
        stockstats.yf,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("A-share OHLCV must not call Yahoo")
        ),
    )
    result = stockstats.load_ohlcv("600519", "2026-09-04")
    assert result["Close"].tolist() == [101.0]


def test_a_share_graph_activates_china_vendor_profile(monkeypatch) -> None:
    selected = {}
    graph = object.__new__(TradingAgentsGraph)
    graph._configured_data_vendors = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    graph._a_share_data_vendors = {
        "core_stock_apis": "baostock",
        "technical_indicators": "baostock",
        "fundamental_data": "baostock",
        "news_data": "akshare",
    }
    monkeypatch.setattr(graph_module, "set_config", lambda value: selected.update(value))
    monkeypatch.setattr(
        graph_module,
        "resolve_instrument_identity",
        lambda ticker: {"company_name": "贵州茅台"},
    )

    context = graph.resolve_instrument_context("600519.SS")

    assert selected["data_vendors"]["core_stock_apis"] == "baostock"
    assert selected["data_vendors"]["news_data"] == "akshare"
    assert "贵州茅台" in context


def test_non_a_share_graph_restores_configured_vendor_profile(monkeypatch) -> None:
    selected = {}
    graph = object.__new__(TradingAgentsGraph)
    graph._configured_data_vendors = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    graph._a_share_data_vendors = {
        "core_stock_apis": "baostock",
        "technical_indicators": "baostock",
        "fundamental_data": "baostock",
        "news_data": "akshare",
    }
    monkeypatch.setattr(graph_module, "set_config", lambda value: selected.update(value))
    monkeypatch.setattr(graph_module, "resolve_instrument_identity", lambda ticker: {})

    graph.resolve_instrument_context("AAPL")

    assert selected["data_vendors"] == graph._configured_data_vendors


def test_akshare_company_news_is_date_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        china_news.ak,
        "stock_news_em",
        lambda symbol: pd.DataFrame(
            [
                {"新闻标题": "valid", "新闻内容": "inside", "发布时间": "2026-09-03", "文章来源": "东财", "新闻链接": "https://example.com/1"},
                {"新闻标题": "future", "新闻内容": "outside", "发布时间": "2026-09-08", "文章来源": "东财", "新闻链接": "https://example.com/2"},
            ]
        ),
    )
    monkeypatch.setattr(china_news, "get_config", lambda: {"news_article_limit": 20})
    result = china_news.get_news("600519", "2026-09-01", "2026-09-04")
    assert "valid" in result
    assert "future" not in result


def test_china_vendors_cover_a_share_agent_tools() -> None:
    assert VENDOR_METHODS["get_stock_data"]["baostock"] is china_stock.get_stock
    assert VENDOR_METHODS["get_news"]["akshare"] is china_news.get_news
    assert VENDOR_METHODS["get_fundamentals"]["baostock"] is china_stock.get_fundamentals


def test_baostock_result_rows_become_dataframe() -> None:
    rows = iter([["2026-09-04", "sh.600519"]])

    class Result(SimpleNamespace):
        fields = ["date", "code"]
        error_code = "0"
        error_msg = "success"

        def next(self):
            try:
                self.current = next(rows)
                return True
            except StopIteration:
                return False

        def get_row_data(self):
            return self.current

    assert china_stock._frame(Result()).to_dict("records") == [
        {"date": "2026-09-04", "code": "sh.600519"}
    ]


def test_a_share_reflection_returns_do_not_call_yahoo(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module.yf,
        "Ticker",
        lambda symbol: (_ for _ in ()).throw(
            AssertionError("A-share reflection must not call Yahoo")
        ),
    )

    def history(symbol, start, end):
        closes = [100.0, 110.0] if symbol == "600519.SS" else [100.0, 102.0]
        return pd.DataFrame({"Close": closes})

    monkeypatch.setattr(china_stock, "_history", history)
    raw, alpha, days = TradingAgentsGraph._fetch_returns(
        None,
        "600519.SS",
        "2026-08-20",
        holding_days=1,
        benchmark="000001.SS",
    )
    assert raw == 0.1
    assert round(alpha, 2) == 0.08
    assert days == 1
