import yfinance as yf
from datetime import datetime

def get_yfinance(ticker: str=None, start_date: str=None, end_date: str=None):
    # 2008-11-06 ～ 2026-02-28（翌日を end に入れると当日まで含まれます）
    df = yf.Ticker(ticker).history(
        start=start_date, end=end_date, interval="1d", auto_adjust=False
    )
    # auto_adjust=False だと 'Adj Close' 列がそのまま出力されます
    df.to_csv(f"{ticker}_{start_date}_{end_date}_daily.csv", encoding="utf-8")

if __name__ == "__main__":
    # get_yfinance('SPXL', "2008-11-06", "2026-03-01")
    # get_yfinance('SSO', "2006-06-19", datetime.now().strftime("%Y-%m-%d"))
    get_yfinance('^SPX', "1927-12-31", datetime.now().strftime("%Y-%m-%d"))