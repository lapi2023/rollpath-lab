import yfinance as yf
from datetime import datetime

def get_yfinance(ticker: str=None, start_date: str=None, end_date: str=datetime.now().strftime("%Y-%m-%d")):
    # 2008-11-06 ～ 2026-02-28（翌日を end に入れると当日まで含まれます）
    df = yf.Ticker(ticker).history(
        start=start_date, end=end_date, interval="1d", auto_adjust=False
    )
    # auto_adjust=False だと 'Adj Close' 列がそのまま出力されます
    df.to_csv(f"{ticker}_d_{start_date}-{end_date}_.csv", encoding="utf-8")

if __name__ == "__main__":
    # get_yfinance('SPXL', "2008-11-06", "2026-03-01")
    # get_yfinance('SSO', "2006-06-19", datetime.now().strftime("%Y-%m-%d"))
    # get_yfinance('^SPX', "1927-12-31", datetime.now().strftime("%Y-%m-%d"))
    get_yfinance('TQQQ', "2010-02-11")
    get_yfinance('QLD', "2006-07-21")