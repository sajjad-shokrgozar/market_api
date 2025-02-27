# market/market.py

import os
import re
import csv
import math
import time
import requests
import datetime
import jdatetime
import numpy as np
import pandas as pd
import warnings

from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

from helpers import Helpers

warnings.filterwarnings('ignore')


class Market:
    """Market class that fetches TSE market data (stocks & options),
    calculates implied volatility, greeks, market cap, etc.
    """

    # Single class-level headers definition for all HTTP requests:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0'
    }

    # Regex pattern for filtering out symbols with digits or ending with 'ح'
    bad_symbol_pattern = re.compile(r'[0-9]|ح$')

    @staticmethod
    def maturity_validation(x):
        """True if the maturity length is at least 6 characters."""
        try:
            return len(x) >= 6
        except:
            return True

    @staticmethod
    def extract_values(row):
        """Extract pmd, qmd, pmo, qmo from the best-limit dictionary."""
        first_order = row[0]
        return (first_order['pmd'], first_order['qmd'], first_order['pmo'], first_order['qmo'])

    @staticmethod
    def init_margin(opt_type, underlying, s_close, s_last, k, c, contract_size, A=0.2, B=0.1):
        """
        Calculate initial margin for an option.
        Some options might have contract_size != 1000.
        'موج' has contract size = 100000.
        """
        if opt_type == 'call':
            loss = abs(min(s_close - k, 0)) * contract_size
        else:  # 'put'
            loss = abs(min(k - s_close, 0)) * contract_size

        val1 = (contract_size * A * s_close - loss)
        val2 = B * k * contract_size
        IM = max(val1, val2)

        # Exception for 'موج'
        if underlying in ['موج']:
            round_multiplier = 100000
        else:
            round_multiplier = 10000

        val = ((math.floor(IM / round_multiplier) + 1) * round_multiplier) + c * contract_size
        return val / contract_size

    @staticmethod
    def black_scholes(s, k, ttm, r, sigma, option_type):
        """
        Plain-vanilla Black–Scholes formula.
        ttm = time to maturity (days)
        r   = risk-free rate
        sigma = implied volatility
        """
        T = ttm / 365.0
        if T <= 0:
            # No time left, option is worth intrinsic value
            intrinsic = max(s - k, 0) if option_type == 'call' else max(k - s, 0)
            return intrinsic

        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
        else:  # 'put'
            return k * np.exp(-r * T) * norm.cdf(-d2) - s * norm.cdf(-d1)

    @staticmethod
    def implied_volatility(S, K, market_price, ttm, r, option_type, tol=1e-5, max_iter=60):
        """
        A bisection method to find implied volatility:
        - S: Underlying price
        - K: Strike
        - market_price: observed option price
        - ttm: time to maturity (days)
        - r: risk-free rate
        - option_type: 'call' or 'put'
        """
        if ttm <= 0 or market_price <= 0:
            return 0.0

        low_vol = 1e-5
        high_vol = 5.0

        for _ in range(max_iter):
            mid_vol = (low_vol + high_vol) / 2.0
            price = Market.black_scholes(S, K, ttm, r, mid_vol, option_type)

            if abs(price - market_price) < tol:
                return mid_vol

            if price > market_price:
                high_vol = mid_vol
            else:
                low_vol = mid_vol

        return 0.0

    @staticmethod
    def delta(s, k, ttm, r, sigma, option_type):
        """Delta of a European option under Black–Scholes."""
        T = ttm / 365.0
        if T <= 0:
            # No time left
            if option_type == 'call':
                return 1.0 if s > k else 0.0
            else:  # put
                # Some define put delta at expiry as -1 if ITM
                return -1.0 if k > s else 0.0

        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return norm.cdf(d1)
        else:  # 'put'
            # Another convention: delta(put) = N(d1) - 1
            return norm.cdf(d1) - 1.0

    @staticmethod
    def gamma(s, k, ttm, r, sigma):
        """Gamma of a European option under Black–Scholes."""
        T = ttm / 365.0
        if T <= 0:
            return 0.0

        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        n_prime_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        return n_prime_d1 / (s * sigma * np.sqrt(T))

    @staticmethod
    def get_last_price(x, total_df):
        """Look up last price of underlying in total_df by matching 'lva'."""
        x = x.strip()
        try:
            return total_df[total_df['lva'] == x]['pdv'].values[0]
        except:
            return None

    @staticmethod
    def get_close_price(x, total_df):
        """Look up close price of underlying in total_df by matching 'lva'."""
        x = x.strip()
        try:
            return total_df[total_df['lva'] == x]['pcl'].values[0]
        except:
            return None

    @classmethod
    def get_option_market_watch(cls, Greeks=False, risk_free_rate=0.3):
        """
        Get option market data from TSE, compute implied volatility, delta, gamma, etc.
        Pass `risk_free_rate` to override default (0.3).
        """
        url = (
            'https://cdn.tsetmc.com/api/ClosingPrice/GetMarketWatch?market=0&industrialGroup=&paperTypes[0]=6&paperTypes[1]=2&paperTypes[2]=1&paperTypes[3]=8&showTraded=false&withBestLimits=true&hEven=0&RefID=0'
        )
        res = requests.get(url, headers=cls.headers)
        market_watch = res.json()['marketwatch']
        df = pd.DataFrame(market_watch)

        # Extract best limits
        temp = df['blDs'].apply(lambda x: cls.extract_values(x))
        extracted_df = pd.DataFrame(temp.tolist(), columns=['pmd1', 'qmd1', 'pmo1', 'qmo1'])
        temp_df = pd.concat([df, extracted_df], axis=1)
        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'lvc', 'pdv', 'qtc', 'pMax', 'pMin', 'py',
            'pcl', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ztd'
        ]]

        total_df = temp_df.copy()

        # Filter only option symbols
        temp_df = temp_df[temp_df['lvc'].str.contains('اختيار')]
        temp_df['type'] = 'call'
        temp_df.loc[temp_df['lvc'].str.contains('اختيارف'), 'type'] = 'put'

        temp_df['lva'] = temp_df['lva'].apply(lambda x: Helpers.characters_modifier(x))
        temp_df[['underlying', 'strike', 'maturity']] = temp_df['lvc'].str.split('-', expand=True)

        # Clean up the 'underlying' field
        temp_df['underlying'] = temp_df['underlying'].replace(r'اختيارف', '', regex=True)\
                                                     .replace(r'اختيارخ', '', regex=True)

        # Select final columns
        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'type', 'underlying',
            'strike', 'maturity', 'pmd1', 'qmd1', 'pmo1', 'qmo1',
            'pdv', 'qtc', 'pMax', 'pMin', 'py', 'pcl', 'ztd'
        ]]

        # Validate maturity
        temp_df = temp_df[temp_df['maturity'].apply(cls.maturity_validation)]
        temp_df = temp_df[~temp_df['maturity'].isna()]

        # Normalize underlying
        temp_df['underlying'] = temp_df['underlying'].apply(lambda x: Helpers.characters_modifier(x))

        # Convert maturity to Gregorian, then compute TTM
        temp_df['g_maturity'] = temp_df['maturity'].apply(lambda x: Helpers.to_gregorian_date(x))
        temp_df['ttm'] = temp_df['g_maturity'].apply(Helpers.cal_ttm)

        # Map underlying's last/close price
        # temp_df['ua_last_price'] = temp_df['underlying'].apply(lambda x: cls.get_last_price(x, total_df))
        # temp_df['ua_close_price'] = temp_df['underlying'].apply(lambda x: cls.get_close_price(x, total_df))
        temp_df['ua_last_price'] = None
        temp_df['ua_close_price'] = None

        # temp_df = temp_df[~temp_df['ua_last_price'].isna()]

        temp_df['strike'] = temp_df['strike'].astype(float)

        temp_df['market'] = 'option'

        temp_df[['IV', 'delta', 'gamma']] = None, None, None

        if Greeks:

            # Calculate Implied Vol, Delta, Gamma
            temp_df['IV'] = temp_df.apply(
                lambda row: cls.implied_volatility(
                    row['ua_last_price'], row['strike'], row['pdv'],
                    row['ttm'], risk_free_rate, row['type']
                ),
                axis=1
            )

            temp_df['delta'] = temp_df.apply(
                lambda row: cls.delta(
                    row['ua_last_price'], row['strike'], row['ttm'],
                    risk_free_rate, row['IV'], row['type']
                ),
                axis=1
            )

            temp_df['gamma'] = temp_df.apply(
                lambda row: cls.gamma(
                    row['ua_last_price'], row['strike'], row['ttm'],
                    risk_free_rate, row['IV']
                ),
                axis=1
            )
            

        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'market', 'type', 'underlying', 'strike', 'ttm',
            'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1', 'qmo1',
            'ua_last_price', 'ua_close_price', 'IV', 'delta', 'gamma'
        ]]
        temp_df.columns = [
            'id', 'code', 'symbol', 'market', 'type', 'underlying', 'strike', 'ttm',
            'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P',
            'ask_Q', 'ua_last', 'ua_close', 'IV', 'delta', 'gamma'
        ]
        return temp_df

    @classmethod
    def get_stock_market_watch(cls):
        """
        Fetch stock market data from TSE.
        """
        url = (
            'https://cdn.tsetmc.com/api/ClosingPrice/GetMarketWatch'
            '?market=0&industrialGroup=&paperTypes%5B0%5D=2&paperTypes%5B1%5D=1'
            '&paperTypes%5B2%5D=8&showTraded=false'
            '&withBestLimits=true&hEven=0&RefID=0'
        )
        res = requests.get(url, headers=cls.headers)
        market_watch = res.json()['marketwatch']
        df = pd.DataFrame(market_watch)

        temp = df['blDs'].apply(lambda x: cls.extract_values(x))
        extracted_df = pd.DataFrame(temp.tolist(), columns=['pmd1', 'qmd1', 'pmo1', 'qmo1'])
        temp_df = pd.concat([df, extracted_df], axis=1)
        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'lvc', 'pdv', 'qtc', 'pMax', 'pMin',
            'py', 'pcl', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'ztd'
        ]]

        # Normalize symbol
        temp_df['lva'] = temp_df['lva'].apply(lambda x: Helpers.characters_modifier(x))

        # Create dummy columns to align with the option structure
        temp_df[['type', 'underlying', 'strike', 'ttm', 'ua_last_price', 'ua_close_price', 'IV', 'delta', 'gamma']] = None
        temp_df['market'] = 'stock'

        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'market', 'type', 'underlying', 'strike',
            'ttm', 'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1',
            'qmo1', 'ua_last_price', 'ua_close_price'
        ]]
        temp_df.columns = [
            'id', 'code', 'symbol', 'market', 'type', 'underlying', 'strike', 'ttm',
            'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P',
            'ask_Q', 'ua_last', 'ua_close'
        ]
        return temp_df

    @classmethod
    def get_market_watch(cls, Greeks=False, risk_free_rate=0.3):
        """
        Combine option and stock market data.
        Pass `risk_free_rate` to apply in option pricing calculations.
        """
        tse_options = cls.get_option_market_watch(Greeks=Greeks, risk_free_rate=risk_free_rate)
        tse_stock = cls.get_stock_market_watch()
        return pd.concat([tse_options, tse_stock])

    @staticmethod
    def save_market_watch(tse_data):
        """Save to an Excel file with a timestamp."""
        time_obj = datetime.datetime.now()
        time_string = time_obj.strftime('%Y%m%d___%H-%M-%S')
        os.makedirs('options', exist_ok=True)
        tse_data.to_excel(f'options/{time_string}.xlsx', index=False)

    @classmethod
    def get_firms_info(cls, symbols=None, fetch_all=False):
        """
        Reads firms_info.csv and returns a list of (symbol, id).

        If neither `symbols` nor `fetch_all` is provided, all rows are returned.
        Otherwise, if `fetch_all` is True, all rows are returned.
        If `fetch_all` is False and `symbols` is provided, only rows with a symbol in `symbols`
        are included.

        Parameters:
            symbols (iterable, optional): An iterable of symbols to filter by. Default is None.
            fetch_all (bool, optional): Whether to ignore filtering and return all rows. Default is False.

        Returns:
            List[Tuple[str, str]]: A list of (symbol, id) tuples.
        """
        # If symbols is not provided, default to returning all rows.
        if symbols is None:
            fetch_all = True

        base_path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_path, 'firms_info.csv')

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fetch_all:
                firms_info = [(row['symbol'], row['id']) for row in reader]
            else:
                firms_info = [
                    (row['symbol'], row['id']) 
                    for row in reader if row['symbol'] in symbols
                ]

        return firms_info

    @classmethod
    def _fetch_symbol_data(cls, symbol_id_tuple):
        """
        Fetch JSON data for a single (symbol, id) pair.
        Skips symbols with digits or ending with 'ح'.
        """
        symbol, firm_id = symbol_id_tuple

        if cls.bad_symbol_pattern.search(symbol):
            return None

        url = f'https://cdn.tsetmc.com/api/Instrument/GetInstrumentInfo/{str(firm_id).strip()}'

        try:
            with requests.Session() as session:
                # Update session with class-level headers
                session.headers.update(cls.headers)
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                instrument_info = resp.json().get('instrumentInfo', {})
                zTitad_value = instrument_info.get('zTitad', None)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

        return {
            'symbol': symbol,
            'id': firm_id,
            'shares': zTitad_value
        }

    @classmethod
    def get_market_cap(cls, symbols=None, max_workers=8):
        """
        Fetches market capitalization data for the given symbols (list of strings).
        If symbols is None, fetch_all=True => fetch all entries from CSV.
        """
        fetch_all = (symbols is None)
        firms_info = cls.get_firms_info(symbols, fetch_all)

        all_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(cls._fetch_symbol_data, fi): fi for fi in firms_info}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_data.append(result)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

        return all_data


    @classmethod
    def get_total_options_symbol(cls):
        market_watch_options = Market.get_option_market_watch()
        option_tickers = list(market_watch_options['symbol'].str.replace(r'[0-9]*', '', regex=True).unique())
        my_option_tickers = ['ضپادا', 'ضهمن', 'ضخود', 'ضستا', 'ضفلا', 'طملت', 'ضذوب', 'ضشنا', 'طهرم', 'ضرویین', 'طپادا', 'ضجار', 'ضسرو', 'ضتاب', 'ضکرومیت', 'طپتروآبان', 'طنارنج', 'طذوب', 'ضران', 'ضملت', 'ضخپارس', 'ضتیام', 'ضنارنج', 'طتاب', 'طکرمان', 'طسپا', 'طستر', 'ضفزر', 'طران', 'طستا', 'ضفلزفارابی', 'ضبساما', 'ضهرم', 'ضتوان', 'ضکرمان', 'طجهش', 'طشنا', 'طملی', 'ضاساس', 'طثمین', 'طکرومیت', 'طخود', 'طوتعاون', 'ضسپا', 'طجار', 'طرویین', 'ضملی', 'ضوتعاون', 'طموج', 'ضستر', 'طهمن', 'طخپارس', 'طفزر', 'ضموج', 'طاطلس', 'طفلا', 'طتوان', 'طبساما', 'ضسامان', 'ضبید', 'ضخاور', 'ضاطلس', 'ضجهش', 'طسامان', 'هامین شهر', 'ضثمین', 'ضپتروآبان', 'ضکاریس', 'طاساس', 'طخاور', 'طتیام', 'طفلزفارابی', 'طکاریس', 'طسرو', 'ضصاد', 'طصاد']
        option_tickers = list(set(option_tickers + my_option_tickers))
        option_tickers = market_watch_options['symbol'].str.replace(r'[0-9]*', '', regex=True).unique()
        df = pd.DataFrame()
        for option_ticker in option_tickers:
            url = 'https://rahavard365.com/api/v2/search?keyword=' + str(option_ticker)
            res = requests.get(url, headers=cls.headers)
            res_json = res.json()['data']
            df = pd.concat([df, pd.DataFrame(res_json)])
        df = df[df.type_id == '16']
        df['trade_symbol'] = df['trade_symbol'].str.strip()
        return df[['trade_symbol', 'name']].values


    @staticmethod
    def get_tes_id_by_symbol(symbols: list):
        driver = Firefox()

        thershold = 5
        while thershold:
            try:
                driver.get('https://www.tsetmc.com/MarketOverall')
                driver.find_element(By.XPATH, '//*[@id="search"]').click()
                break
            except:
                thershold -= 1
                time.sleep(1)

        pattern = r'href="/instInfo/(\d+)"'

        data = []
        for symbol in symbols:

            thershold = 3
            while thershold:
                driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').clear()
                driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').send_keys(str(symbol))
                time.sleep(.7)
                # driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').send_keys(str(symbol)[-1])
                # time.sleep(.1)

                try:
                    # result rows
                    result_box = driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/div/div[2]/div/div/div/div[1]/div[2]/div[3]/div[2]/div/div')
                    result_rows = result_box.find_elements(By.CLASS_NAME, 'ag-row')
                    if not len(result_rows):
                        break

                    for row in result_rows:
                        result_title = row.find_element(By.TAG_NAME, 'div')
                        result_symbol = re.search(r'(.*)-(.*)', result_title.text.strip())[1]

                        if Helpers.characters_modifier(result_symbol.strip()) == Helpers.characters_modifier(symbol.strip()):
                            href = result_title.find_element(By.TAG_NAME, 'a').get_attribute('outerHTML')
                            id = re.search(pattern, href)[1]
                            data.append([symbol, id])
                            thershold = 0
                            break
                except:
                    thershold -= 1
                    time.sleep(.1)
                    continue

        return data



# print(Market.get_firms_info(['فملی', 'فولاد']))
# print(Market.get_option_market_watch(Greeks=True))
# print(Market.get_market_watch(Greeks=True))
# print(Market.get_stock_market_watch())

# print(Market.get_tes_id_by_symbol(symbols=['فولاد', 'فملی', 'شاراک']))


