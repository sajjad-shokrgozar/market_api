# market/market.py

import os
import re
import csv
import math
import time
import requests
import datetime
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
    def vega(s, k, ttm, r, sigma):
        """Vega of a European option under Black–Scholes."""
        T = ttm / 365.0
        if T <= 0:
            return 0.0

        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        n_prime_d1 = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        return s * np.sqrt(T) * n_prime_d1

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
    def get_option_market_watch(cls, Greeks=False, risk_free_rate=0.3, iv_prime_ua_price_multiplier=0):
        """
        Get option market data from TSE, compute implied volatility, delta, gamma, vega, etc.
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
            'insCode', 'insID', 'lva', 'lvc', 'type', 'underlying',
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
        # temp_df = temp_df[temp_df['g_maturity'] != 0]
        temp_df['ttm'] = temp_df['g_maturity'].apply(Helpers.cal_ttm)

        # Map underlying's last/close price
        temp_df['ua_last_price'] = temp_df['underlying'].apply(lambda x: cls.get_last_price(x, total_df))
        temp_df['ua_close_price'] = temp_df['underlying'].apply(lambda x: cls.get_close_price(x, total_df))
        # temp_df['ua_last_price'] = None
        # temp_df['ua_close_price'] = None

        # temp_df = temp_df[~temp_df['ua_last_price'].isna()]

        temp_df['strike'] = temp_df['strike'].astype(float)

        temp_df['market'] = 'option'

        temp_df[['IV', 'IV_prime', 'delta', 'gamma', 'vega']] = None, None, None, None, None

        if Greeks:

            # Calculate Implied Vol, Delta, Gamma, Vega
            temp_df['IV'] = temp_df.apply(
                lambda row: cls.implied_volatility(
                    row['ua_last_price'], row['strike'], (row['pmd1'] + row['pmo1']) / 2,
                    row['ttm'], risk_free_rate, row['type']
                ),
                axis=1
            )
            if iv_prime_ua_price_multiplier != 0:
                temp_df['IV_prime'] = temp_df.apply(
                    lambda row: cls.implied_volatility(
                        row['ua_last_price'], row['strike'], row['pdv'] + row['ua_last_price'] * iv_prime_ua_price_multiplier,
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

            temp_df['vega'] = temp_df.apply(
                lambda row: cls.vega(
                    row['ua_last_price'], row['strike'], row['ttm'],
                    risk_free_rate, row['IV']
                ),
                axis=1
            )
            

        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'lvc', 'market', 'type', 'underlying', 'strike', 'ztd', 'ttm',
            'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1', 'qmo1', 'pMax', 'pMin',
            'ua_last_price', 'ua_close_price', 'IV', 'IV_prime', 'delta', 'gamma', 'vega'
        ]]
        temp_df.columns = [
            'id', 'code', 'symbol', 'name', 'market', 'type', 'underlying', 'strike', 'size', 'ttm',
            'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P',
            'ask_Q', 'max_limit', 'min_limit', 'ua_last', 'ua_close', 'IV', 'IV_prime', 'delta', 'gamma', 'vega'
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
        temp_df[['type', 'underlying', 'strike', 'ttm', 'ua_last_price', 'ua_close_price', 'IV', 'IV_prime', 'delta', 'gamma', 'vega']] = None
        temp_df['market'] = 'stock'

        temp_df = temp_df[[
            'insCode', 'insID', 'lva', 'lvc', 'market', 'type', 'underlying', 'strike',
            'ttm', 'pcl', 'pdv', 'qtc', 'pmd1', 'qmd1', 'pmo1',
            'qmo1', 'ztd', 'pMax', 'pMin', 'ua_last_price', 'ua_close_price'
        ]]
        temp_df.columns = [
            'id', 'code', 'symbol', 'name', 'market', 'type', 'underlying', 'strike', 'ttm',
            'close', 'last', 'volume', 'bid_P', 'bid_Q', 'ask_P',
            'ask_Q', 'size', 'max_limit', 'min_limit', 'ua_last', 'ua_close'
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
        """
        Retrieves the total option symbols available in the market.

        This method fetches option market watch data, extracts unique option tickers,
        combines them with a predefined list of option tickers, and then queries an API
        to obtain additional details for each ticker.

        Returns:
            numpy.ndarray: An array containing trade symbols and corresponding names.
        """
        market_watch_options = Market.get_option_market_watch()
        option_tickers = list(market_watch_options['symbol'].str.replace(r'[0-9]*', '', regex=True).unique())

        # Predefined list of option tickers
        my_option_tickers = ['ضپادا', 'ضهمن', 'ضخود', 'ضستا', 'ضفلا', 'طملت', 'ضذوب', 'ضشنا', 'طهرم', 
                             'ضرویین', 'طپادا', 'ضجار', 'ضسرو', 'ضتاب', 'ضکرومیت', 'طپتروآبان', 'طنارنج', 
                             'طذوب', 'ضران', 'ضملت', 'ضخپارس', 'ضتیام', 'ضنارنج', 'طتاب', 'طکرمان', 
                             'طسپا', 'طستر', 'ضفزر', 'طران', 'طستا', 'ضفلزفارابی', 'ضبساما', 'ضهرم', 
                             'ضتوان', 'ضکرمان', 'طجهش', 'طشنا', 'طملی', 'ضاساس', 'طثمین', 'طکرومیت', 
                             'طخود', 'طوتعاون', 'ضسپا', 'طجار', 'طرویین', 'ضملی', 'ضوتعاون', 'طموج', 
                             'ضستر', 'طهمن', 'طخپارس', 'طفزر', 'ضموج', 'طاطلس', 'طفلا', 'طتوان', 
                             'طبساما', 'ضسامان', 'ضبید', 'ضخاور', 'ضاطلس', 'ضجهش', 'طسامان', 'هامین شهر', 
                             'ضثمین', 'ضپتروآبان', 'ضکاریس', 'طاساس', 'طخاور', 'طتیام', 'طفلزفارابی', 
                             'طکاریس', 'طسرو', 'ضصاد', 'طصاد']
        
        option_tickers = list(set(option_tickers + my_option_tickers))
        df = pd.DataFrame()

        for option_ticker in option_tickers:
            url = f'https://rahavard365.com/api/v2/search?keyword={option_ticker}'
            res = requests.get(url, headers=cls.headers)
            res_json = res.json()['data']
            df = pd.concat([df, pd.DataFrame(res_json)])

        df = df[df.type_id == '16']
        df['trade_symbol'] = df['trade_symbol'].str.strip()
        return df[['trade_symbol', 'name']].values


    @staticmethod
    def get_tes_id_by_symbol(symbols: list):
        """
        Retrieves the TES (Tehran Stock Exchange) ID for given symbols.

        This method automates a browser session using Selenium to search for 
        stock symbols on the Tehran Stock Exchange website and extract their IDs.

        Args:
            symbols (list): A list of stock symbols.

        Returns:
            list: A list of lists containing symbol and corresponding TES ID.
        """
        driver = Firefox()
        symbols = [Helpers.inverse_characters_modifier(symbol) for symbol in symbols]
        threshold = 5

        while threshold:
            try:
                driver.get('https://www.tsetmc.com/MarketOverall')
                driver.find_element(By.XPATH, '//*[@id="search"]').click()
                break
            except:
                threshold -= 1
                time.sleep(1)

        pattern = r'href="/instInfo/(\d+)"'
        data = []

        for symbol in symbols:
            threshold = 3
            while threshold:
                driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').clear()
                driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').send_keys(str(symbol[:-1]))
                time.sleep(.7)
                driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/input').send_keys(str(symbol[-1]))
                time.sleep(1)


                try:
                    result_box = driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/div/div/div[2]/div/div/div/div[1]/div[2]/div[3]/div[2]/div/div')
                    result_rows = result_box.find_elements(By.CLASS_NAME, 'ag-row')
                    # return result_rows
                    if not len(result_rows):
                        break

                    for row in result_rows:
                        result_title = row.find_element(By.TAG_NAME, 'div')
                        result_symbol = re.search(symbol, result_title.text.strip())[0]

                        if Helpers.characters_modifier(result_symbol.strip()) == Helpers.characters_modifier(symbol.strip()):
                            href = result_title.find_element(By.TAG_NAME, 'a').get_attribute('outerHTML')
                            tes_id = re.search(pattern, href)[1]
                            with open('symbol_id_data.txt', 'a', encoding='utf8') as f:
                                f.write(str(symbol) + ';' + str(tes_id) + '\n')
                            data.append([symbol, tes_id])
                            threshold = 0
                            break
                except:
                    threshold -= 1
                    time.sleep(0.1)
                    if threshold == 0:
                        with open('symbol_id_error.txt', 'a', encoding='utf8') as f:
                            f.write(str(symbol) + '\n')
                    continue

        return data


    @classmethod
    def market_covered_call_return(cls, underlying_list='all', min_ttm=15, max_ttm=60, min_rcut=-40, max_rcut=-10, min_r=-20):
        """
        Calculates the return of covered call options in the market based on given filters.

        Parameters:
        ----------
        underlying_list : list or str, optional
            A list of underlying assets to filter for. Defaults to 'all' (includes all available assets).
        min_ttm : int, optional
            The minimum time to maturity (TTM) in days. Defaults to 15.
        max_ttm : int, optional
            The maximum time to maturity (TTM) in days. Defaults to 60.
        min_rcut : float, optional
            The minimum required cut-off return (r_cut) percentage. Defaults to -40.
        max_rcut : float, optional
            The maximum required cut-off return (r_cut) percentage. Defaults to -10.
        min_r : float, optional
            The minimum required return percentage. Defaults to -20.

        Returns:
        -------
        dict:
            A dictionary containing:
            - 'market_r': The weighted average return of the filtered covered call options.
            - 'details': A DataFrame of filtered covered call options, sorted by return ('r').
        
        Notes:
        ------
        - The function retrieves stock and option market data, merges them, and filters for covered call opportunities.
        - The function currently uses a manually defined list of funds that should be populated dynamically.
        - Various trading fees are considered based on the asset type (stock or fund).
        - The final return ('r') is calculated using the formula for annualized return based on bid and ask prices.
        - The weighted average return ('market_r') is computed based on trading volume.
        """

        stock_df = cls.get_stock_market_watch()
        option_df = cls.get_option_market_watch()
        market_watch = pd.concat([stock_df, option_df])

        # FIXME: Fill this list automatically, not manually
        funds_list = ['اهرم', 'توان', 'موج', 'جهش', 'نارنج اهرم', 'آساس', 'اطلس', 'بیدار', 'کاریس', 'آگاس', 'شتاب', 'خودران', 'سرو', 'بساما', 'پتروپاداش', 'رویین']

        market_watch = market_watch[['id', 'symbol', 'market', 'underlying', 'strike', 'ttm', 'min_limit', 'max_limit', 'bid_P', 'bid_Q', 'ask_P', 'ask_Q', 'volume']]

        market_watch_option_df = market_watch[market_watch['market'] == 'option']
        market_watch_stock_df = market_watch[market_watch['market'] == 'stock']
        market_watch_option_df.columns = ['op_' + column for column in market_watch_option_df.columns]
        market_watch_stock_df.columns = ['st_' + column for column in market_watch_stock_df.columns]

        cc_df = pd.merge(market_watch_option_df, market_watch_stock_df, how='left', left_on='op_underlying', right_on='st_symbol')
        cc_df = cc_df[['op_id', 'op_symbol', 'op_underlying', 'op_strike', 'op_ttm', 'op_volume', 'op_ask_P', 'op_ask_Q', 'op_bid_P', 'op_bid_Q', 'st_ask_P', 'st_ask_Q', 'st_bid_P', 'st_bid_Q', 'st_min_limit', 'st_max_limit']]
        cc_df['ua_type'] = 'stock'
        cc_df.loc[cc_df['op_underlying'].isin(funds_list), 'ua_type'] = 'fund'
        cc_df.loc[cc_df['ua_type'] == 'stock', 'ua_sell_fee'] = 0.0088
        cc_df.loc[cc_df['ua_type'] == 'stock', 'ua_buy_fee'] = 0.003712
        cc_df.loc[cc_df['ua_type'] == 'stock', 'short_settlement_fee'] = 0.0055
        cc_df.loc[cc_df['ua_type'] == 'stock', 'long_settlement_fee'] = 0.0005
        cc_df.loc[cc_df['ua_type'] == 'fund', 'ua_sell_fee'] = 0.0011875
        cc_df.loc[cc_df['ua_type'] == 'fund', 'ua_buy_fee'] = 0.00116
        cc_df.loc[cc_df['ua_type'] == 'fund', 'short_settlement_fee'] = 0.0005
        cc_df.loc[cc_df['ua_type'] == 'fund', 'long_settlement_fee'] = 0.0005
        cc_df['op_fee'] = .00103
        cc_df['r_cut'] = cc_df['op_strike'] / cc_df['st_ask_P'] - 1

        #FIXME: برای صف خرید ها قیمت صف گذاشته بشه. الان قیمت بالاتر یا پایینتر از صف رو میزاره
        # above problem fixed
        cc_df.loc[cc_df['st_ask_P'] > cc_df['st_max_limit'], 'st_ask_P'] = cc_df.loc[cc_df['st_ask_P'] > cc_df['st_max_limit'], 'st_max_limit']
        cc_df.loc[cc_df['st_bid_P'] < cc_df['st_min_limit'], 'st_bid_P'] = cc_df.loc[cc_df['st_bid_P'] < cc_df['st_min_limit'], 'st_min_limit']
        # end

        cc_df = cc_df[cc_df['op_ttm'] != 0]
        cc_df['r'] = ((cc_df['op_strike'] * (1 - cc_df['short_settlement_fee'])) / (cc_df['st_ask_P'] * (1 + cc_df['ua_buy_fee']) - cc_df['op_bid_P'] * (1 - cc_df['op_fee']))) ** (365/cc_df['op_ttm']) - 1
        cc_df = cc_df[['op_symbol', 'op_underlying', 'op_ttm', 'op_bid_P', 'st_ask_P', 'op_volume', 'r_cut', 'r']]
        cc_df.columns = ['symbol', 'underlying', 'ttm', 'op_bid_P', 'st_ask_P', 'traded_value', 'r_cut', 'r']
        cc_df = cc_df.applymap(lambda x: None if isinstance(x, complex) else x)
        cc_df = cc_df[(~cc_df['r'].isna()) & (~cc_df['r_cut'].isna())]
        cc_df['r_cut'] = cc_df['r_cut'].apply(lambda x: round(x * 100, 1))
        cc_df['r'] = cc_df['r'].apply(lambda x: round(x * 100, 1))

        filterd_cc_df = cc_df[(cc_df['ttm'] > min_ttm) & (cc_df['ttm'] < max_ttm) & (cc_df['r_cut'] > min_rcut) & (cc_df['r_cut'] < max_rcut) & (cc_df['r'] > min_r)]
        if underlying_list != 'all':
            filterd_cc_df = filterd_cc_df[(filterd_cc_df['underlying'].isin(underlying_list))]
        filterd_cc_df = filterd_cc_df.sort_values('r', ascending=False)[['symbol', 'underlying', 'ttm', 'traded_value', 'r_cut', 'r']]
        
        filterd_cc_df['weighted_r'] = filterd_cc_df['traded_value'] * filterd_cc_df['r']
        market_r = filterd_cc_df['weighted_r'].sum() / filterd_cc_df['traded_value'].sum()

        return {
            'market_r': round(float(market_r), 1),
            'details': filterd_cc_df
        }
    
