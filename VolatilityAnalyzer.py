
import numpy as np
from numpy.fft import fft
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class VolatilityAnalyzer:
    def __init__(self):
        """
        PriceAnalyzer class for analyzing price data.

        Parameters:
        - data: Pandas DataFrame containing price data.
        """
        #self.data = data

    def get_statistical_volatility(self, data, lookback, selected_price):
        """
        Calculate statistical volatility based on the mean and standard deviation of the selected price.

        Parameters:
        - lookback: Number of recent data points to consider.
        - selected_price: Name of the column representing the selected price.

        Returns:
        - Statistical Volatility values for each rolling window.
        """
        # Ensure the data is sorted by index if it's not already sorted
        data = data.sort_index()

        # Apply the rolling window calculation
        data['Statistical'] = data[selected_price].rolling(window=lookback).apply(self.calculate_volatility, raw=False)

        data = data.dropna()
        data = data.copy()

        #data['Statistical'] = MinMaxScaler().fit_transform(data[['Statistical_unscaled']])
        print('STATISTICAL VOLATILITY')
        print(data.head())
        print('................................................')
        return data

    def calculate_volatility(self, price_series):
        """
        Helper function to calculate volatility based on the mean and standard deviation.

        Parameters:
        - price_series: Series containing the selected price data.

        Returns:
        - Statistical Volatility value.
        """
        price_mean = np.mean(price_series)
        price_std = np.std(price_series)

        volatility = 1 - (price_mean - price_std) / price_mean
        volatility = volatility * 2
        return volatility

    def get_spectral_volatility(self, data, lookback, selected_price):
        """
        Calculate spectral volatility based on the power spectral density (PDS) of the selected price.

        Parameters:
        - lookback: Number of recent data points to consider.
        - selected_price: Name of the column representing the selected price.

        Returns:
        - Spectral Volatility values for each rolling window.
        """
        # Ensure the data is sorted by index if it's not already sorted
        data = data.sort_index()

        # Apply the rolling window calculation
        data['PDS'] = data[selected_price].rolling(window=lookback).apply(self.calculate_spectral_volatility, raw=False)
        
        # Calculate PDS ratio
        data['Spectral'] = data['PDS'].pct_change() -1  # (current PDS Ratio)/(previous PDS Ratio) - 1

        data = data.dropna()
        data = data.copy()
        #data['Spectral'] = MinMaxScaler().fit_transform(data[['Spectral_unscaled']])

        print('STATISTICAL VOLATILITY')
        print(data.head())
        print('................................................')
        return data

    def calculate_spectral_volatility(self, price_series):
        """
        Helper function to calculate spectral volatility based on the power spectral density.

        Parameters:
        - price_series: Series containing the selected price data.

        Returns:
        - Spectral Volatility value.
        """
        myfft = fft(price_series)
        
        n = len(price_series)
        T_long = 1.0
        T_mid = 1.0
        T_short = 1.0

        f_long = np.linspace(0.0, 1.0 / (2.0 * T_long), n // 2)
        pds = 2.0 / n * np.abs(myfft[:n // 2])

        low_freq_idx = f_long < 0.1
        high_freq_idx = f_long >= 0.1

        low_freq_power = np.mean(pds[low_freq_idx])
        high_freq_power = np.mean(pds[high_freq_idx])

        if high_freq_power == 0:
            return 0
        else:
            return low_freq_power / high_freq_power
    
if __name__ == '__main__':
    
    data = {
            'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
            'Close': [10, 9, 10,9, 11, 11, 11, 9, 10, 10]
        }
    price_data = pd.DataFrame(data)
    volatility = VolatilityAnalyzer()
    vol_stat = volatility.get_statistical_volatility(price_data, 10, 'Close')
    vol_spec = volatility.get_spectral_volatility(price_data, 10, 'Close')
    
    print(f"Statistical Volatility : {vol_stat}")
    print(f"Spectral Volatility : {vol_spec}")