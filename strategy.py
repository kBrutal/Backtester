from scipy.signal import butter, cheby1, filtfilt
import pandas as pd
import numpy as np
from easydict import EasyDict
from scipy.signal import get_window, fftconvolve
from scipy.ndimage import gaussian_filter1d
from pykalman import KalmanFilter


class BaseStrategy:
    def __init__(
        self,
        high_csv: pd.DataFrame,
        low_csv: pd.DataFrame,
        config: EasyDict,
        glob: EasyDict,
    ):
        self.high_csv = high_csv
        self.shifted_high_csv = high_csv.shift(1)
        self.low_csv = low_csv
        self.config = config
        self.glob = glob
        self.preprocessing()

    def preprocessing(self):
        pass

    def check_long_entry(self, high_pointer: int):
        pass

    def check_short_entry(self, high_pointer: int):
        pass

    def check_long_exit(self, high_pointer: int):
        pass

    def check_short_exit(self, high_pointer: int):
        pass


class EMAStrategy(BaseStrategy):
    def preprocessing(self):
        self.high_csv["long_EMA"] = (
            self.high_csv["close"].ewm(span=12, adjust=False).mean()
        )
        self.high_csv["short_EMA"] = (
            self.high_csv["close"].ewm(span=9, adjust=False).mean()
        )

    def check_long_entry(self, high_pointer: int):
        long_ema = self.high_csv["long_EMA"].iloc[high_pointer]
        short_ema = self.high_csv["short_EMA"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if short_ema > long_ema:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        long_ema = self.high_csv["long_EMA"].iloc[high_pointer]
        short_ema = self.high_csv["short_EMA"].iloc[high_pointer]
        Close = self.high_csv["close"].iloc[high_pointer]
        if short_ema < long_ema:
            self.glob.tp = 0.1
            self.glob.sl = 0.05
            self.glob.entry_price = Close
            self.glob.trailing_price = Close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        return 0

    def check_short_exit(self, high_pointer: int):
        return 0


class My_Strategy_9(BaseStrategy):

    def hma(self, length=5):
        """Calculate the Hull Moving Average (HMA)."""
        import numpy as np
        data = self.high_csv
        length = int(length)
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))

        # Weighted Moving Average (WMA) function
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

        # Calculate WMA for full length and half length
        wma_full = wma(data['close'], length)
        wma_half = wma(data['close'], half_length)

        # Calculate the difference
        diff = 2 * wma_half - wma_full

        # HMA is WMA of diff over sqrt(length)
        hma = wma(diff, sqrt_length)
        return hma

    def laguerre_filter(self, alpha=0.85):
        """Calculate the Laguerre Filter."""
        import numpy as np
        data = self.high_csv
        gamma = 1 - alpha
        src = (data['high'] + data['low']) / 2  # hl2

        L0 = np.zeros(len(src))
        L1 = np.zeros(len(src))
        L2 = np.zeros(len(src))
        L3 = np.zeros(len(src))
        LagF = np.zeros(len(src))

        for i in range(len(src)):
            if i == 0:
                L0_prev = L1_prev = L2_prev = L3_prev = 0
            else:
                L0_prev = L0[i - 1]
                L1_prev = L1[i - 1]
                L2_prev = L2[i - 1]
                L3_prev = L3[i - 1]

            L0[i] = (1 - gamma) * src.iloc[i] + gamma * L0_prev
            L1[i] = -gamma * L0[i] + L0_prev + gamma * L1_prev
            L2[i] = -gamma * L1[i] + L1_prev + gamma * L2_prev
            L3[i] = -gamma * L2[i] + L2_prev + gamma * L3_prev

            LagF[i] = (L0[i] + 2 * L1[i] + 2 * L2[i] + L3[i]) / 6

        return pd.Series(LagF, index=data.index)

    def tema(self, length=800):
        """Calculate the Triple Exponential Moving Average (TEMA)."""
        close = self.high_csv['close']
        # Calculate the first EMA
        ema1 = close.ewm(span=length, adjust=False).mean()
        # Calculate the second EMA
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        # Calculate the third EMA
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        # Calculate TEMA
        tema = 3 * (ema1 - ema2) + ema3
        return tema

    def hawkes_process(self, k=3):
        """Implement Hawkes process for volatility estimation."""
        kappa = k
        alpha = np.exp(-kappa)
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)

        output = np.zeros(len(self.high_csv))
        output[0] = self.high_csv['close'].iloc[0]

        for i in range(1, len(self.high_csv)):
            output[i] = output[i - 1] * alpha + self.high_csv['close'].iloc[i]

        return pd.Series(output * kappa, index=self.high_csv.index)

    def adx(self, period=13):
        high = self.high_csv['high']
        low = self.high_csv['low']
        close = self.high_csv['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx

    def preprocessing(self):
        self.high_csv['HMA'] = self.hma(length=5)
        self.high_csv['Laguerre Filter'] = self.laguerre_filter(alpha=0.85)
        self.high_csv['TEMA'] = self.tema(length=500)  # Calculate TEMA and add to DataFrame
        hawkes_window_long = 20
        hawkes_window_short = 20
        long_percentile = 0.1
        short_percentile = 0.1
        self.high_csv['vol_hawkes'] = self.hawkes_process(k=3)
        self.high_csv['long_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window_long,
                                                                               min_periods=1).quantile(
            long_percentile)
        self.high_csv['short_threshold'] = self.high_csv['vol_hawkes'].rolling(window=hawkes_window_short,
                                                                                min_periods=1).quantile(
            short_percentile)
        self.high_csv['adx'] = self.adx(period=12)


    def check_long_entry(self, high_pointer: int):
        if self.glob.status == 1 or high_pointer < 1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        prev_close = self.high_csv['close'].iloc[high_pointer - 3:high_pointer].mean()
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 25:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['HMA'].iloc[high_pointer] > self.high_csv['Laguerre Filter'].iloc[high_pointer]
        cond2 = self.high_csv['HMA'].iloc[high_pointer - 1] <= self.high_csv['Laguerre Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55
        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is increasing
            if tema_curr > tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.12
                self.glob.sl = 0.06
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_short_entry(self, high_pointer: int):
        if self.glob.status == -1 or high_pointer < 1:
            return 0
        close = self.high_csv['close'].iloc[high_pointer]
        prev_close = self.high_csv['close'].iloc[high_pointer - 3:high_pointer].mean()
        mean_volume_last_5_days = self.high_csv['volume'].iloc[high_pointer - 25:high_pointer].mean()
        volume = self.high_csv['volume'].iloc[high_pointer].mean()

        cond1 = self.high_csv['HMA'].iloc[high_pointer] < self.high_csv['Laguerre Filter'].iloc[high_pointer]
        cond2 = self.high_csv['HMA'].iloc[high_pointer - 1] >= self.high_csv['Laguerre Filter'].iloc[high_pointer - 1]
        cond3 = 15 < self.high_csv['adx'].iloc[high_pointer] < 55
        if cond1 and cond2 and cond3:
            tema_curr = self.high_csv['TEMA'].iloc[high_pointer]
            tema_prev = self.high_csv['TEMA'].iloc[high_pointer - 1]

            # Check if TEMA is decreasing
            if tema_curr < tema_prev:
                # Set higher take profit and stop loss
                self.glob.tp = 0.24
                self.glob.sl = 0.06
            elif volume > mean_volume_last_5_days:
                # Set lower take profit and stop loss
                self.glob.tp = 0.12
                self.glob.sl = 0.06
            else:
                return 0

            self.glob.entry_price = close
            self.glob.trailing_price = close
            return 1
        return 0

    def check_long_exit(self, high_pointer: int):
        if self.glob.status != 1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['long_threshold'].iloc[high_pointer]:
            return 1
        return 0

    def check_short_exit(self, high_pointer: int):
        if self.glob.status != -1:
            return 0
        if self.high_csv['vol_hawkes'].iloc[high_pointer] < self.high_csv['short_threshold'].iloc[high_pointer]:
            return 1
        return 0