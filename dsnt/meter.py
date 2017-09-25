from tele.meter import Meter
import numpy as np
from scipy.stats import norm


class MaxValueMeter(Meter):
    def __init__(self, skip_reset=False):
        super().__init__(skip_reset)

    def add(self, new_value):
        if self._value is not None and new_value <= self._value:
            return False

        self._value = new_value
        return True

    def reset(self):
        self._value = None

    def value(self):
        return self._value


class SumMeter(Meter):
    def __init__(self, initial_value=0, skip_reset=False):
        self.initial_value = initial_value
        self._value = None
        super().__init__(skip_reset)

    def add(self, value):
        self._value += value

    def value(self):
        return self._value

    def reset(self):
        self._value = self.initial_value


class MedianValueMeter(Meter):
    def __init__(self, skip_reset=False):
        self.values = []
        super().__init__(skip_reset)

    def add(self, value):
        self.values.append(value)

    def value(self):
        data = np.asarray(self.values)

        median = np.median(data)

        k = norm.ppf(0.75) # Assume data is normally distributed
        mad = np.median(np.fabs(data - median) / k)

        return median, mad

    def reset(self):
        self.values.clear()
