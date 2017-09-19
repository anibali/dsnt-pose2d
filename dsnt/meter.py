from tele.meter import Meter


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
