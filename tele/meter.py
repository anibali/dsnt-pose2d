from io import StringIO

class Meter():
  def __init__(self, skip_reset):
    self.reset()
    self.skip_reset = skip_reset

  def reset(self):
    pass

  def value(self):
    return None

class ValueMeter(Meter):
  def __init__(self, skip_reset=False):
    super().__init__(skip_reset)

  def set_value(self, value):
    self._value = value

  def reset(self):
    self._value = None

  def value(self):
    return self._value

class StringBuilderMeter(Meter):
  def __init__(self):
    self._value = StringIO()

  def add(self, part):
    self._value.write(part)

  def add_line(self, line):
    self.add(line)
    self.add('\n')

  def reset(self):
    self._value = StringIO()

  def value(self):
    return self._value.getvalue()
