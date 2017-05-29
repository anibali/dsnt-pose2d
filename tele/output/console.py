import tele, tele.meter
from collections import OrderedDict
import torchnet.meter

class TextCell(tele.DisplayCell):
  def __init__(self):
    super().__init__()

  def render(self, step_num, meters):
    meter_name, meter = next(iter(meters.items()))
    value = meter.value()
    if isinstance(meter, torchnet.meter.AverageValueMeter):
      (mean, std) = value
      value_str = u'{:0.4f}\u00b1{:0.4f}'.format(mean, std)
    elif isinstance(value, float):
      value_str = '{:0.4f}'.format(value)
    else:
      value_str = str(value)
    return '='.join([meter_name, value_str])

class ConsoleOutput(tele.TelemetryOutput):
  auto_cell_types = {
    torchnet.meter.AverageValueMeter: lambda name, meter: TextCell(),
    torchnet.meter.TimeMeter: lambda name, meter: TextCell(),
  }

  def __init__(self):
    super().__init__()

  def render_all(self, step_num, meters):
    render_results = super().render_all(step_num, meters)
    print('[{:4d}] '.format(step_num) + ', '.join(render_results))
    return render_results
