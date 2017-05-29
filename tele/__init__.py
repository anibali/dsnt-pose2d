from collections import OrderedDict
from io import StringIO

class Telemetry():
  def __init__(self, meters):
    self.meters = meters
    self.outputs = []
    self.step_num = 1

  def add_output(self, output):
    self.outputs.append(output)
    output.prepare(self.meters)

  def step(self):
    for output in self.outputs:
      output.render_all(self.step_num, self.meters)
    for meter_name, meter in self.meters.items():
      meter.reset()
    self.step_num += 1

  def __getitem__(self, meter_name):
    return self.meters[meter_name]

class TelemetryOutput():
  def __init__(self):
    self.cell_index = {}
    self.cell_list = []
    self.auto_cell = True

  def set_cells(self, cell_list):
    self.cell_list = cell_list
    for meter_names, cell in cell_list:
      for meter_name in meter_names:
        if meter_name in self.cell_index:
          meter_cells = self.cell_index[meter_name]
        else:
          meter_cells = []
          self.cell_index[meter_name] = meter_cells
        meter_cells.append(cell)
  
  def set_auto_default_cell(self, auto_cell):
    self.auto_cell = auto_cell

  def prepare(self, meters):
    if self.auto_cell:
      meters_without_cells = set(meters.keys()) - set(self.cell_index.keys())
      for meter_name in sorted(meters_without_cells):
        meter = meters[meter_name]
        cell = None
        if meter.__class__ in self.__class__.auto_cell_types:
          cell = self.__class__.auto_cell_types[meter.__class__](meter_name, meter)
        if cell is not None:
          self.cell_list.append(([meter_name], cell))
          self.cell_index[meter_name] = cell

  def render_all(self, step_num, meters):
    render_results = []
    for i, (meter_names, cell) in enumerate(self.cell_list):
      cell_meters = OrderedDict([(mn, meters[mn]) for mn in meter_names])
      render_results.append(cell.render(step_num, cell_meters))
    return render_results

class DisplayCell():
  def render(self, step_num, meters):
    raise 'not implemented'
