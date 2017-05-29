import requests
import json
from threading import Thread

class Client:
  def __init__(self, netloc, always_sync=False):
    self.base_url = 'http://' + netloc
    self.session = requests.Session()
    self.session.headers.update({
      'content-type': 'application/json'
    })
    self.always_sync = always_sync

  def request(self, method, path, data=None):
    if data is not None:
      data = json.dumps(data)
    req = requests.Request(method, self.base_url + path, data=data)
    prepped = self.session.prepare_request(req)
    r = self.session.send(prepped)
    return r.json()
  
  def async_request(self, method, path, data=None):
    if self.always_sync:
      self.request(method, path, data)
    else:
      def do_request():
        self.request(method, path, data)
      Thread(target=do_request, daemon=False).start()

  def new_notebook(self, title):
    data = {
      'data': {
        'type': 'notebooks',
        'attributes': { 'title': title }
      }
    }
    res = self.request('post', '/api/v2/notebooks', data)
    return Notebook(self, res['data']['id'])

class Notebook:
  def __init__(self, client, id):
    self.client = client
    self.id = id

  def new_frame(self, title, bounds=None):
    data = {
      'data': {
        'type': 'frames',
        'attributes': { 'title': title },
        'relationships': {
          'notebook': { 'id': self.id }
        }
      }
    }
    if bounds is not None:
      data['data']['attributes'].update(bounds)
    res = self.client.request('post', '/api/v2/frames', data)
    return Frame(self.client, res['data']['id'])

class Frame:
  def __init__(self, client, id):
    self.client = client
    self.id = id

  def set_title(self, title):
    data = {
      'data': {
        'id': self.id,
        'type': 'frames',
        'attributes': { 'title': title },
      }
    }
    self.client.request('patch', '/api/v2/frames/' + self.id, data)

  def set_content(self, content_type, content_body):
    data = {
      'data': {
        'id': self.id,
        'type': 'frames',
        'attributes': {
          'type': content_type,
          'content': { 'body': content_body }
        },
      }
    }
    self.client.async_request('patch', '/api/v2/frames/' + self.id, data)

  def vega(self, spec):
    self.set_content('vega', spec)

  def vegalite(self, spec):
    self.set_content('vegalite', spec)

  def text(self, message):
    self.set_content('text', message)

  def html(self, html):
    self.set_content('html', html)

  def progress(self, current_value, max_value):
    percentage = min(100 * current_value / max_value, 100)
    html = """<div class="progress">
      <div class="progress-bar" role="progressbar"
       aria-valuenow="{percentage:0.2f}" aria-valuemin="0" aria-valuemax="100"
       style="width: {percentage:0.2f}%; min-width: 40px;"
      >
        {percentage:0.2f}%
      </div>
    </div>""".format(percentage = percentage)
    self.html(html)

  def line_graph(self, xss, yss, series_names=None, x_title=None, y_title=None,
                 y_axis_min=None, y_axis_max=None):
    if not isinstance(xss[0], list):
      xss = [xss] * len(yss)

    show_legend = True
    if series_names is None:
      show_legend = False
      series_names = ['series_{:03d}'.format(i) for i in range(len(xss))]

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    tables = []
    marks = []
    for i, xs in enumerate(xss):
      marks.append({
        'type': 'line',
        'from': { 'data': 'table_{:03d}'.format(i) },
        'properties': {
          'enter': {
            'x': { 'scale': 'x', 'field': 'x' },
            'y': { 'scale': 'y', 'field': 'y' },
            'stroke': { 'scale': 'c', 'value': series_names[i] }
          }
        }
      })

      points = []
      for j, x in enumerate(xs):
        y = yss[i][j]
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
        points.append({ 'x': x, 'y': y })
      tables.append(points)

    data = []
    for i, table in enumerate(tables):
      data.append({
        'name': 'table_{:03d}'.format(i),
        'values': table
      })

    spec = {
      'width': 370,
      'height': 250,
      'data': data,
      'scales': [
        {
          'name': 'x',
          'type': 'linear',
          'range': 'width',
          'domainMin': min_x,
          'domainMax': max_x,
          'nice': True,
          'zero': False
        }, {
          'name': 'y',
          'type': 'linear',
          'range': 'height',
          'domainMin': y_axis_min or min_y,
          'domainMax': y_axis_max or max_y,
          'nice': True,
          'zero': False
        }, {
          'name': 'c',
          'type': 'ordinal',
          'range': 'category10',
          'domain': series_names
        }
      ],
      'axes': [
        { 'type': 'x', 'scale': 'x', 'title': x_title },
        {'type': 'y', 'scale': 'y', 'title': y_title, 'grid': True}
      ],
      'marks': marks
    }

    if show_legend:
      spec['legends'] = [{ 'fill': 'c' }]

    self.vega(spec)
