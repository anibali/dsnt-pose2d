#!/usr/bin/env python3

import os
import sys
import traceback
import urllib.request
import urllib.error
import json
import re
from bs4 import BeautifulSoup
from subprocess import call


def frame_to_dict(frame_title, frames_data):
    subject_frame = None
    for frame in frames_data:
        if frame['attributes']['title'] == frame_title:
            subject_frame = frame

    frame_content = subject_frame['attributes']['content']['body']
    soup = BeautifulSoup(frame_content, 'html.parser')

    content_dict = {}
    key = None
    for i, td in enumerate(soup.find_all('td')):
        if i % 2 == 0:
            key = td.text
        else:
            content_dict[key] = td.text

    return content_dict


def fetch_details(showoff_netloc, notebook_ids):
    with urllib.request.urlopen('http://{}/api/v2/tags'.format(showoff_netloc)) as f:
        raw_tags = json.load(f)

    tags = {}
    for raw_tag in raw_tags['data']:
        notebook_id = int(raw_tag['relationships']['notebook']['data']['id'])
        if not notebook_id in tags:
            tags[notebook_id] = []
        tags[notebook_id].append(raw_tag['attributes']['name'])

    notebook_details = {}

    for notebook_id in notebook_ids:
        url = 'http://{}/api/v2/notebooks/{:d}'.format(showoff_netloc, notebook_id)

        try:
            with urllib.request.urlopen(url) as f:
                response = json.load(f)

            notebook_title = response['data']['attributes']['title']
            frames_data = [x for x in response['included'] if x['type'] == 'frames']

            inspect_dict = frame_to_dict('Inspect', frames_data)
            args_dict = frame_to_dict('Command-line arguments', frames_data)
            args_dict = {re.search('^args\.(.*)$', k).group(1): v for k, v in args_dict.items()}

            host = re.search('^\[([^\]]*)\]', notebook_title).group(1)
            epoch = int(inspect_dict['epoch'])
            experiment_id = inspect_dict['experiment_id']

            notebook_details[notebook_id] = {
                'host': host,
                'epoch': epoch,
                'experiment_id': experiment_id,
                'notebook_id': notebook_id,
                'args': args_dict,
                'tags': tags[notebook_id],
            }
        except urllib.error.URLError as error:
            print('Unable to fetch details for notebook {:d}\n↳ {}'.format(
                  notebook_id, error),
                  file=sys.stderr)
        except Exception as error:
            print('Unable to parse details for notebook {:d}\n↳ {}: {}'.format(
                  notebook_id, error.__class__.__name__, error),
                  file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


    return notebook_details


def generate_experiment_aliases(notebook):
    aliases = []

    base_model = notebook['args']['base_model']
    dilate = notebook['args']['dilate']
    output_strat = notebook['args']['output_strat']
    try:
        reg = notebook['args']['reg']
    except KeyError:
        reg = 'none'
    tags = notebook['tags']

    if 'time' in tags:
        aliases.append('time-{}-d{}'.format(base_model, dilate))

    if 'out-strat' in tags:
        if output_strat == 'dsnt':
            aliases.append('outstrat-{}{}-{}-d{}'.format(output_strat, reg, base_model, dilate))
        elif output_strat == 'gauss':
            aliases.append('outstrat-{}-{}-d{}'.format(output_strat, base_model, dilate))

    return aliases


def main():
    showoff_netloc = 'anibali-ltu.duckdns.org:16676'

    with urllib.request.urlopen('http://{}/api/v2/notebooks'.format(showoff_netloc)) as f:
        raw_notebooks = json.load(f)
    notebook_ids = [int(nb['id']) for nb in raw_notebooks['data']]

    notebooks = fetch_details(showoff_netloc, notebook_ids)

    if not os.path.exists('out/by-alias'):
        os.makedirs('out/by-alias')

    for notebook_id, notebook in notebooks.items():
        print('# {} (notebook_id={:d})'.format(notebook['experiment_id'], notebook_id))
        aliases = generate_experiment_aliases(notebook)
        if len(aliases) == 0:
            continue
        if notebook['epoch'] == 119:
            src = '{}:/home/aiden/commie/home/aiden/Projects/PyTorch/dsnt/out/{}'.format(
                notebook['host'], notebook['experiment_id'])
            if not os.path.exists('out/{}'.format(notebook['experiment_id'])):
                call(['scp', '-r', src, 'out/'])
            for alias in aliases:
                call(['ln', '-snf', '../{}'.format(notebook['experiment_id']), 'out/by-alias/{}'.format(alias)])
            with open(os.path.join('out', notebook['experiment_id'], 'notebook_details.json'), 'w') as f:
                json.dump(notebook, f)
        else:
            print('Still training')


if __name__ == '__main__':
    main()
