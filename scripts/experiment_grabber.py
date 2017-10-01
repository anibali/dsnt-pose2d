#!/usr/bin/env python3

import os
import sys
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
    notebook_details = {}

    for notebook_id in notebook_ids:
        url = 'http://{}/api/v2/notebooks/{:d}'.format(showoff_netloc, notebook_id)

        try:
            with urllib.request.urlopen(url) as f:
                response = json.load(f)

            notebook_title = response['data']['attributes']['title']
            frames_data = response['data']['relationships']['frames']['data']

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
            }
        except urllib.error.URLError as error:
            print('Unable to fetch details for notebook {:d}\n↳ {}'.format(notebook_id, error),
                  file=sys.stderr)
        except Exception as error:
            print('Unable to parse details for notebook {:d}\n↳ {}'.format(notebook_id, error),
                  file=sys.stderr)

    return notebook_details


def generate_experiment_aliases(notebook):
    aliases = []

    base_model = notebook['args']['base_model']
    dilate = notebook['args']['dilate']
    truncate = notebook['args']['truncate']
    preact = notebook['args']['preact']
    output_strat = notebook['args']['output_strat']

    if base_model == 'resnet34' and output_strat == 'dsnt' and dilate == '2':
        aliases.append('preact-{}-d{}-t{}'.format(preact, dilate, truncate))

    if base_model =='resnet34' and preact == 'softmax':
        aliases.append('outstrat-{}-d{}-t{}'.format(output_strat, dilate, truncate))

    if output_strat == 'dsnt' and preact == 'softmax':
        aliases.append('depth-{}-d{}-t{}'.format(base_model, dilate, truncate))

    return aliases


def main():
    notebook_ids = list(range(365, 395 + 1))
    showoff_netloc = 'anibali-ltu.duckdns.org:16676'

    notebooks = fetch_details(showoff_netloc, notebook_ids)

    for notebook_id, notebook in notebooks.items():
        print('# {} (notebook_id={:d})'.format(notebook['experiment_id'], notebook_id))
        if notebook['epoch'] == 119:
            src = '{}:/home/aiden/commie/home/aiden/Projects/PyTorch/dsnt/out/{}'.format(
                notebook['host'], notebook['experiment_id'])
            if not os.path.exists('out/{}'.format(notebook['experiment_id'])):
                call(['scp', '-r', src, 'out/'])
            aliases = generate_experiment_aliases(notebook)
            for alias in aliases:
                call(['ln', '-snf', '../{}'.format(notebook['experiment_id']), 'out/by-alias/{}'.format(alias)])
            with open(os.path.join('out', notebook['experiment_id'], 'notebook_details.json'), 'w') as f:
                json.dump(notebook, f)
        else:
            print('Still training')


if __name__ == '__main__':
    main()
