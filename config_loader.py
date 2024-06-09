import yaml
import time
import os
import shutil
import constants as cs


def read_yaml_fields():
    with open(cs.YAMLFILE, 'r') as ff:
        content = yaml.safe_load(ff)
        a = set(content.get('actions', []))
        c = content.get('counter', {})
        return a, c


def update_yaml_fields(new_actions, new_counter):
    with open(cs.YAMLFILE, 'r') as ff:
        content = yaml.safe_load(ff)
    content['actions'] = new_actions
    content['counter'] = new_counter
    with open(cs.YAMLFILE, 'w') as ff:
        yaml.safe_dump(content, ff)


def reset_yaml_fields():
    current_timestamp_time = time.time()
    file_name, file_extension = os.path.splitext(cs.YAMLFILE)
    destination = ("backups/" +
                   file_name + "_" +
                   str(current_timestamp_time) +
                   file_extension)
    shutil.copy(cs.YAMLFILE, destination)
    content = {'actions': set(), 'counter': {}}
    with open(cs.YAMLFILE, 'w') as ff:
        yaml.safe_dump(content, ff)
