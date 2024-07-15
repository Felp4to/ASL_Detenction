import constants as cs
import yaml

def read_yaml_fields():
    with open(cs.YAMLFILE, 'r') as ff:
        content = yaml.safe_load(ff)
        a = set(content.get('actions', []))
        c = content.get('counter', {})
        return a, c
