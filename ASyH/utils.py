import inspect
from typing import Dict, Any


def ToDo():
    print(inspect.currentframe().f_back.f_code.co_name +
          "This feature is not implemented yet.", end='\n')


def flatten_dict(indict: Dict[str, Any],
                 root_key: str = '',
                 key_separator: str = '.'):
    items = []
    for key, val in indict.items():
        context = root_key + key_separator + key if root_key else key
        if isinstance(val, dict):
            items.extend(
                flatten_dict(val, context, key_separator=key_separator).items())
        else:
            items.append((context, val))
    return dict(items)
