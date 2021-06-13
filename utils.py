#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 29-12-2020
# --- Last Modified: Sat 24 Apr 2021 23:03:49 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Utils.
"""
import argparse
from typing import Any, List, Tuple, Union

class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.
    Borrowed from Nvidia StyleGAN2.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _str_to_list(v):
    v_values = v.strip()[1:-1]
    module_G_list = [x.strip() for x in v_values.split(',')]
    return module_G_list


def _str_to_list_of_int(v):
    v_values = v.strip()[1:-1]
    step_list = [int(x.strip()) for x in v_values.split(',')]
    return step_list


def _str_to_list_of_str(v):
    v_values = v.strip()[1:-1]
    step_list = [x.strip() for x in v_values.split(',')]
    return step_list


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')
