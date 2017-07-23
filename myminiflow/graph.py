#coding=utf-8

import logging

from . import ops


class Graph(object):
    def __init__(self):
        self._name_op_map = {}
        self._trainable_variable = {}

    def get_name_op_map(self):
        return self._name_op_map

    def get_unique_name(self, origin_name):
        unique_name = origin_name
        index = 0
        while unique_name in self._name_op_map.keys():
            unique_name = "{}_{}".format(origin_name, index)
            index += 1
        return unique_name

    def add_to_graph(self, op):
        unique_name = self.get_unique_name(op.get_name())
        op.set_name(unique_name)
        self._name_op_map[unique_name] = op

    def add_to_trainable_variable(self, key, value):
        if key in self._trainable_variable:
            logging.warn("trainable variables exists key {}.".format(key))
        else:
            self._trainable_variable[key] = value

    def get_trainable_variable(self):
        return self._trainable_variable


_default_graph = Graph()


def get_default_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = Graph()
    else:
        return _default_graph
