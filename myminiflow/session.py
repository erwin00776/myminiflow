#coding=utf-8

from . import ops


class Session(object):
    def __init__(self):
        pass

    def run(self, op, feed_dict=None, options=None):
        name_to_op = op._graph.get_name_op_map()
        if feed_dict is not None:
            for op_or_opname, value in feed_dict.items():
                if isinstance(op_or_opname, str):
                    placeholder_op = name_to_op[op_or_opname]
                else:
                    placeholder_op = op_or_opname
                if isinstance(placeholder_op, ops.PlaceholderOp):
                    placeholder_op.set_value(value)
        result = op.forward()
        return result
