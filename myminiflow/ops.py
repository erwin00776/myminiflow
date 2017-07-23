#coding=utf-8

import logging
from abc import abstractmethod

import graph


class Op(object):
    def __init__(self, name="Op"):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError

    def __add__(self, other):
        return AddOp(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return MultipleOp(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class ConstantOp(Op):
    def __init__(self, value, name="ConstantOp"):
        super(ConstantOp, self).__init__(name)
        self._value = value
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def get_value(self):
        return self._value

    def forward(self):
        return self._value

    def grad(self, partial_derivative_opname=None):
        raise 0


class PlaceholderOp(Op):
    def __init__(self, dtype=None, shape=None, name="PlaceholderOp"):
        super(PlaceholderOp, self).__init__(name)
        # TODO
        self._dtype = dtype
        self._shape = shape
        self._value = None
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def set_value(self, value):
        logging.debug("placeholder {} set value: {}".format(self.name, value))
        self._value = value

    def get_value(self):
        return self._value

    def forward(self):
        return self._value

    def grad(self, partial_derivative_opname=None):
        return 0


class VariableOp(Op):
    def __init__(self, value, is_trainable, name="VariableOp"):
        super(VariableOp, self).__init__(name)
        self._value = value
        self._is_trainable = is_trainable
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)
        if self.is_trainable:
            self._graph.add_to_trainable_variable(self.get_name(), self)

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def forward(self):
        return self._value

    def grad(self, partial_derivative_opname=None):
        if partial_derivative_opname is None:
            grad = 1
        else:
            if self._name == partial_derivative_opname:
                grad = 1
            else:
                grad = 0
        return grad


class AddOp(Op):
    def __init__(self, input1, input2, name="AddOp"):
        super(AddOp, self).__init__(name)
        if not isinstance(input1, Op):
            self._op1 = ConstantOp(input1)
        else:
            self._op1 = input1
        if not isinstance(input2, Op):
            self._op2 = ConstantOp(input2)
        else:
            self._op2 = input2
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        result = self._op1.forward() + self._op2.forward()
        return result

    def grad(self, partial_derivative_opname=None):
        result = self._op1.grad(partial_derivative_opname) + self._op2.grad(partial_derivative_opname)
        return result


class MultipleOp(Op):
    def __init__(self, input1, input2, name="MultipleOp"):
        super(MultipleOp, self).__init__(self)
        if not isinstance(input1, Op):
            self._op1 = ConstantOp(input1)
        else:
            self._op1 = input1
        if not isinstance(input2, Op):
            self._op2 = ConstantOp(input2)
        else:
            self._op2 = input2
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        result = self._op1.forward() * self._op2.forward()
        return result

    def grad(self):
        result = self._op1.grad() * self._op2.forward() + \
                    self._op1.forward() * self._op2.grad()
        return result



