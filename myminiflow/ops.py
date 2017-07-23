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

    def __sub__(self, other):
        return MinusOp(self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return MultipleOp(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class GlobalVariablesInitializerOp(Op):
    def __init__(self, name='GlobalVariablesInitializerOp'):
        super(GlobalVariablesInitializerOp, self).__init__(name)
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        pass

    def grad(self):
        raise NotImplementedError


class LocalVariablesInitializerOp(Op):
    def __init__(self, name='LocalVariablesInitializerOp'):
        super(LocalVariablesInitializerOp, self).__init__(name)
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        pass

    def grad(self):
        raise NotImplementedError


def get_variable(name="Variable", value=None, shape=None, dtype=None,
                 initializer=None, regularizer=None, reuse=None, trainable=None):
    _graph = graph.get_default_graph()
    if name in _graph.get_name_op_map():
        return _graph.get_name_op_map()[name]
    else:
        variable = VariableOp(value=value, name=name)
        return variable


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
    def __init__(self, value, is_trainable=True, name="VariableOp"):
        super(VariableOp, self).__init__(name)
        self._value = value
        self._is_trainable = is_trainable
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)
        if self._is_trainable:
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
            if self.name == partial_derivative_opname:
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


class MinusOp(Op):
    def __init__(self, input1, input2, name="MinusOp"):
        super(MinusOp, self).__init__(name)
        if not isinstance(input1, Op):
            self._op1 = ConstantOp(input1)
        else:
            self._op1 = input1
        if not isinstance(input2, Op):\
            self._op2 = ConstantOp(input2)
        else:
            self._op2 = input2

        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        result = self._op1.forward() - self._op2.forward()
        return result

    def grad(self, partial_derivative_opname=None):
        result = self._op1.grad(partial_derivative_opname) - self._op2.grad(partial_derivative_opname)
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

    def grad(self, partial_derivative_opname=None):
        op1_value = self._op1.forward()
        op2_value = self._op2.forward()
        if isinstance(self._op1, PlaceholderOp) or isinstance(self._op1, ConstantOp):
            op1_grad = self.op1.forward()
            if isinstance(self._op2, PlaceholderOp) or isinstance(self._op2, ConstantOp):
                op2_grad = 0
            else:
                op2_grad = self._op2.grad(partial_derivative_opname)
            result = op1_grad * op2_grad
        elif isinstance(self._op2, PlaceholderOp) or isinstance(self._op2, ConstantOp):
            op2_grad = self._op2.forward()
            op1_grad = self._op1.grad(partial_derivative_opname)
            result = op1_grad * op2_grad
        else:
            op1_grad = self._op1.grad(partial_derivative_opname)
            op2_grad = self._op2.grad(partial_derivative_opname)
            result = op1_grad * op2_value + op1_value * op2_grad
        return result


class PowerOp(Op):
    def __init__(self, input, power, name="PowerOp"):
        super(PowerOp, self).__init__(name)
        if not isinstance(input, Op):
            self._op = ConstantOp(input)
        else:
            self._op = input
        self._power = power

        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        result = pow(self._op.forward(), self._power)
        return result

    def grad(self, partial_derivative_opname=None):
        if isinstance(self._op, PlaceholderOp) or isinstance(self._op, ConstantOp):
            grad = 0
        elif isinstance(self._op, VariableOp):
            grad = self._power * pow(self._op.forward(), self._power - 1)
        else:
            grad = self._power * pow(self._op.forward(), self._power - 1) * \
                   self._op.grad(partial_derivative_opname)
        return grad


class SquareOp(PowerOp):
    def __init__(self, input, name="SquareOp"):
        super(SquareOp, self).__init__(input, 2, name)
