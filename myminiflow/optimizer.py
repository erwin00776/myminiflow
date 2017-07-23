from . import graph
from . import ops


class OptimizerMinimizeOp(ops.Op):
    def __init__(self, optimizer, loss, name="OptimizerMinimize"):
        super(OptimizerMinimizeOp, self).__init__(name)
        self._optimizer = optimizer
        self._loss = loss

        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        variablename_grad_name = self._optimizer.compute_gradients(self._loss)
        self._optimizer.apply_gradients(variablename_grad_name)

    def grad(self):
        raise NotImplementedError


class Optimizer(object):
    def __init__(self, name="Optimizer"):
        self.name = name

    def minimize(self, loss):
        pass

    def compute_gradients(self, loss):
        pass

    def apply_gradients(self, variablename_grad_map):
        pass

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class GradientDescentOptimzer(Optimizer):
    def __init__(self, learning_rate=0.01, name="GradientDescent"):
        super(GradientDescentOptimzer, self).__init__(name)
        self._learning_rate = learning_rate
        self._graph = graph.get_default_graph()

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def get_graph(self, graph):
        self._graph = graph

    def minimize(self, loss, global_step=None):
        return OptimizerMinimizeOp(self, loss)

    def compute_gradients(self, loss):
        variablename_variable_map = self._graph.get_trainable_variable()
        variablename_grad_map = {}
        for variable_name, variable in variablename_variable_map.items():
            grad = loss.grad(variable_name)
            variablename_grad_map[variable_name] = grad
        return variablename_grad_map

    def apply_gradients(self, variablename_grad_map):
        variablename_variable_map = self._graph.get_trainable_variable()
        for variablename, variable in variablename_variable_map.items():
            grad = variablename_grad_map[variablename]
            final_grad = self._learning_rate * grad
            variable.set_value(variable.get_value() - final_grad)