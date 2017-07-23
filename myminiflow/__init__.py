from . import session
from . import graph
from . import ops

int32 = int
float32 = float
float64 = float

Graph = graph.Graph

Session = session.Session
constant = ops.ConstantOp
placeholder = ops.PlaceholderOp
add = ops.AddOp
multiple = ops.MultipleOp


"""
Variable = ops.VariableOp
placeholder = ops.PlaceholderOp
minus = ops.MinusOp
multiple = ops.MultipleOp
divide = ops.DivideOp
square = ops.SquareOp
global_variables_initializer = ops.GlobalVariablesInitializerOp
local_variables_initializer = ops.LocalVariablesInitializerOp
get_variable = ops.get_variable
"""