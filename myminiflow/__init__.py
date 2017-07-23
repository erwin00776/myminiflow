from . import session
from . import graph
from . import ops
from . import train

int32 = int
float32 = float
float64 = float

Graph = graph.Graph

Session = session.Session
constant = ops.ConstantOp
placeholder = ops.PlaceholderOp
Variable = ops.VariableOp
add = ops.AddOp
minus = ops.MinusOp
multiple = ops.MultipleOp
power = ops.PowerOp
square = ops.SquareOp
global_variables_initializer = ops.GlobalVariablesInitializerOp
local_variables_initializer = ops.LocalVariablesInitializerOp
get_variable = ops.get_variable

"""
divide = ops.DivideOp

"""