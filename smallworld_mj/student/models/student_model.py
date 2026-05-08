"""Student model scaffold.

TODO for students: experiment with this parameter-conditioned residual GRU, or
replace it with a Transformer/RSSM/ODE-style dynamics model that preserves the
same `step(...)` interface.
"""

from smallworld_mj.solution.model import ParamResidualGRU as StudentWorldModel
