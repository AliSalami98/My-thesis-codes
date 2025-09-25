
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP

T0 = 273
p0 = 101325
state = AbstractState("HEOS", "Air")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()


def get_state(input_type, input1, input2):
    state = AbstractState("HEOS", "CO2")
    state.update(input_type, input1, input2)
    return state