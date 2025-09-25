
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP

T0 = 273.15
p0 = 101325
Air_state = AbstractState("HEOS", "Air")
Air_state.update(CP.PT_INPUTS, p0, T0)
h0 = Air_state.hmass()
s0 = Air_state.smass()


def get_state(input_type, input1, input2):
    state = AbstractState("HEOS", "CO2")
    state.update(input_type, input1, input2)
    return state