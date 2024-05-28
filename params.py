import pandas as pd
import numpy as np
from functools import reduce

_INTEGERS = [
]
_FLOAT_AS_INTEGER = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    
]
_FLOAT_ONE = [
    'FastingBS',
    'HeartDisease',
]
_FLOAT_TWO = [
    'Oldpeak',
]
_FLOAT_THREE = [  
]
_FLOAT_FOUR = [  
]
_PERCENT_ZERO = [
    '0',
    '1',
]
_PERCENT_ONE = [
    'Accuracy',
    'Balanced Accuracy',
    'F1 Score',
]
_PERCENT_TWO = [
]
_PERCENT_THREE = [

]
_PERCENT_FOUR = [
]
def create_format_dict(
    integers=_INTEGERS,
    float_as_integer=_FLOAT_AS_INTEGER,
    float_one=_FLOAT_ONE,
    float_two=_FLOAT_TWO,
    float_three=_FLOAT_THREE,
    float_four=_FLOAT_FOUR,
    percent_zero=_PERCENT_ZERO,
    percent_one=_PERCENT_ONE,
    percent_two=_PERCENT_TWO,
    percent_three=_PERCENT_THREE,
    percent_four=_PERCENT_FOUR,
):
    format_dict_placeholder = []
    # _INTEGERS
    format_dict_placeholder.append(dict.fromkeys(integers, '{:.d}'))

    # _FLOAT_AS_INTEGER
    format_dict_placeholder.append(dict.fromkeys(float_as_integer, '{:.0f}'))

    # _FLOAT_ONE
    format_dict_placeholder.append(dict.fromkeys(float_one, '{:.1f}'))

    # _FLOAT_TWO
    format_dict_placeholder.append(dict.fromkeys(float_two, '{:.2f}'))
    
    # _FLOAT_THREE
    format_dict_placeholder.append(dict.fromkeys(float_three, '{:.3f}'))
    
    # _FLOAT_FOUR
    format_dict_placeholder.append(dict.fromkeys(float_four, '{:.4f}'))

    # _PERCENT_ZERO
    format_dict_placeholder.append(dict.fromkeys(percent_zero, '{:.0%}'))

    # _PERCENT_ONE
    format_dict_placeholder.append(dict.fromkeys(percent_one, '{:.1%}'))

    # _PERCENT_TWO
    format_dict_placeholder.append(dict.fromkeys(percent_two, '{:.2%}'))

    # _PERCENT_THREE
    format_dict_placeholder.append(dict.fromkeys(percent_three, '{:.3%}'))

    # _PERCENT_FOUR
    format_dict_placeholder.append(dict.fromkeys(percent_four, '{:.4%}'))

    # MERGING THE DICTIONARIES
    format_dict = reduce(lambda a, b: dict(a, **b), format_dict_placeholder)
    return format_dict
