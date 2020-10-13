from enum import IntEnum

class MttkrpMethod(IntEnum):
    MTTKRP = 0
    TWOSTEP0 = 1
    TWOSTEP1 = 2
    AUTO = 3

def mode_string(modes):
    mode_name = ""
    for m in modes:
        mode_name += str(m) + '-'
    return mode_name[:-1]
