import copy

# CLAIX18 (Turbo - SKYLAKE xeon platinum 8160)
CPU_MHZ = {'1': 3.5, '12': 2.6, '24': 2}
CPU_FPC = 32
CPU_FPS = copy.deepcopy(CPU_MHZ)
for k, v in CPU_FPS.items():
    CPU_FPS[k] = v * CPU_FPC * 1e9 * int(k)

# Node 72 (No Turbo - HASWELL)
# CPU_MHZ = {'1': 2.5, '12': 2.5}
# CPU_FPC = 16
# CPU_FPS = CPU_MHZ
# for k, v in CPU_FPS:
#     CPU_FPS[k] = v * CPU_FPC * 1e9 * int(k)

input_path = '../../data/'
lut_output_path = '../../data/'
