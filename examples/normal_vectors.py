
import numpy as np

from rcwa_tlv import RCWA, Device, Source
from rcwa_tlv.devices.shapes import add_circle

n_height = 200
n_width = 400
period_h = 450
period_w = 900
nm2pixels = n_width / period_w

# reflexion region
er1 = 1.5 ** 2
ur1 = 1.0

# transmission region
er2 = 1.0
ur2 = 1.0

layer = {
    'er': 1.0 * np.ones((n_height, n_width), dtype=np.complex64),
    'length_z': 45.0,
}

layers = [layer]
add_circle(layers[0]['er'], (n_height // 2, n_width // 4), 70 * nm2pixels, 40.0 + 1j * 6)
add_circle(layers[0]['er'], (n_height // 2, 3 * n_width // 4), 120 * nm2pixels, 40.0 + 1j * 6)

pq = np.arange(5, 20, 2)

ref = []
ref_nv = []
trn = []
trn_nv = []

for pq_ in pq:
    print(f'working on pq: {pq_}')
    device = Device(layers, period_x=period_w, period_y=period_h, er1=1.445 ** 2, ur1=1.0, er2=1.0, ur2=1.0, p=pq_, q=pq_)
    source = Source(0.0, 0.0, 1550, 1., 0.)

    rcwa = RCWA(device, source, gamma=1.0, apply_nv=False, dtype=np.float32)
    rcwa_nv = RCWA(device, source, gamma=1.0, apply_nv=True, dtype=np.float32)

    result = rcwa()
    result_nv = rcwa_nv()

    ref.append(result['tot_r'])
    trn.append(result['tot_t'])
    ref_nv.append(result_nv['tot_r'])
    trn_nv.append(result_nv['tot_t'])

    print(f'tot_r: {result["tot_r"]}, tot_t: {result["tot_t"]}')
    print(f'tot_r_nv: {result_nv["tot_r"]}, tot_t_nv: {result_nv["tot_t"]}')

n = 1
