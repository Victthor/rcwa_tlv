
import numpy as np

from rcwa_tlv import RCWA, Device, Source
from rcwa_tlv.devices.shapes import add_circle
from rcwa_tlv.materials import Si, SiO2

lam0 = np.arange(400, 1550, 5)
n_height = 400
n_width = 400
period_x = 900
period_y = 900
nm2pixels = n_width / period_x

p, q = 9, 9

si = Si()
sio2 = SiO2()

# reflexion region
er1 = 1.0
ur1 = 1.0

# transmission region
er2 = 1.0
ur2 = 1.0

for lam0_ in lam0:

    layer = {
        'er': er1 * np.ones((n_height, n_width), dtype=np.complex64),
        'length_z': 290,
    }

    erd = si(lam0_)

    layers = [layer]
    add_circle(layers[0]['er'], (n_height // 2, n_width // 2), 230 * nm2pixels, erd)

    device = Device(layers, period_x=period_x, period_y=period_y, er1=erd, ur1=ur1, er2=er2, ur2=ur2, p=p, q=q)
    source = Source(0.0, 0.0, lam0_, 1., 0.)

    rcwa = RCWA(device, source, gamma=1.0, apply_nv=False, dtype=np.float32, renormalize=False)

    result = rcwa()

    print(f'lam0: {lam0_}; r: {result["tot_r"]}; t: {result["tot_t"]}')

b = 1