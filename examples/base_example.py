
import numpy as np

from rcwa_tlv import RCWA, Device, Source
from rcwa_tlv.devices.shapes import add_circle
from rcwa_tlv.materials import Si

lam0 = 400
n_height = 400
n_width = 400
period = 900
nm2pixels = n_width / period

p, q = 11, 11

si = Si()
erd = si(lam0)

# reflexion region
er1 = 1.0
ur1 = 1.0

# transmission region
er2 = 1.0
ur2 = 1.0

layer = {
    'er': er1 * np.ones((n_height, n_width), dtype=np.complex64),
    'length_z': 290,
}

layers = [layer]
add_circle(layers[0]['er'], (n_height // 2, n_width // 2), 230 * nm2pixels, erd)

device = Device(layers, period_x=period, period_y=period, er1=erd, ur1=ur1, er2=er2, ur2=ur2, p=p, q=q)
source = Source(0.0, 0.0, lam0, 1., 0.)

rcwa = RCWA(device, source, gamma=1.0, apply_nv=False, dtype=np.float32, renormalize=False)

result = rcwa()

print(result['tot_r'], result['tot_t'])
