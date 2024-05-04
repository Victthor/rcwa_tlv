

import numpy as np

from rcwa_tlv import RCWA, Device, Source
from rcwa_tlv.devices.shapes import add_circle
from rcwa_tlv.materials import Si, SiO2

lam0 = 1550  # np.arange(400, 1550, 5)

n_disks = 3
n_height = 400
n_width = n_disks * n_height

period_x = n_disks * 500
period_y = 500
nm2pixels = n_width / period_x

init_radius = 50
radius_incr = 60

q = 11
p = np.arange(5, 27, 2)

si = Si()
sio2 = SiO2()

# reflexion region
er1 = 1.0
ur1 = 1.0

# transmission region
er2 = 1.0
ur2 = 1.0

# build device
layer = {
    'er': er1 * np.ones((n_height, n_width), dtype=np.complex64),
    'length_z': 290,
}

erd = si(lam0)

for i_disk in range(n_disks):
    add_circle(
        layer['er'],
        (n_height // 2, (1 + i_disk * 2) * n_width // (2 * n_disks)),
        (init_radius + radius_incr * i_disk) * nm2pixels,
        erd
    )

for p_ in p:
    device = Device([layer], period_x=period_x, period_y=period_y, er1=erd, ur1=ur1, er2=er2, ur2=ur2, p=p_, q=q)

    source = Source(0.0, 0.0, lam0, 1., 0.)

    rcwa = RCWA(device, source, gamma=0.5, apply_nv=True, dtype=np.float32, renormalize=False)

    result_nv = rcwa(apply_nv=True)
    result = rcwa(apply_nv=False)

    print(f'lam0: {lam0}; r   : {result["tot_r"]}; t   : {result["tot_t"]}')
    print(f'lam0: {lam0}; r nv: {result_nv["tot_r"]}; t nv: {result_nv["tot_t"]}')

b = 1
