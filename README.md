
# RCWA (Rigorous Coupled Wave Analysis)

<img src=docs/images/Gemini_Generated_Image_z1rgs1z1rgs1z1rg.jpeg width=50% height=50%>

## Installation  

Regular user:  
* `pip install rcwa_tlv`  

Developer:  
* clone repository:  
  `git clone https://github.com/Victthor/rcwa_tlv.git`
* install locally:  
  `pip install -e .`

## Usage  
[base_example.py](examples%2Fbase_example.py)  
```python
import numpy as np

from rcwa_tlv import RCWA, Device, Source
from rcwa_tlv.devices.shapes import add_circle

n_height = 400
n_width = 400
period = 900
nm2pixels = n_width / period

# reflexion region
er1 = 1.5 ** 2
ur1 = 1.0

# transmission region
er2 = 1.0
ur2 = 1.0

layer = {
    'er': 1.0 * np.ones((n_height, n_width)),
    'length_z': 21,
}

layers = [layer]
add_circle(layers[0]['er'], (n_height // 2, n_width // 2), 300 * nm2pixels, er1)

device = Device(layers, period_x=period, period_y=period, er1=1.445 ** 2, ur1=1.0, er2=1.0, ur2=1.0, p=11, q=11)
source = Source(0.0, 0.0, 700, 1., 0.)

rcwa = RCWA(device, source, gamma=0.7, apply_nv=False, dtype=np.float32)

result = rcwa()

```

## References:
https://doi.org/10.1364/JOSA.71.000811
