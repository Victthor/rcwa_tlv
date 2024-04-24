
class Device:
    def __init__(self, layers: list, period_x: float, period_y: float, er1, ur1, er2, ur2, p: int, q: int):
        self.layers = layers
        self.period_x = period_x
        self.period_y = period_y
        self.er1, self.ur1, self.er2, self.ur2, self.p, self.q = er1, ur1, er2, ur2, p, q
