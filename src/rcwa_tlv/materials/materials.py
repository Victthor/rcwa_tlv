
import os
import numpy as np
from scipy.interpolate import splrep, splev


class Material:
    def __init__(self, filename: str):
        self.data = np.loadtxt(filename)

        self.spl_eps_real = None
        self.spl_eps_imag = None

        self.min_lam0 = np.min(self.data[:, 0])
        self.max_lam0 = np.max(self.data[:, 0])

        self._interpolate_data()

    def _interpolate_data(self):
        wavelengths = self.data[:, 0]

        eps_real = self.data[:, 1] ** 2 - self.data[:, 2] ** 2
        eps_imag = 2 * self.data[:, 1] * self.data[:, 2]

        self.spl_eps_real = splrep(wavelengths, eps_real, s=0)
        self.spl_eps_imag = splrep(wavelengths, eps_imag, s=0)

    def __call__(self, lam0: float):

        if self.spl_eps_real is None or self.spl_eps_imag is None:
            raise RuntimeError(f'spl_eps_real or spl_eps_imag not set')

        if lam0 < self.min_lam0 or lam0 > self.max_lam0:
            raise RuntimeError(f'lam0: {lam0} is out of range: {self.min_lam0} - {self.max_lam0}')

        epsr_real = splev(lam0, self.spl_eps_real)
        epsr_imag = splev(lam0, self.spl_eps_imag)

        return epsr_real - 1j * epsr_imag


class Si(Material):
    def __init__(self, filename='Si.txt'):
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))


class SiO2(Material):
    def __init__(self, filename='SiO2.txt'):
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))


class TiO2(Material):
    def __init__(self, filename='TiO2.txt'):
        super().__init__(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))


if __name__ == '__main__':
    si = TiO2()
    er = si(20)

    v = 1
