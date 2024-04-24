
import numpy as np


class Source:
    def __init__(self, theta: float, phi: float, lam0: float, pte: float, ptm: float):
        self.theta = theta
        self.phi = phi
        self.lam0 = lam0

        if abs(theta) < np.finfo(float).eps:
            self.ate = np.array([0., 1., 0.])
        else:
            self.ate = np.array([-np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), 0])
            self.ate /= np.linalg.norm(self.ate)

        self.atm = np.cross(self.ate, np.array([0., 0., 1.]))
        self.atm /= np.linalg.norm(self.atm)
        self.polarization_vec = pte * self.ate + ptm * self.atm
        self.polarization_vec /= np.linalg.norm(self.polarization_vec)
