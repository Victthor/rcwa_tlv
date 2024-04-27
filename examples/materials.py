
from rcwa_tlv.materials import Si, SiO2

if __name__ == "__main__":
    si = Si()
    sio2 = SiO2()

    lam0 = 1550

    print(f'er_si = {si(lam0)}\ner_SiO2 = {sio2(lam0)}')
