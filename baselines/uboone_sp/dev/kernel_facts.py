import sys, numpy as np
sys.path.insert(0, "baselines/uboone_sp")
from sp.response import load_kernel
for pl in "UVY":
    for src in ("fit", "example"):
        R, wo, tzb = load_kernel(pl, src)
        c = R[:, int(np.nonzero(wo == 0)[0][0])]
        t = np.arange(len(c)) - tzb
        ipk, imn = int(np.argmax(c)), int(np.argmin(c))
        s = f"{pl} {src:7s}: central wire peak {1000*c.max():.2f} ADC/1000e at dt={t[ipk]:+d}, min {1000*c.min():.2f} at dt={t[imn]:+d}"
        if pl != "Y":
            zc = ipk + int(np.argmax(c[ipk:] < 0)); s += f", zero crossing dt={t[zc]:+d}"
        s += f"; sum over kernel {R.sum():+.5f} ADC/e (x1000 = {1000*R.sum():+.2f} ADC*tick per 1000 e-); central-wire sum {1000*c.sum():+.2f}"
        s += f"; |neighbour wire +-1| peak / central {np.abs(R[:, int(np.nonzero(wo==1)[0][0])]).max()/np.abs(c).max():.2f}"
        print(s)
