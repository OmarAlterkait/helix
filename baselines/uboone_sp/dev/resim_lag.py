import sys, os, numpy as np
sys.path.insert(0, 'baselines/uboone_sp/v2'); sys.path.insert(0, 'baselines/uboone_sp')
import resim as RS, h5py, hdf5plugin
from sp.data import Event, registry
jax, cfg, RK = RS.setup(); reg = registry()
E = Event(260)
step = os.path.join(RS.SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_"))
fh = h5py.File(step, "r"); g = fh[f"event_{E.event:03d}"]
for v in (0, 1):
    vg = g[f"volume_{v}"]
    org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
    pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org; q = vg["charge"][:].astype(np.float64)
    print('vol', v, 'x range', pos[:,0].min(), pos[:,0].max(), 'y', pos[:,1].min(), pos[:,1].max(), 'z', pos[:,2].min(), pos[:,2].max(), 'q', q.sum())
    for p in range(3):
        gid = 3*v+p
        C = E.clean(gid, reg)
        d, raw = RS.resim_plane(jax, cfg, RK, pos, q, v, p)
        d = d[:C.shape[0], :C.shape[1]]
        A = np.fft.rfft2(C); B = np.fft.rfft2(d)
        cc = np.fft.irfft2(A.conj()*B, s=C.shape)
        i, j = np.unravel_index(np.argmax(cc), cc.shape)
        dw = i if i < C.shape[0]//2 else i - C.shape[0]; dt = j if j < C.shape[1]//2 else j - C.shape[1]
        print(' plane', gid, 'shape', C.shape, d.shape, 'lag(resim vs stored) dw', dw, 'dt', dt, 'peak corr', cc.max()/np.linalg.norm(C)/np.linalg.norm(d),
              'stored wires', np.nonzero(C.any(1))[0][[0,-1]], 'resim wires', np.nonzero(d.any(1))[0][[0,-1]])
