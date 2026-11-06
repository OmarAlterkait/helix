import numpy as np, glob, json, sys
sys.path.insert(0,'/global/u1/o/oalter/nu/helix')
from helix.probe.resolution import unkey
M='/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests'
Q0=2043.2578108909702
def load(d):
    Z=[dict(np.load(f)) for f in sorted(glob.glob(d+'/*.npz'))]
    return {k:np.concatenate([z[k] for z in Z]) for k in Z[0]}
A=load(M+'/t3b_test'); B=load(M+'/t2t3_test')
assert (A['keys']==B['keys']).all()
gg,_,_=unkey(A['keys']); p=gg%3; T=A['Tq']; yT=np.log1p(T/Q0); cov=A['covered']
def hw(x):
    a,b=np.percentile(x,[16,84]); return (b-a)/2
models={'sp_dec_c25':A['y_sp_dec_c25_base'],'sp_dec_cond_c25':A['y_sp_dec_cond_c25_base'],'lb_dec_ct':A['y_lb_dec_ct_base'],
        'M6d':A['y_M6d_base'],'M6dc':A['y_M6dc_base'],'M7d':B['y_M7d_base'],'M7dc':B['y_M7dc_base']}
out={}
for pl,nm in enumerate('UVY'):
    print(nm)
    for lo,hi in [(4e3,16e3),(16e3,64e3),(64e3,256e3),(256e3,np.inf)]:
        m=(p==pl)&(T>=lo)&(T<hi)&cov
        r={k:float(hw(v[m]-yT[m])) for k,v in models.items()}; out[f'{nm}_{lo:.0f}']=r
        print(f'  {lo/1e3:>4.0f}k+ n={m.sum():6d} '+'  '.join(f'{k}={x:.4f}' for k,x in r.items()))
json.dump(out,open(M+'/t3b_summary.json','w'),indent=1)
S=json.load(open(M+'/t2t3_summary.json'))['test']['T1g']
for k in ['Y_64000_wire','Y_64000_A4','Y_64000_D3','Y_64000_D2','U_64000_wire','U_64000_A4','V_64000_wire','Y_16000_A4','Y_16000_wire']:
    print(k, {i:(round(v.get('hw68',0),4),round(v.get('med',0),4),v['n']) for i,v in S[k].items()})
