"""Trajectory id of every stress-drop / aging event (for trajectory-blocked CV and memory tests)."""
import numpy as np, pandas as pd
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
S = _os.path.join(_HERE, 'out')
u0c=-4.60751861; N=49999
df=pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index','strain_index'])
tau=(df['stress'].to_numpy()/1e4).reshape(-1,N)
pe=(df['pe'].to_numpy()-u0c).reshape(-1,N)
del df
dtau=np.diff(tau,axis=1); du=np.diff(pe,axis=1)
drop=(dtau<0).ravel(); age=((du<0)&(dtau>=0)).ravel()
ntr=tau.shape[0]
tid_all=np.repeat(np.arange(ntr,dtype=np.int32),N-1)
np.savez_compressed(S+'/event_tids.npz', tid_drop=tid_all[drop], tid_age=tid_all[age])
print('saved', drop.sum(), age.sum())
