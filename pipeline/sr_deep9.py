"""Audit run: the largest cell pushed to complexity 9 (4.5M trees). ~11 min on 8 cores."""
import numpy as np, sys, time, json
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
from symreg import enumerate_search, make_checker
from discover_dist import target
xc,y,sig,nev=target(18,11,1e-4)
chk=make_checker(np.geomspace(xc.min(),xc.max(),80),decreasing=True)
t0=time.time()
par,res=enumerate_search(xc,y,sigma=sig,max_complexity=9,max_consts=2,checker=chk,
                          n_restarts=4,verbose=True,probe=np.geomspace(xc.min(),xc.max(),17))
print('mc=9 took %.0fs admissible=%d'%(time.time()-t0,len(res)))
for p in par: print('  C=%2d rmse=%.4f  %s'%(p['complexity'],p['rmse'],p['string']))
json.dump([dict(complexity=p['complexity'],rmse=p['rmse'],string=p['string']) for p in par],
          open(_os.path.join(_HERE,'out','sr_deep9.json'),'w'),indent=1)
