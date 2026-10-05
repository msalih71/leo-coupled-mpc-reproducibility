"""Paired 90-minute block bootstrap for out-of-sample density errors.
Blocks are non-overlapping 18 points at five-minute cadence; 10,000 draws.
The resulting CIs describe sampling variability within these short events only.
"""
import numpy as np,pandas as pd
from pathlib import Path
rng=np.random.default_rng(20260924)
rows=[]
for product,folder,pattern in [('POD',Path('observed_results'),'*_aligned_5min_JB.csv'),('ACC',Path('observed_acc_results'),'*_ACC_aligned_5min_JB.csv')]:
 for file in sorted(folder.glob(pattern)):
  df=pd.read_csv(file); n=len(df); obs=df.density.to_numpy(float)
  logs={name:np.log(df[col].to_numpy(float)/obs) for name,col in [('TIM','TIM_kg_m3'),('MSIS','MSIS00_kg_m3'),('JB','JB2008_kg_m3')]}
  blocks=[]
  for j in range(0,n,18):
   a=slice(j,min(j+18,n)); blocks.append((min(18,n-j),*[float(np.mean(v[a]**2)) for v in logs.values()]))
  block=np.asarray(blocks); sel=rng.integers(0,len(block),(10000,len(block)))
  sampled=block[sel];means=(sampled[:,:,1:]*sampled[:,:,0,None]).sum(axis=1)/sampled[:,:,0].sum(axis=1)[:,None]
  rms=np.sqrt(means)
  point={name:float(np.sqrt(np.mean(v**2))) for name,v in logs.items()}
  row={'product':product,'event':df.event.iloc[0],'satellite':df.satellite.iloc[0],'n':n,'n_90min_blocks':len(block)}
  for k,(a,b) in {'JB_minus_TIM':('JB','TIM'),'JB_minus_MSIS':('JB','MSIS'),'TIM_minus_MSIS':('TIM','MSIS')}.items():
   i=['TIM','MSIS','JB'].index(a);j=['TIM','MSIS','JB'].index(b)
   dif=rms[:,i]-rms[:,j]
   row[k]=point[a]-point[b]
   row[k+'_ci025'],row[k+'_ci975']=map(float,np.quantile(dif,[.025,.975]))
  rows.append(row)
pd.DataFrame(rows).to_csv('observed_block_bootstrap.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda v:f'{v:.3f}'))
