"""Compare our JB2008 input wrapper against the unmodified SET driver.
Uses short local extracts of official index files to accommodate its 10000-entry arrays.
"""
from pathlib import Path
from datetime import date, timedelta
import subprocess, shutil, numpy as np, sys
sys.path.insert(0, str(Path('../code').resolve()))
from orbit_env import sun_eci, jd_from_dt64, gmst
src=Path('jb_sources'); out=Path('jb_driver_check');out.mkdir(exist_ok=True)
shutil.copyfile(src/'JB08DRV.exe',out/'JB08DRV.exe')
solar={}
for line in (src/'SOLFSMY.TXT').open():
 v=line.split()
 try:solar[date(int(v[0]),1,1)+timedelta(days=int(v[1])-1)]=[float(z) for z in v[3:11]]
 except (ValueError,IndexError):pass
for stamp in ('2015-03-17T12:00','2022-02-03T12:00','2024-05-09T12:00'):
 day=date.fromisoformat(stamp[:10]); lo=day-timedelta(days=10);hi=day+timedelta(days=3)
 for file in ('SOLFSMY.TXT','DTCFILE.TXT'):
  with (out/file).open('w') as target:
   for line in (src/file).open():
    v=line.split()
    try:d=date(int(v[1]),1,1)+timedelta(days=int(v[2])-1) if file.startswith('DTC') else date(int(v[0]),1,1)+timedelta(days=int(v[1])-1)
    except (ValueError,IndexError):continue
    if lo<=d<=hi:target.write(line)
 inp=f'{day.year:5d}{day.timetuple().tm_yday:5d}{12:5d}{0:5d}{0.:7.3f}' + ''.join(f'  {x:4d}' for x in (500,500,1)) + ''.join(f'  {x:3d}' for x in (300,300,1,10,90,-90))
 (out/'JB2008_AUTO_INPUT.DAT').write_text(inp+'\n'+'test\n'*6)
 subprocess.run(['./JB08DRV.exe'],cwd=out,capture_output=True,check=True)
 lines=(out/'JB2008_AUTO_OUTPUT.DAT').read_text().splitlines();assert len(lines)>=20
 true=float(next(line for line in lines if len(line.split())==6 and abs(float(line.split()[2]))<0.01).split()[-1].replace('D','E'))
 dtemp=float(lines[7].split()[-1]); F=solar[day-timedelta(days=1)];M=solar[day-timedelta(days=2)];Y=solar[day-timedelta(days=5)]
 j=jd_from_dt64(np.array([np.datetime64(stamp)]))[0];sun=sun_eci(np.array([j]))[0]
 args=[j-2400000.5,np.arctan2(sun[1],sun[0]),np.arcsin(sun[2]),(gmst(np.array([j]))[0]+np.deg2rad(300))%(2*np.pi),0,500,F[0],F[1],F[2],F[3],M[4],M[5],Y[6],Y[7],dtemp]
 result=subprocess.check_output([str((src/'jb_call.exe').resolve())],input=(' '.join(map(str,args))+'\n').encode())
 ours=float(result)
 print(stamp,'driver',true,'wrapper',ours,'relative_diff',ours/true-1,flush=True)
 assert abs(ours/true-1)<1e-3
