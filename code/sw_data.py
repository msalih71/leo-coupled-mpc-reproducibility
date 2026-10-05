"""Space-weather inputs: converts the CelesTrak SW-All legacy .txt file bundled
with the `spaceweather` package (observed section, updated 2025-07-21) into the
CSV layout read by pymsis, and exposes 3-hourly ap / daily F10.7 series."""
import os, numpy as np, spaceweather

HERE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(os.path.dirname(spaceweather.__file__), 'data', 'SW-All.txt')
CSV = os.path.join(HERE, '..', 'data', 'SW-All_observed.csv')

def build_csv():
    rows = []
    obs = False
    for line in open(TXT):
        if line.startswith('BEGIN OBSERVED'): obs = True; continue
        if line.startswith('END OBSERVED'): break
        if not obs: continue
        f = line.split()
        y, m, d = int(f[0]), int(f[1]), int(f[2])
        kp = f[5:13]; ap = f[14:22]; apavg = f[22]
        f107adj, f107obs, ctr81obs = f[26], f[30], f[31]
        date = f'{y:04d}-{m:02d}-{d:02d}'
        rows.append(','.join([date, f[3], f[4]] + kp + [f[13]] + ap + [apavg, f[23], f[24], f[25],
                     f107obs, f107adj, 'OBS', ctr81obs, f[32], f[27], f[28]]))
    hdr = ('DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,'
           'AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,'
           'F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81')
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, 'w') as fo:
        fo.write(hdr + '\n' + '\n'.join(rows) + '\n')
    return CSV

def setup_pymsis():
    if not os.path.exists(CSV): build_csv()
    from pymsis import utils
    utils.use_space_weather_file(CSV)

def ap3h(t0, t1):
    """3-hourly ap between datetime64 t0 and t1 (inclusive of intervals starting in range)."""
    setup_pymsis()
    from pymsis import utils
    utils._load_f107_ap_data  # ensure loaded
    d = utils._load_f107_ap_data()
    return d

if __name__ == '__main__':
    print(build_csv())
    setup_pymsis()
    from pymsis import msis
    dts = np.array(['2015-03-17T12:00', '2022-02-04T00:00', '2024-05-11T00:00'], dtype='datetime64[m]')
    print(msis.get_f107_ap(dts))
