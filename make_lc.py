import eleanor
import os
import re

tic_list = []
sector_list = []
for filename in os.listdir('/home/earleyn/young_bois'):
    res_tic = re.findall("tic(\d+)_", filename)
    tic_list.append(int(res_tic[0]))
    res_sector = re.findall("s0(\d+)_", filename)
    sector_list.append(int(res_sector[0]))

for i, tic in enumerate(tic_list):
    star = eleanor.Source(tic=tic, sector=sector_list[i], tc=True)
    data = eleanor.TargetData(star)
    data.save(directory='/home/earleyn/lc_young_bois')
