import eleanor
#import os
#import re
import csv

#Columba
ra = []
dec = []
with open('/home/earleyn/columba/Columba.txt', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        RA = row[1]
        DEC = row[2]
        ra.append(float(RA))
        dec.append(float(DEC))

for (r, d) in zip(ra, dec):
    try:
        star = eleanor.multi_sectors(coords=(r,d), sectors='all', tc=True)
        for s in star:
            data = eleanor.TargetData(s)
            data.save(directory='/home/earleyn/columba/lc')
    except:
        pass


'''#young_bois
tic_list = []
sector_list = []
bad = []
for filename in os.listdir('/home/earleyn/old_lc'):
    res_tic = re.findall("tic(\d+)_", filename)
    tic_list.append(int(res_tic[0]))
    res_sector = re.findall("s0(\d+)_", filename)
    sector_list.append(int(res_sector[0]))

for i, tic in enumerate(tic_list):
    try:
        star = eleanor.Source(tic=tic, sector=sector_list[i], tc=True)
        data = eleanor.TargetData(star)
        data.save(directory='/home/earleyn/lc_young_bois')
    except:
        bad.append((sector_list[i],tic))
        pass

if len(bad)>0:
    with open('/home/earleyn/young_bois_bad_tics.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['TIC', 'Sector'])
        for entry in bad:
            sector = entry[0]
            tic = entry[1]
            filewriter.writerow([tic, sector])
'''
