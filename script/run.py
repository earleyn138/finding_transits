import os
import sys
import re
import glob
import argparse

def get_tics(lc_dir):
    '''
    '''
    def repeat(tics):
        ''' Finds multi-sector data: same tics, different sector
        '''
        length = len(tics)
        repeated_tics = []

        for i in range(length):
            k = i + 1
            for j in range(k, length):
                if tics[i] == tics[j] and tics[i] not in repeated_tics:
                    repeated_tics.append(tics[i])
        return repeated_tics

    tic_list = []
    for filename in os.listdir(lc_dir):
        res_tic = re.findall("_tic(\d+)_", filename)
        tic = int(res_tic[0])
        tic_list.append(tic)

    multi_tic = repeat(tic_list)
    single_tic = []
    for i, tic in enumerate(tic_list):
        if (tic not in multi_tic) == True:
            single_tic.append(tic)

    all_tics = []
    for t in single_tic:
        all_tics.append(t)
    for t in multi_tic:
        all_tics.append(t)

    return all_tics

path = sys.argv[1]
lc_type = sys.argv[2] #do_corr, do_psf, or do_raw
dtype = sys.argv[3] #use_files or use_tics
lc_dir = path+'lc'
all_tics = get_tics(lc_dir)

for tic in all_tics:
    os.system('python short_run.py '+str(tic)+' '+path+' '+' run_all '+lc_type+' '+dtype)
