import glob
import sys

directory = sys.argv[1] # for example: /Users/nicholasearley/TESS_data/columba/figures/

def repeat(tics):
    length = len(tics)
    repeated_tics = []

    for i in range(length):
        k = i + 1
        for j in range(k, length):
            if tics[i] == tics[j] and tics[i] not in repeated_tics:
                repeated_tics.append(tics[i])
    return repeated_tics

glob_files=glob.glob(directory+'GPmodel*')
tic_runs = []
for f in glob_files:
    fixed_fn = f.split(directory+'GPmodel_lc_tic')[1]
    tic_runs.append(fixed_fn)

tics = []
for t in tic_runs:
    new = t[:-5]
    tics.append(int(new))

multi_run = repeat(tics)
print('Number of possible planetary systems: '+str(len(multi_run)))

p = 0
for t in tics:
    if t not in multi_run:
        p+=1

p = p+len(multi_run)
print('Number of tics: '+str(p))

print(multi_run)
