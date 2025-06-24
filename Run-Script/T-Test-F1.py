from scipy.stats import ttest_ind

# BCIIV2a
# F1
logit = [0.59412307, 0.56561918, 0.57143049, 0.55897005, 0.5636052]
delta = [0.48915753, 0.47025744, 0.45637199, 0.48588527, 0.4739633]
ss = [0.5112782,  0.5,        0.47940075, 0.4797048,  0.39506173]
imd = [0.33663366, 0.37362637, 0.41747573, 0.44444444, 0.35087719]
merlin = [0, 0, 0, 0, 0]
for benchmark in [delta, ss, imd, merlin]:
    print(ttest_ind(logit, benchmark))

# #R
logit = [29, 26, 31, 32, 26]
delta = [35, 44, 34,39, 36]
ss = [1796, 2089, 1772, 2593, 1784]
imd = [30, 21, 25, 29, 17]
merlin = [80, 123, 179, 227, 279]
for benchmark in [delta, ss, imd, merlin]:
    print(ttest_ind(logit, benchmark))
