import os
import re


pkls = os.listdir('.')
pkls = [x for x in pkls if x.endswith('.json')]

bad = ['tl100', 'sc1']
for x in pkls:
    xx = re.split('[._]', x)
    for b in bad:
        if b in xx: xx.remove(b)
    os.rename(x, '_'.join(xx[:-1]) + '.json')
    #os.rename(x, )