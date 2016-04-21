#encoding: utf-8
import sys
import random
import numpy as np

argvs = sys.argv
argc = len(argvs)
if argc != 2:
    print 'Usage: python %s <input filepath>' % argvs[0]
    sys.exit()

input_filepath = argvs[1]

lines = np.loadtxt(input_filepath, str)

random.shuffle(lines)

with open('101Caltech_shuffles.txt', 'w') as f:
    for line in lines:
        print line
        f.write(line)
        f.write("\n")
