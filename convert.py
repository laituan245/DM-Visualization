import os
import numpy as np
import random

true_labels = []
f = open('data/reals.txt', 'r')
for line in f:
    line = line.strip()
    if len(line) > 0:
        if ':' in line: line = line.split(':')[0]
        true_labels.append(line)
f.close()
true_label_str = ';'.join(true_labels)

f = open('data/predictions.txt', 'r')
lines = f.readlines()
lines = [line.strip() for line in lines if len(line.strip()) > 0]
f.close()

labels = []
for line in lines:
    es = [float(e) for e in line[1:-1].split(',')]
    labels.append(str(int(np.argmax(es))))
label_str = ';'.join(labels)


f = open('tmp/predict_outputs.txt', 'w+')
f.write('{}\n'.format(label_str))
f.write('{}\n'.format(label_str))
for line in lines:
    es = [float(e) for e in line.strip()[1:-1].split(',')]
    f.write('{}\n'.format(es))
f.close()

f = open('tmp/true_outputs.txt', 'w+')
f.write('{}\n'.format(true_label_str))
f.write('{}\n'.format(true_label_str))
for i in range(len(lines)):
    line = lines[i]
    es = [float(e) for e in line.strip()[1:-1].split(',')]
    true_label = int(true_labels[i])
    for j in range(4):
        if j == true_label:
            es[j] = 1
        else:
            es[j] = 0
    f.write('{}\n'.format(es))
f.close()

# Run Perl scripts
os.system('./Visual2.pl data/trimmap tmp/predict_outputs.txt -p > outputs/prediction.pdb')
os.system('./Visual2.pl data/trimmap tmp/true_outputs.txt -p > outputs/native.pdb')

# Generate pml files
pdb_f = open('outputs/prediction.pdb', 'r')
pdb_f.readline()
f = open('outputs/prediction.pml', 'w+')
f.write('load prediction.pdb, prediction\nhide all\n')
for line in pdb_f:
    if line.startswith('MODEL'): break
    f.write(line)
f.close()
pdb_f.close()

pdb_f = open('outputs/native.pdb', 'r')
pdb_f.readline()
f = open('outputs/native.pml', 'w+')
f.write('load native.pdb, native\nhide all\n')
for line in pdb_f:
    if line.startswith('MODEL'): break
    f.write(line)
f.close()
pdb_f.close()
