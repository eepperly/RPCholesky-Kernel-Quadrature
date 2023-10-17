#!/usr/bin/env python

import scipy.io
import ase.io
from dscribe.descriptors import MBTR
import fileinput
import numpy as np
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def fix_file(directory = "molecules/"):
    os.chdir(directory)
    file_number = 1
    for line in fileinput.input(files = sorted(os.listdir(".")), inplace=True):
        if fileinput.isfirstline():
            eprint("File", file_number)
            file_number += 1
            numatoms = int(line.rstrip())
            counter = 0
            print(line.rstrip())
        else:
            if counter <= numatoms:
                print(line.rstrip())
                counter += 1
            else:
                fileinput.nextfile()
    
def get_molecules(directory = "molecules/", max_atoms = 29, max_mols = np.Inf):
    compounds = []
    data = []
    for f in sorted(os.listdir("molecules/")):
        print(f)
        if len(compounds) >= max_mols:
            break

        try:
            mol = ase.io.read("molecules/"+f)
            with open("molecules/"+f) as myfile:
                line = list(myfile.readlines())[1]
                data.append(list(map(float, line.split()[2:17])))
            compounds.append(mol)
        except ase.io.extxyz.XYZError:
            fix_file(directory)
            return get_molecules(directory = directory, max_atoms = max_atoms, max_mols = max_mols)
        except ValueError:
            pass

    mbtr = MBTR(species=["C","H","O","N","F"],normalization="l2",geometry={"function": "inverse_distance"},grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1})
    c = list(zip(compounds, data))
    np.random.shuffle(c)
    compounds, data = zip(*c)

    X = np.array([mbtr.create(mol) for mol in compounds])
    Y = np.array(data).reshape((X.shape[0],15))

    return X, Y 

X, Y = get_molecules(max_mols=2e4)
scipy.io.savemat("../../matlab/tests/qm9.mat", {"X" : X, "Y":Y})
