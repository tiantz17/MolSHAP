import pickle
import argparse
import pandas as pd
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", default="./demo/demo.csv", help="Input table in .csv format. Please refer to README for more requirements.", type=str)
parser.add_argument("--output", "-o", default="./demo/", help="Output folder.", type=str)
parser.add_argument("--core", "-c", default="", help="Scarffold for decomposition, in SMILES format.", type=str)
args = parser.parse_args()
prefix = args.input.split('/')[-1].replace('.csv', '')

# Load table
print("Side-chain decomposition for {}".format(args.input))
table = pd.read_csv(args.input)
assert 'ID' in table.columns
assert table['ID'].unique().shape[0] == table.shape[0], 'Compound ID conflicts!'
assert 'SMILES' in table.columns
assert 'Activity' in table.columns

# Get compounds
list_smiles = table['SMILES'].values
list_mol = [get_mol_from_smiles(smiles) for smiles in list_smiles]
print("{} compounds found.".format(len(list_smiles)))

# Find scarffold
if args.core == '':
    ss = rdFMCS.FindMCS(list_mol, matchValences=False, matchChiralTag=False, timeout=2, 
                        completeRingsOnly=True, ringMatchesRingOnly=True)
    core = Chem.MolFromSmarts(ss.smartsString)
else:
    core = Chem.MolFromSmiles(args.core)

# Start decomposition
# first decompose using rdkit
list_decompose, list_failed = rdRGD.RGroupDecompose(core, list_mol, asSmiles=False)
list_keys = list(list_decompose[0].keys())
first_idx = []
for i in range(len(list_mol)):
    if i not in list_failed:
        first_idx.append(i)
assert len(first_idx) == len(list_decompose)

# second removing side chains with multiple attachment atoms
second_idx = []
success_idx = []
list_res = [{r:Chem.MolToSmiles(item[r]) for r in item} for item in list_decompose]
for i, item in enumerate(list_res):
    flag = True
    for ikey in list_keys[1:]:
        for jkey in list_keys[1:]:
            if ikey not in item:
                flag = False
                break
            if ikey != jkey and "[*:" + jkey[1:] in item[ikey]:
                flag = False
                break
        if not flag:
            break
    if flag:
        second_idx.append(i)
        success_idx.append(first_idx[i])
list_success = []
for i in second_idx:
    list_success.append(list_decompose[i])
assert len(list_success) == len(success_idx)

# Results
core_ = list_decompose[0]['Core']
num_core_atoms = core_.GetNumAtoms()
num_side_chains = len(list_decompose[0]) - 1
num_success = len(list_success)
num_failure = len(list_failed)
print("Scarffold: {} atoms, {}".format(num_core_atoms, Chem.MolToSmiles(list_decompose[0]['Core'])))
print("Side chains: {}, [{}]".format(num_side_chains, ", ".join(list_keys[1:])))
print("{} compounds in total, {} succeed, {} failed.".format(len(list_mol), num_success, num_failure))

# Saving
pickle.dump([core, list_success, success_idx], open(args.output+"/{}_decomp.pk".format(prefix), 'wb'))