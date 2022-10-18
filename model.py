from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from utils import *

def score_cls(label, pred, threshold=2):
    label = np.array(label).reshape(-1)
    pred = np.array(pred).reshape(-1)
    
    label3 = list(label) + [-item for item in label]
    pred3 = list(pred) + [-item for item in pred]
    t = np.log10(threshold)
    pred3 = [1 if item>t else -1 if item<-t else 0 for item in pred3]
    results3 = classification_report(label3, pred3, output_dict=True)
    acc3 = results3['accuracy']
    macrof1 = results3['macro avg']['f1-score']

    return { 
            "MacroF1": macrof1,
            "Accuracy": acc3,
           }

class MolSHAP:
    def __init__(self, method='GP'):
        self.method = method
        if method == 'SVR':
            self.model = SVR()
        elif method == 'RF':
            self.model = RandomForestRegressor(n_estimators=1000)
        elif method == 'GBT':
            self.model = GradientBoostingRegressor(n_estimators=1000)
        elif method == 'GP':
            self.model = GaussianProcessRegressor(kernel=DotProduct())
        else:
            raise NotImplementedError
        
    def train(self, X, y):
        self.model = self.model.fit(X, y)
        
    def train_onehot(self, ori_idx, y):
        for i, j in zip(*np.where(ori_idx == 0)):
            ori_idx[i, j] = self.dict_empty[self.list_keys[j]]
        list_mol = [makebond(item)[0] for item in self.list_frag_mol[ori_idx]]        
        list_newmol = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in list_mol]
        list_newmol = [Chem.RemoveHs(mol) for mol in list_newmol]
        X = np.array(get_fps(list_newmol))
        self.model = self.model.fit(X, y)
        
    def predict(self, X, return_std=False):
        if return_std:
            y, std = self.model.predict(X, return_std=True)
            return y, std
        else:
            y = self.model.predict(X)
            return y
    
    def init(self, list_keys, list_frag, list_frag_mol, frag2idx):
        self.list_keys = list_keys
        self.list_frag = list_frag
        self.list_frag_mol = list_frag_mol
        self.frag2idx = frag2idx

    def preload(self, ori_idx):
        fixed = []
        variant = []
        dict_empty = {}
        for i in range(len(self.list_keys)):
            candidate = np.unique(ori_idx[:,i])
            if len(candidate) == 1:
                dict_empty[self.list_keys[i]] = candidate[0]
                fixed.append(i)
            else:
                variant.append(i)
                # need change empty side chain
                list_frag_r = self.list_frag[candidate]
                list_count_ring = np.array([frag.count(":[*:{}]".format(i)) for frag in list_frag_r])
                list_count_nonring = np.array([frag.count("[*:{}]".format(i)) for frag in list_frag_r])
                ring_counts = pd.value_counts(list_count_ring)
                nonring_counts = pd.value_counts(list_count_nonring)
                flag = False
                if ring_counts.index[0] == 2:
                    empty = "c(:[*:{}]):[*:{}]".format(i, i)
                elif ring_counts.index[0] == 0:
                    empty = '.'.join(["[H][*:{}]".format(i)] * nonring_counts.index[0])
                else:
                    flag = True
                if flag or empty not in self.list_frag[candidate]:
                    length = np.array([Chem.MolFromSmiles(frag, sanitize=False).GetNumAtoms() for frag in self.list_frag[candidate]])
                    empty = self.list_frag[np.random.choice(candidate[length == length.min()])]
                dict_empty[self.list_keys[i]] = self.frag2idx[empty]
        self.ori_idx = ori_idx
        self.fixed = np.array(fixed)
        self.variant = np.array(variant)
        self.dict_empty = dict_empty
        
    def predict_onehot(self, new_idx, return_std=False):
        for i, j in zip(*np.where(new_idx == 0)):
            new_idx[i, j] = self.dict_empty[self.list_keys[j]]
        list_mol = [makebond(item)[0] for item in self.list_frag_mol[new_idx]]
        list_newmol = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in list_mol]
        list_newmol = [Chem.RemoveHs(mol) for mol in list_newmol]

        X = np.array(get_fps(list_newmol))
        if return_std:
            y, std = self.predict(X, return_std=True)
            return y, std
        else:
            y = self.predict(X)
            return y