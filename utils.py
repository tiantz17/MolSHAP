import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from rdkit.Chem import rdDepictor
from collections import defaultdict

def get_mol_from_smiles(smiles, remove_Hs=True):
    mol = Chem.MolFromSmiles(smiles)
    if remove_Hs:
        mol = Chem.RemoveHs(mol)
    return mol

def get_fp(mol, nBits=1024, radius=2):
    """
    calculating molecular fingerprints
    Input: rdkit Mol
    Output: rdkit fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=nBits,useChirality=True)
    return fp

def get_fps(mol_list, nBits=1024, radius=2):
    """
    calculating molecular fingerprints
    Input: list of rdkit Mol
    Output: list of rdkit fingerprint
    """
    fps = []
    for i, mol in enumerate(mol_list):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=nBits,useChirality=True)
        fps.append(fp)
    return fps

def makebond_from_smiles(res):
    list_rg = res 
    list_r = []
    existed_r = set()
    newmol = Chem.MolFromSmiles(list_rg[0], sanitize=False)
    list_r.append(set([atom.GetIdx() for atom in newmol.GetAtoms()]))
    existed_r = existed_r.union(list_r[-1])
    for mol in list_rg[1:]:
        newmol = Chem.CombineMols(newmol, Chem.MolFromSmiles(mol, sanitize=False))
        list_r.append(set([atom.GetIdx() for atom in newmol.GetAtoms()]).difference(existed_r))
        existed_r = existed_r.union(list_r[-1])
    
    newmol = Chem.RWMol(newmol)
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    for atm in atoms:
        idx = atm.GetIdx()
        atom_name = atm.GetAtomMapNum()
        if atom_name == 0:
            continue
        mapper[atom_name].append(idx)
    for idx, a_list in mapper.items():
        nbr = [[x.GetOtherAtom(newmol.GetAtomWithIdx(atm)) for x in newmol.GetAtomWithIdx(atm).GetBonds()] for atm in a_list]
        if len(a_list) == 2:
            newmol.AddBond(nbr[0][0].GetIdx(), nbr[1][0].GetIdx(), order=Chem.rdchem.BondType.SINGLE)
        elif len(a_list) >= 3:
            count = defaultdict(lambda:0)
            for atms in nbr:
                for atm in atms:
                    count[atm.GetIdx()] += 1
            maxnum = 0
            maxidx = None
            for atm in count:
                if count[atm] > maxnum:
                    maxnum = count[atm]
                    maxidx = atm
            for a, bs in zip(a_list, nbr):
                for b in bs:
                    bondtype = newmol.GetBondBetweenAtoms(a, b.GetIdx()).GetBondType()
                    newmol.RemoveBond(a, b.GetIdx())
                    if int(maxidx) != b.GetIdx():
                        newmol.AddBond(int(maxidx), b.GetIdx(), bondtype)

    num_atm = newmol.GetNumAtoms()
    idx2idx = {i:i for i in range(num_atm)}
    for atm in sorted(np.concatenate(list(mapper.values())))[::-1]:
        idx2idx[atm] = -1
        for i in range(num_atm):
            if idx2idx[i]>atm:
                idx2idx[i] -= 1
        newmol.RemoveAtom(int(atm))

    newmol = newmol.GetMol()
    newmol.UpdatePropertyCache()
    Chem.Kekulize(newmol)
    return newmol, [[idx2idx[idx] for idx in item if idx2idx[idx]>=0] for item in list_r]

def makebond(res):
    list_rg = res 
    list_r = []
    existed_r = set()
    newmol = list_rg[0]
    list_r.append(set([atom.GetIdx() for atom in newmol.GetAtoms()]))
    existed_r = existed_r.union(list_r[-1])
    for mol in list_rg[1:]:
        newmol = Chem.CombineMols(newmol, mol)
        list_r.append(set([atom.GetIdx() for atom in newmol.GetAtoms()]).difference(existed_r))
        existed_r = existed_r.union(list_r[-1])
    
    newmol = Chem.RWMol(newmol)
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    for atm in atoms:
        idx = atm.GetIdx()
        atom_name = atm.GetAtomMapNum()
        if atom_name == 0:
            continue
        mapper[atom_name].append(idx)
    for idx, a_list in mapper.items():
        nbr = [[x.GetOtherAtom(newmol.GetAtomWithIdx(atm)) for x in newmol.GetAtomWithIdx(atm).GetBonds()] for atm in a_list]
        if len(a_list) == 2:
            newmol.AddBond(nbr[0][0].GetIdx(), nbr[1][0].GetIdx(), order=Chem.rdchem.BondType.SINGLE)
        elif len(a_list) >= 3:
            count = defaultdict(lambda:0)
            for atms in nbr:
                for atm in atms:
                    count[atm.GetIdx()] += 1
            maxnum = 0
            maxidx = None
            for atm in count:
                if count[atm] > maxnum:
                    maxnum = count[atm]
                    maxidx = atm
            for a, bs in zip(a_list, nbr):
                for b in bs:
                    bond = newmol.GetBondBetweenAtoms(a, b.GetIdx()).GetBondType()
                    newmol.RemoveBond(a, b.GetIdx())
                    if int(maxidx) != b.GetIdx():
                        newmol.AddBond(int(maxidx), b.GetIdx(), bond)
    num_atm = newmol.GetNumAtoms()
    idx2idx = {i:i for i in range(num_atm)}
    for atm in sorted(np.concatenate(list(mapper.values())))[::-1]:
        idx2idx[atm] = -1
        for i in range(num_atm):
            if idx2idx[i]>atm:
                idx2idx[i] -= 1
        newmol.RemoveAtom(int(atm))
    newmol = newmol.GetMol()
    newmol.UpdatePropertyCache()
    Chem.Kekulize(newmol)
    return newmol, [[idx2idx[idx] for idx in item if idx2idx[idx]>=0] for item in list_r]

def plot_with_color(mol, list_r, lineWidth=3, fontSize=22):
    colors = [
        (100, 100, 100), 
        (86,180,233), # blue
        (230,159,0), # yellow
        (0,190,150), # green
        (204,121,167), # pale rose
        (180,141,255), # purple
        (254,46,152), # rose
        (254,97,0), # orange
        (120,94,240), # purple
        (100,143,255), # royal blue
        (213,94,0), # brown
        (0,114,178), # dark blue 
        (240,228,66), # light yellow
        (204,121,167), # pale rose
        (255,176,0) # yellow orange
    ]

    for i,x in enumerate(colors):
        colors[i] = tuple(y/255 for y in x)
        
        
    list_bond_idx = [[bond.GetIdx(), bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in mol.GetBonds()]
    dict_color = {int(idx):[colors[i%len(colors)]] for i in range(1, len(list_r)) for idx in list_r[i]}
    list_b = [[int(b[0]) for b in list_bond_idx if b[1] in list_r[i] and b[2] in list_r[i]] for i in range(len(list_r))]
    dict_b_color = {int(b):[colors[i%len(colors)]] for i in range(1, len(list_b)) for b in list_b[i]}
    atomrads = {i:0.4 for i in dict_color}
    widthmults = {i:2 for i in dict_b_color}

    Chem.GetSSSR(mol)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    rinfo = mol.GetRingInfo()
    rings = []
    for aring in rinfo.AtomRings():
        for i in range(1, len(list_r)):
            overlap = np.intersect1d(aring, list_r[i])
            if len(overlap) == len(aring):
                rings.append([aring, colors[i%len(colors)]])

    d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(500,300)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.bondLineWidth = lineWidth

    d2d.DrawMoleculeWithHighlights(mol,
                                   "",
                                   dict_color,
                                   dict_b_color,
                                   atomrads, 
                                   widthmults)
    d2d.ClearDrawing()
    for (aring,color) in rings:
        ps = []
        for aidx in aring:
            pos = rdkit.Geometry.Point2D(conf.GetAtomPosition(aidx))
            ps.append(pos)
        d2d.SetFillPolys(True)
        d2d.SetColour(color)
        d2d.DrawPolygon(ps)
    dos.clearBackground = False
    d2d.SetFontSize(fontSize)

    #----------------------
    # now draw the molecule, with highlights:
    d2d.DrawMoleculeWithHighlights(mol,"",dict_color,
                                           dict_b_color,
                                           atomrads, widthmults)
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    return png