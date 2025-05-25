from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.vectors import calc_dihedral
from hadder import AddHydrogen
import numpy as np
import mdtraj as md
import os
from tqdm import tqdm


def get_ss8_dim9(struture, pdb_path):
    """
    Calculate the secondary structure of a protein using DSSP and return it as an 8-class one-hot vector.
    Returns a 2D numpy array with shape (n_residues, 9).
    """
    dssp = DSSP(struture[0], pdb_path)
    ss8 = []
    for key in dssp.keys():
        ss = dssp[key][2] # Secondary structure
        ss_onehot = [0] * 9
        if ss == 'H':
            ss_onehot[0] = 1
        elif ss == 'B':
            ss_onehot[1] = 1
        elif ss == 'E':
            ss_onehot[2] = 1
        elif ss == 'G':
            ss_onehot[3] = 1
        elif ss == 'I':
            ss_onehot[4] = 1
        elif ss == 'T':
            ss_onehot[5] = 1
        elif ss == 'S':
            ss_onehot[6] = 1
        elif ss == '-':
            ss_onehot[7] = 1
        else:
            ss_onehot[8] = 1
        ss8.append(ss_onehot)
    return np.array(ss8)


def get_atom(residue, atom_name):
    """ Helper function to safely get an atom from a residue """
    return residue[atom_name] if atom_name in residue else None


def calculate_chi_angles(residue):
    """
    Calculate Chi1, Chi2, Chi3, Chi4, and Chi5 angles for a given residue.
    If an angle cannot be calculated, use 0 as a placeholder.
    """
    chi_angles = [0, 0, 0, 0, 0]

    # Common atoms for all chi angles
    n = get_atom(residue, 'N')
    ca = get_atom(residue, 'CA')
    cb = get_atom(residue, 'CB')

    # Chi1: N-CA-CB-CG (or equivalent)
    if residue.resname in ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
        if residue.resname in ['ILE', 'VAL']:
            cg = get_atom(residue, 'CG1')
        else:
            cg = get_atom(residue, 'CG')
        
        if n and ca and cb and cg:
            chi1 = calc_dihedral(n.get_vector(), ca.get_vector(), cb.get_vector(), cg.get_vector())
            chi_angles[0] = chi1

    # Chi2: CA-CB-CG-CD (or equivalent)
    if residue.resname in ['ARG', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'TRP', 'TYR']:
        cd = get_atom(residue, 'CD')
        if ca and cb and cg and cd:
            chi2 = calc_dihedral(ca.get_vector(), cb.get_vector(), cg.get_vector(), cd.get_vector())
            chi_angles[1] = chi2
        elif residue.resname in ['GLN', 'GLU']:
            cd = get_atom(residue, 'CD')
            if ca and cb and cg and cd:
                chi2 = calc_dihedral(ca.get_vector(), cb.get_vector(), cg.get_vector(), cd.get_vector())
                chi_angles[1] = chi2

    # Chi3: CB-CG-CD-CE (or equivalent)
    if residue.resname in ['ARG', 'GLN', 'GLU', 'LYS', 'MET']:
        ce = get_atom(residue, 'CE')
        if cb and cg and cd and ce:
            chi3 = calc_dihedral(cb.get_vector(), cg.get_vector(), cd.get_vector(), ce.get_vector())
            chi_angles[2] = chi3

    # Chi4: CG-CD-CE-NZ (or equivalent)
    if residue.resname in ['ARG', 'LYS']:
        nz = get_atom(residue, 'NZ')
        if cg and cd and ce and nz:
            chi4 = calc_dihedral(cg.get_vector(), cd.get_vector(), ce.get_vector(), nz.get_vector())
            chi_angles[3] = chi4

    # Chi5: CD-CE-NZ (only for ARG)
    if residue.resname == 'ARG':
        ne = get_atom(residue, 'NE')
        if cd and ce and nz and ne:
            chi5 = calc_dihedral(cd.get_vector(), ce.get_vector(), nz.get_vector(), ne.get_vector())
            chi_angles[4] = chi5

    return chi_angles


def get_dihedrals_dim16(structure, pdb_path):
    angles_matrix = []
    
    # Calculate Phi and Psi angles
    dssp = DSSP(structure[0], pdb_path)
    for key in dssp.keys():
        res = dssp[key]
        phi, psi = res[4], res[5]
        angles_matrix.append([
            np.sin(phi*np.pi/180), np.cos(phi*np.pi/180), np.sin(psi*np.pi/180), np.cos(psi*np.pi/180)
        ])

    # Calculate Omega angles
    angles_matrix[0] += [np.sin(0), np.cos(0)]
    residues = list(structure[0]['A'].get_residues())
    for i in range(1, len(residues)):
        c1 = get_atom(residues[i-1], 'C')
        n2 = get_atom(residues[i], 'N')
        ca2 = get_atom(residues[i], 'CA')
        c2 = get_atom(residues[i], 'C')
        omega_angle = calc_dihedral(c1.get_vector(), n2.get_vector(), ca2.get_vector(), c2.get_vector())
        angles_matrix[i] += [np.sin(omega_angle), np.cos(omega_angle)]

    # Calculate Chi angles
    for i, residue in enumerate(structure[0]['A']):
        chi_angles = calculate_chi_angles(residue)
        sin_cos_angles = []
        for angle in chi_angles:
            sin_cos_angles.append(np.sin(angle))
            sin_cos_angles.append(np.cos(angle))
        angles_matrix[i] += sin_cos_angles

    return np.array(angles_matrix)


def get_atom_features_dim7(structure):
    """
    Calculate atomic mass, B-factor, whether it is a residue side-chain atom, electronic charge, the number of hydrogen 
    atoms bonded to it, whether it is in a ring and the van der Waals radius of the atom.
    """
    atomic_masses = {'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 
                    'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Ag': 107.87, 'Sn': 118.71, 'I': 126.90, 
                    'Au': 196.97, 'Pb': 207.2, 'U': 238.03}
    electronic_charges = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 
                        'Ca': 20, 'Fe': 26, 'Cu': 29, 'Zn': 30, 'Ag': 47, 'Sn': 50, 'I': 53, 'Au': 79, 'Pb': 82, 'U': 92}
    vdw_radii = {'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 
                'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31, 'Fe': 1.93, 'Cu': 1.96, 'Zn': 1.87, 'Ag': 1.72, 'Sn': 2.17, 'I': 1.98, 'Au': 1.66, 'Pb': 2.02, 'U': 1.86}
    ring_atoms = {'HIS': ['ND1', 'CE1', 'NE2', 'CD2'],
                'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TRP': ['CG', 'CD1', 'CD2', 'CE3', 'NE1', 'CE2', 'CZ2', 'CH2']}

    residue_features = []
    for residue in structure[0]['A']:
        atom_features = []
        for atom in residue:
            if atom.element == 'H':
                continue
            atom_features.append([
                atomic_masses.get(atom.element, 0.0),
                atom.bfactor,
                int(atom.name not in residue.child_dict),
                electronic_charges.get(atom.element, 0),
                len([neighbor for neighbor in atom.get_parent() if neighbor.element == 'H']),
                int(residue.resname in ring_atoms and atom.name in ring_atoms[residue.resname]),
                vdw_radii.get(atom.element, 0.0)
            ])
        residue_features.append(np.mean(atom_features, axis=0).tolist())
    return np.array(residue_features)


def get_hbond_features_dim2(pdb_file):
    """
    Calculate hydrogen bond features using MDtraj.
    """
    traj = md.load(pdb_file)
    hbonds = md.kabsch_sander(traj)
    ax0 = hbonds[0].toarray().mean(axis=0)
    ax1 = hbonds[0].toarray().mean(axis=1)
    return np.column_stack((ax0, ax1))


def get_centroids(structure):
    """
    Calculate the centroid of each residue's side chain.
    """
    centroids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                side_chain_atoms = [atom for atom in residue.get_atoms() if atom.get_id() not in ['N', 'CA', 'C', 'O']]
                atom_coords = np.array([atom.get_coord() for atom in side_chain_atoms])
                centroid = np.mean(atom_coords, axis=0)
                centroids.append(centroid.tolist())
    return np.array(centroids)


def get_pef_features_dim1(structure, reference_index=0, r=1.0):
    """
    Calculate pseudo position embedding features.
    """
    centroids = get_centroids(structure)
    reference_coords = centroids[reference_index]
    distances = np.linalg.norm(centroids - reference_coords, axis=1)  # 欧几里得距离
    pseudo_position_embedding = distances / r
    return pseudo_position_embedding.reshape(-1, 1)


def get_residue_features_dim27(structure):
    """
    Calculate residue features including hydrophobicity, polarity, charge, pKa, volume, and mass.
    """
    category = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']
    hydrophobicity = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 
                      'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
    polar = {'ALA': 0, 'ARG': 1, 'ASN': 1, 'ASP': 1, 'CYS': 0, 'GLN': 1, 'GLU': 1, 'GLY': 0, 'HIS': 1, 'ILE': 0,
             'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0, 'SER': 1, 'THR': 1, 'TRP': 0, 'TYR': 0, 'VAL': 0}
    charge = {'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0, 'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.1, 'ILE': 0,
              'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0, 'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0}
    pKa = {'ALA': 2.34, 'ARG': 9.04, 'ASN': 2.02, 'ASP': 1.88, 'CYS': 1.96, 'GLN': 2.17, 'GLU': 2.19, 'GLY': 2.34, 'HIS': 1.82, 'ILE': 2.36,
           'LEU': 2.36, 'LYS': 2.18, 'MET': 2.28, 'PHE': 1.83, 'PRO': 1.99, 'SER': 2.21, 'THR': 2.15, 'TRP': 2.83, 'TYR': 2.20, 'VAL': 2.32}
    volume = {'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5, 'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
              'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7, 'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0}
    mass = {'ALA': 89.1, 'ARG': 174.2, 'ASN': 132.1, 'ASP': 133.1, 'CYS': 121.2, 'GLN': 146.2, 'GLU': 147.1, 'GLY': 75.1, 'HIS': 155.2, 'ILE': 131.2,
            'LEU': 131.2, 'LYS': 146.2, 'MET': 149.2, 'PHE': 165.2, 'PRO': 115.1, 'SER': 105.1, 'THR': 119.1, 'TRP': 204.2, 'TYR': 181.2, 'VAL': 117.1}

    categories, hydrophobicities, polarities, charges, pKas, volumes, masses = [], [], [], [], [], [], []
    for residue in structure[0]['A']:
        resname = residue.resname
        cat = np.zeros(len(category))
        cat[category.index(resname) if resname in category else -1] = 1
        categories.append(cat)
        hydrophobicities.append(hydrophobicity.get(resname, 0))
        polarities.append(polar.get(resname, 0))
        charges.append(charge.get(resname, 0))
        pKas.append(pKa.get(resname, 0))
        volumes.append(volume.get(resname, 0))
        masses.append(mass.get(resname, 0))
    return np.column_stack((categories, hydrophobicities, polarities, charges, pKas, volumes, masses))


if __name__ == '__main__':

    data_dir = ''   # data dir, who has the raw and property folders
    parser = PDBParser(QUIET=True)

    seq_names = ['I1BJN3', 'I1K0K6', 'I1R9B2', 'I1RF61']

    # Get the properties of the proteins
    print('Processing...')
    for sn in tqdm(seq_names):
        if not os.path.exists(os.path.join(data_dir, f'raw/{sn}.pdb')):
            continue

        pdb_file = os.path.join(data_dir, f'raw/{sn}.pdb')
        pdb_file_addH = os.path.join(data_dir, f'raw/{sn}.pdb')
        AddHydrogen(pdb_file, pdb_file_addH)
        structure = parser.get_structure('protein', pdb_file_addH)
        ss8 = get_ss8_dim9(structure, pdb_file_addH)
        angles_matrix = get_dihedrals_dim16(structure, pdb_file_addH)
        atom_feature = get_atom_features_dim7(structure)
        hbond_feature = get_hbond_features_dim2(pdb_file_addH)
        pef_feature = get_pef_features_dim1(structure)
        residue_feature = get_residue_features_dim27(structure)
        seq_feature = np.concatenate((ss8, angles_matrix, atom_feature, hbond_feature, pef_feature, residue_feature), axis=1)
        np.save(os.path.join(data_dir, f'property/{sn}_prop.npy'), seq_feature)
        os.remove(pdb_file_addH)

    print('Done.')
