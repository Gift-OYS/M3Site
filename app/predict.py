import sys
import os
sys.path.append(f'{os.getcwd()}/esm3')
import torch
from utils.util_functions import read_pdb, extract_sequence, get_features
from Bio.PDB import PDBParser
from hadder import AddHydrogen
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.get_property_embs import get_ss8_dim9, get_dihedrals_dim16, get_atom_features_dim7, get_hbond_features_dim2, get_pef_features_dim1, get_residue_features_dim27
from torch_geometric.data import Data
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.sdk.api import ESMProtein, SamplingConfig

def model_predict(model, pdb_file, function):
    function = 'Unknown' if function == '' else function

    model_path = f'pretrained/{model.lower()}.pth'
    if model == 'M3Site-ESM3-abs':
        plm_path = ESM3_OPEN_SMALL
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif model == 'M3Site-ESM3-full':
        plm_path = ESM3_OPEN_SMALL
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    elif model == 'M3Site-ESM2-abs':
        plm_path = 'facebook/esm2_t33_650M_UR50D'
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif model == 'M3Site-ESM2-full':
        plm_path = 'facebook/esm2_t33_650M_UR50D'
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    elif model == 'M3Site-ESM1b-abs':
        plm_path = 'facebook/esm1b_t33_650M_UR50S'
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif model == 'M3Site-ESM1b-full':
        plm_path = 'facebook/esm1b_t33_650M_UR50S'
        blm_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

    login(token=os.environ.get("ESM3TOKEN"))

    text_tokenizer = AutoTokenizer.from_pretrained(blm_path)
    text_model = AutoModel.from_pretrained(blm_path)

    model = torch.load(model_path, map_location='cpu')
    if 'esm3' not in plm_path:
        seq_tokenizer = AutoTokenizer.from_pretrained(plm_path)
        seq_model = AutoModel.from_pretrained(plm_path)
    else:
        seq_model = ESM3.from_pretrained(plm_path)

    # 得到structure
    structure = read_pdb(pdb_file)

    # 得到prop
    parser = PDBParser(QUIET=True)
    pdb_file_addH = pdb_file.split('.')[0] + '_addH.pdb'
    AddHydrogen(pdb_file, pdb_file_addH)
    struct = parser.get_structure('protein', pdb_file_addH)
    ss8 = get_ss8_dim9(struct, pdb_file_addH)
    angles_matrix = get_dihedrals_dim16(struct, pdb_file_addH)
    atom_feature = get_atom_features_dim7(struct)
    hbond_feature = get_hbond_features_dim2(pdb_file_addH)
    pef_feature = get_pef_features_dim1(struct)
    residue_feature = get_residue_features_dim27(struct)
    prop = np.concatenate((ss8, angles_matrix, atom_feature, hbond_feature, pef_feature, residue_feature), axis=1)
    os.remove(pdb_file_addH)

    # 提取三维信息
    alpha_carbons = [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']
    positions = [atom.coord for atom in alpha_carbons]
    atom_indices = list(range(len(alpha_carbons)))

    # 获取结点特征
    sequence = extract_sequence(structure)
    assert len(sequence) == len(alpha_carbons)
    if 'esm3' not in plm_path:
        node_features = get_features(sequence, seq_tokenizer, seq_model)
    else:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = seq_model.encode(protein)
        output = seq_model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        node_features = output.per_residue_embedding[1:-1]

    # 构建边
    edges, edge_attrs = [], []
    for i, atom1 in enumerate(alpha_carbons):
        for j, atom2 in enumerate(alpha_carbons):
            if i < j:
                distance = np.linalg.norm(atom1.coord - atom2.coord)
                if distance < 8.0:
                    edges.append((i, j))
                    edge_attrs.append(distance)
    edge_index = torch.tensor(np.array(edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    # 处理文本
    func = get_features(function, text_tokenizer, text_model, modal='text')

    # 构建Data对象
    data = Data(x=torch.tensor(np.array(atom_indices), dtype=torch.long).unsqueeze(1),
                edge_index=edge_index,
                edge_attr=edge_attr.unsqueeze(1),
                esm_rep=node_features,
                prop=torch.tensor(prop, dtype=torch.float),
                pos=torch.tensor(np.array(positions), dtype=torch.float),
                func=func)

    model_output = model(data)
    output = model_output.argmax(dim=-1).numpy()
    confs = torch.max(model_output, dim=-1)[0].detach().numpy()

    res = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
    for i in range(len(output)):
        if output[i] != 0:
            res[str(output[i]-1)].append(i+1)   # 返回的是从1开始编号的
            
    return res, confs, sequence