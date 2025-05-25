import sys
import os
sys.path.append(os.getcwd())
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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_predict(model, pdb_file, function, plm_path=None, blm_path=None, device='cpu'):
    function = 'Unknown' if function == '' else function
    model_path = f'pretrained/{model.lower()}.pth'

    if plm_path is not None and blm_path is not None and os.path.exists(plm_path) and os.path.exists(blm_path):
        pass
    else:
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

    model = torch.load(model_path, map_location=device)
    if 'esm3' not in plm_path:
        seq_tokenizer = AutoTokenizer.from_pretrained(plm_path)
        seq_model = AutoModel.from_pretrained(plm_path)
    else:
        if os.path.exists(plm_path):
            seq_model = ESM3.from_pretrained(plm_path, True, device)
        else:
            seq_model = ESM3.from_pretrained(plm_path, False, device)

    # get structure
    structure = read_pdb(pdb_file)

    # get property features
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

    # extract 3D 
    alpha_carbons = [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']
    positions = [atom.coord for atom in alpha_carbons]
    atom_indices = list(range(len(alpha_carbons)))

    # get sequence
    sequence = extract_sequence(structure)
    assert len(sequence) == len(alpha_carbons)
    if 'esm3' not in plm_path:
        node_features = get_features(sequence, seq_tokenizer, seq_model)
    else:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = seq_model.encode(protein)
        output = seq_model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        node_features = output.per_residue_embedding[1:-1]

    # construct graph
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

    # process text
    func = get_features(function, text_tokenizer, text_model, modal='text')

    # construct Data object
    data = Data(x=torch.tensor(np.array(atom_indices), dtype=torch.long).unsqueeze(1),
                edge_index=edge_index,
                edge_attr=edge_attr.unsqueeze(1),
                esm_rep=node_features,
                prop=torch.tensor(prop, dtype=torch.float),
                pos=torch.tensor(np.array(positions), dtype=torch.float),
                func=func).to(device)

    model_output = model(data)
    output = model_output.argmax(dim=-1).detach().cpu().numpy()
    confs = torch.max(model_output, dim=-1)[0].detach().cpu().numpy()

    res = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
    for i in range(len(output)):
        if output[i] != 0:
            res[str(output[i]-1)].append(i+1)   # start from 1

    return res, confs, sequence