'''Dataset'''

import os
import json
from torch_geometric.data import Dataset
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class ProteinDataset(Dataset):
    def __init__(self, config, type, transform=None, pre_transform=None, process=False):
        self.config = config
        self.type = type
        self.parser = PDBParser(QUIET=True)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(f'{config.model.model_dir}/{config.model.esm_version}')
        self.seq_model = AutoModel.from_pretrained(f'{config.model.model_dir}/{config.model.esm_version}')
        self.text_tokenizer = AutoTokenizer.from_pretrained(f'{config.model.model_dir}/{config.model.pubmed_version}')
        self.text_model = AutoModel.from_pretrained(f'{config.model.model_dir}/{config.model.pubmed_version}')
        self.text_max_length = config.dataset.text_max_length
        self.threshold = config.dataset.edge_radius
        self.root = config.dataset.data_path
        self.df = pd.read_csv(f'{self.root}/split/{type}/{type}_{config.dataset.split}.tsv', sep='\t')
        self.df = self.df.dropna(subset=['Function [CC]']).reset_index(drop=True)
        self.label_json = json.load(open(f'{self.root}/label.json', 'r'))

        # to_accelerate_read 1
        self.pre_return_list = []
        for file in tqdm(os.listdir(self.raw_dir)):
            entity = file.split('-')[1]
            if file.endswith('.pdb') and entity in self.df['Entry'].values:
                propcess_dir = os.path.join(self.root, config.dataset.tag)
                pt_path = os.path.join(propcess_dir, f'{file.split(".")[0]}.pt')
                if os.path.exists(pt_path):
                    tmp_d = torch.load(pt_path)
                    if tmp_d.x.shape[0] == len(self.label_json[entity]):
                        self.pre_return_list.append(file)
        # to_accelerate_read 2
        names = [i.split('.')[0] for i in self.pre_return_list]
        self.pre_processed_file_names = [f'{name}.pt' for name in names]

        super().__init__(self.root, transform, pre_transform)
        if process:
            self.my_process()

    @property
    def raw_file_names(self):
        return self.pre_return_list

    @property
    def processed_file_names(self):
        return self.pre_processed_file_names

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.config.dataset.tag)

    @property
    def property_dir(self):
        return os.path.join(self.root, 'property')
    
    def download(self):
        pass

    def my_process(self):
        for raw_path in tqdm(self.raw_paths):

            name = raw_path.split('/')[-1].split('.')[0]
            if os.path.exists(os.path.join(self.processed_dir, f'{name}.pt')):
                continue

            entity = raw_path.split('/')[-1].split('-')[1]
            structure = self._read_pdb(raw_path)
            prop = self._get_props(entity)
            if prop is None:
                continue
            data = self._get_graph(structure, prop, entity)
            name = raw_path.split('/')[-1].split('.')[0]
            print('Saving', name)
            torch.save(data, os.path.join(self.processed_dir, f'{name}.pt'))

    def __len__(self):
        return len(self.raw_file_names)

    def __getitem__(self, idx):
        name = self.raw_file_names[idx].split('.')[0]
        entity = name.split('-')[1]
        label = torch.tensor(self.label_json[entity], dtype=torch.long)
        data = torch.load(os.path.join(self.processed_dir, f'{name}.pt'))
        data.y = label
        return data

    def _get_props(self, entity):
        prop_path = os.path.join(self.property_dir, f'{entity}_prop.npy')
        if not os.path.exists(prop_path):
            return None
        prop = np.load(prop_path)
        return torch.tensor(prop, dtype=torch.float)
    
    def _read_pdb(self, file_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        return structure
    
    def _extract_sequence(self, structure):
        sequence = []
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in three_to_one:
                        sequence.append(three_to_one[residue.get_resname()])
                    else:
                        sequence.append('X')
        return ''.join(sequence)

    def _get_features(self, sequence, tokenizer, model, modal='sequence'):
        if modal == 'sequence':
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[0]
        else:
            inputs = tokenizer(sequence, max_length=self.text_max_length, padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.pooler_output[0]

    def _get_graph(self, structure, prop, entity):
        alpha_carbons = [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']
        positions = [atom.coord for atom in alpha_carbons]
        atom_indices = list(range(len(alpha_carbons)))
        
        # 获取结点特征
        sequence = self._extract_sequence(structure)
        assert len(sequence) == len(alpha_carbons)
        node_features = self._get_features(sequence, self.seq_tokenizer, self.seq_model)

        # 构建边
        edges, edge_attrs = [], []
        for i, atom1 in enumerate(alpha_carbons):
            for j, atom2 in enumerate(alpha_carbons):
                if i < j:
                    distance = np.linalg.norm(atom1.coord - atom2.coord)
                    if distance < self.threshold:
                        edges.append((i, j))
                        edge_attrs.append(distance)
        edge_index = torch.tensor(np.array(edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

        data = Data(x=torch.tensor(np.array(atom_indices), dtype=torch.long).unsqueeze(1),
                    edge_index=edge_index,
                    edge_attr=edge_attr.unsqueeze(1),
                    esm_rep=node_features,
                    prop=prop,
                    pos=torch.tensor(np.array(positions), dtype=torch.float),
                    )
        return data
