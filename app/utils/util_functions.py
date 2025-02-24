import torch
from Bio.PDB import PDBParser


def read_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    return structure


def extract_sequence(structure):
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


def get_features(sequence, tokenizer, model, modal='sequence'):
    if modal == 'sequence':
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0]
    else:
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output[0]


def get_emb_dim(esm_version):
    if 'esm2_t33_650M_UR50D' in esm_version:
        return 1280
    elif 'esm1b_t33_650M_UR50S' in esm_version:
        return 1280

def merge_ranges(data, max_value=200):
    result = {}
    all_values, used_values = set(range(max_value + 1)), set()
    
    for key, values in data.items():
        if not values:
            result[key] = []
            continue
        values.sort()
        used_values.update(values)
        merged = []
        start, end = values[0], values[0]
        for i in range(1, len(values)):
            if values[i] == end + 1:
                end = values[i]
            else:
                merged.append(f"{start}-{end}" if start != end else str(start))
                start = values[i]
                end = values[i]

        merged.append(f"{start}-{end}" if start != end else str(start))
        result[key] = merged

    remaining_values = sorted(all_values - used_values)  # 未使用的值
    if remaining_values:
        merged = []
        start = remaining_values[0]
        end = remaining_values[0]
        for i in range(1, len(remaining_values)):
            if remaining_values[i] == end + 1:
                end = remaining_values[i]
            else:
                merged.append(f"{start}-{end}" if start != end else str(start))
                start = remaining_values[i]
                end = remaining_values[i]
        merged.append(f"{start}-{end}" if start != end else str(start))
        result["b"] = merged
    else:
        result["b"] = []
    
    return result