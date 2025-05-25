# M<sup>3</sup>Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Identification and Classification

## 1. Preparation

### 1.1 Environment

You can manage the environment by Anaconda. We have provided the environment configuration file `environment.yml` for reference. You can create the environment by the following command:
```bash
conda env create -f environment.yml
```
or you can create the environment by `requirements.txt`:
```bash
conda create -n m3site python=3.11 dssp -c ostrokach -y
conda activate m3site
pip install -r requirements.txt
```

### 1.2 Dataset

#### 1.2.1 Use Preprocessed Data

We have provided the preprocessed data `3_abs.zip` in [link](https://pan.baidu.com/s/1JP3OgoU7reIbSsz-RFwiXQ?pwd=5dnb) to reproduce the main results in our paper for convenience. You can download and unzip the data to the `data` folder. The preprocessed data contains 53,461 protein sequences as the type `torch_geometric.data.data.Data`. For each protein `UniProt_ID.pt`, the data contains the following attributes:
- `x`: The index of the amino acids in the protein sequence, with shape `[num_nodes, 1]`.
- `edge_index`: The edge index of the protein structure, with shape `[2, num_edges]`.
- `edge_attr`: The edge attributes of the protein structure, with shape `[num_edges, 1]`.
- `pos`: The 3D coordinates of the amino acids in the protein structure, with shape `[num_nodes, 3]`.
- `esm_rep`: The ESM3 embedding of the protein sequence, with shape `[num_nodes, 1536]`.
- `prop`: The properties of the protein structure, with shape `[num_nodes, 62]`.
- `func`: The functional embedding of the protein sequence, with shape `[768]`.

The labels of the protein active sites are stored in `data/labels.json`, where the keys are the protein IDs and the values are the active site labels. The active site labels are in the format of a list of integers, where each integer represents the class of the active site. The classes are defined as: `0`: Non-active site, `1`: CRI, `2`: SCI, `3`: PI, `4`: PTCR, `5`: IA, `6`: SSA.

We also provide the `train_0.x.tsv`, `valid_0.x.tsv`, and `test_0.x.tsv` files in the `data/splits` folder, which are split by sequence identity.


#### 1.2.2 Preprocess the Data by Yourself

**Replace PLM/BLM**: If you want to replace the PLM or BLM, you just need to replace the `esm_rep` and `func` attributes in the `torch_geometric.data.data.Data` object.

**From Scratch**: If you want to preprocess the data from scratch, you can download the data from UniProt and filter, cluster, and split as the description in our manuscript. To help you access the 62-dim `prop` attribute, we have provided the `utils/get_property.py` script to generate the `.npy` files for each protein. As for the clustering and splitting steps, you can refer to the [this repository](https://github.com/Gift-OYS/MMSite) for the details.

## 2. Training

### 2.1 Configuration

You can specify the configuration in `config.yaml`, in which you can set the hyperparameters, the path of the data, and so on.

### 2.2 Training
You can train the model by the following command:
```bash
python train.py --config /path/to/config.yaml
```

Then, you will get `best_model_xxx.pth` model in the `runs/timestamp` folder, which is the final model.

## 3. Inference & Demo

For inference, you can use the trained model to predict the active site of a protein. You can refer to `app/inference.ipynb` for the inference process. 

> *Tips*: If you use ESM3 to generate the protein embedding, you may need to have access the ESM3 model. You can refer to the [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) for instruction.

To facilitate the use of our model, we provide a demo for inference based on Gradio. The source code is hosted in `app` folder.
