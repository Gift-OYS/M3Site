# M<sup>3</sup>Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Identification and Classification

## 1. Environment

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

## 2. Dataset and Training

### 2.1 Use Preprocessed Data

#### 2.1.1 Dataset

We have provided the preprocessed data `esm3_abs.zip` in [LINK](https://pan.baidu.com/s/1JP3OgoU7reIbSsz-RFwiXQ?pwd=5dnb) to reproduce the main results in our paper for convenience. You can download and unzip the data to the `data` folder. The preprocessed data contains 25,883 protein as the type `torch_geometric.data.data.Data`. For each protein `UniProt_ID.pt`, the data contains the following attributes:
- `x`: The index of the amino acids in the protein sequence, with shape `[num_nodes, 1]`.
- `edge_index`: The edge index of the protein structure, with shape `[2, num_edges]`.
- `edge_attr`: The edge attributes of the protein structure, with shape `[num_edges, 1]`.
- `pos`: The 3D coordinates of the amino acids in the protein structure, with shape `[num_nodes, 3]`.
- `esm_rep`: The ESM3 embedding of the protein sequence, with shape `[num_nodes, 1536]`.
- `prop`: The properties of the protein structure, with shape `[num_nodes, 62]`.
- `func`: The functional embedding of the protein sequence, with shape `[768]`.

The labels of the protein active sites are stored in `data/labels.json`, where the keys are the protein IDs and the values are the active site labels. The active site labels are in the format of a list of integers, where each integer represents the class of the active site. The classes are defined as: `0: Non-active site, 1: CRI, 2: SCI, 3: PI, 4: PTCR, 5: IA, 6: SSA`.

We also provide the `train_0.x.tsv`, `valid_0.x.tsv`, and `test_0.x.tsv` files in the `data/splits` folder, which are split by sequence identity.

In summary, within the data path folder, you should have the following structure:
```
data/
├── esm3_abs/
│   ├── Q0TLW0.pt
│   └── ...
├── labels.json
└── splits/
    ├── train/
    │   ├── train_0.1.tsv
    │   ├── train_0.3.tsv
    │   ├── train_0.5.tsv
    │   ├── train_0.7.tsv
    │   └── train_0.9.tsv
    ├── valid/
    │   ├── valid_0.1.tsv
    │   ├── valid_0.3.tsv
    │   ├── valid_0.5.tsv
    │   ├── valid_0.7.tsv
    │   └── valid_0.9.tsv
    └── test/
        ├── test_0.1.tsv
        ├── test_0.3.tsv
        ├── test_0.5.tsv
        ├── test_0.7.tsv
        └── test_0.9.tsv
```

> *Tips*: If you want to just replace the PLM or BLM, you just need to replace the `esm_rep` and `func` attributes in the `torch_geometric.data.data.Data` object.

#### 2.1.2 Training

If you want to train the model using our preprocessed data, you just need to:
1. Specify the data path `dataset:data_path` in the `configs/config.yaml` file. 
2. Specify the split threshold `dataset:split` in the `configs/config.yaml` file, which can be `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`.
3. Run the following command to train the model:
```bash
python train.py --config /path/to/config.yaml
```

After the training, you will get the training logs and checkpoints in the `runs/timestamp` folder. The training logs will be saved in `run.log`, and the checkpoints will be saved as `best_model_xxx.pth` files.

### 2.2 Use Custom Data

#### 2.2.1 Preprocess the Data from Scratch

If you want to preprocess the data from scratch, you can download the data from [UniProt](https://www.uniprot.org/) and [AlphaFold database](https://alphafold.com/). and filter, cluster, and split as the description in our manuscript. You can refer to the [this repository](https://github.com/Gift-OYS/MMSite) for the details to cluster and split the data. After that, you should put your `pdb` files in the `data/raw` folder.

**Generate Structual Properties**: You can use the `utils/get_property.py` script to generate the `UniProtID_prop.npy` files for each protein in the `data/raw` folder. The script will generate the 62-dim structural properties of the protein structure `UniProtID_prop.npy` and save them in the `data/property` folder. The strucure of the `data` folder should be like:
```
data/
├── raw/
│   ├── P0A6F5.pdb
│   ├── P0A6F6.pdb
│   └── ...
├── property/
│   ├── P0A6F5_prop.npy
│   ├── P0A6F6_prop.npy
│   └── ...
├── labels.json
└── splits/
    ├── train/
    │   └── train_0.x.tsv
    ├── valid/
    │   └── valid_0.x.tsv
    └── test/
        └── test_0.x.tsv
```

**Prepare Pretrained PLM and BLM**: When you train our M<sup>3</sup>Site model from scratch, you need to prepare the pretrained PLM and BLM model, and place them in the `pretrained_model` folder. As for the specific type, you can refer to our paper

#### 2.2.2 Training

Similar to Section 2.1.2, you should specify the data path `dataset:data_path` and the split threshold `dataset:split` in the `configs/config.yaml` file. Besides, you should also specify the pretrained PLM and BLM model path in the `configs/config.yaml` file. After that, you can run the training command:
```bash
python train.py --config /path/to/config.yaml
```
After the training, you will get the training logs and checkpoints in the `runs/timestamp` folder. The training logs will be saved in `run.log`, and the checkpoints will be saved as `best_model_xxx.pth` files.

## 3. Inference & Demo

For inference, you can use the trained model to predict the active site of a protein. You can refer to `app/inference.ipynb` for the inference process. We also provide some cases in the `app/case_study` folder, which contains some example `.pdb` files. You can use these files to test the inference process.

> *Tips*: If you use ESM3 to generate the protein embedding, you may need to have access the ESM3 model. You can refer to the [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) for instruction.

To facilitate the use of our model, we provide a demo for inference based on Gradio. The source code with Dockerfile are in the `app` folder. You can directly deploy it in [Hugging Face Spaces](https://huggingface.co/spaces) or run it locally.