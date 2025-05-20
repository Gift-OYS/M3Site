# M<sup>3</sup>Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Identification and Classification

## 1. Preparation

### 1.1 Environment
You can manage the environment by Anaconda. We have provided the environment configuration file `environment.yml` for reference. You can create the environment by the following command:
```bash
conda env create -f environment.yml
```
or you can create a new environment by `requirements.txt`:
```bash
conda create -n m3site python=3.11 dssp -c ostrokach -y
conda activate m3site
pip install -r requirements.txt
```

### 1.2 Dataset

If you want to preprocess the data by yourself, you can download the data from UniProt and filter, cluster, and split as the description in our manuscript. To help you access the property of protein structure, you can refer to `utils/get_property.py` to generate `.npy` files for each protein, and refer to this [repository](https://github.com/Gift-OYS/MMSite) for split approach.

However, we recommend you to download and unzip the preprocessed data from the following [link](https://pan.baidu.com/s/1JP3OgoU7reIbSsz-RFwiXQ?pwd=5dnb) to reproduce the main results in our paper for convenience. You can put all the downloaded data in the `data` folder.

## 2. Training

### 2.1 Configuration
You can specify the configuration in `config.yaml`, in which you can set the hyperparameters, the path of the data, and so on. You can refer to the `config.yaml` in the `configs` folder for reference.

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
