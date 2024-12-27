# M<sup>3</sup>Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Iden-tification and Classification

> This implementation is based on our previous work [MMSite: A Multi-modal Framework for the Identification of Active Sites in Proteins](https://openreview.net/pdf?id=XHdwlbNSVb) (NeurIPS 2024), whose code is available at [GitHub](https://github.com/Gift-OYS/MMSite).

## 1. Preparation

### 1.1 Environment
You can manage the environment by Anaconda. We have provided the environment configuration file `environment.yml` for reference. You can create the environment by the following command:
```bash
conda env create -f environment.yml
```

### 1.2 Dataset

If you want to preprocess the data by yourself, you can download the data from UniProt and filter, cluster, and split as the description in our manuscript. To help you access the property of protein structure, you can refer to `utils/get_property.py` to generate `.npy` files for each protein, and refer to this [repository](https://github.com/Gift-OYS/MMSite) for split approach.

However, we recommend you to download and unzip the preprocessed data from the following [link](xxxx) to reproduce the main results in our paper for convenience. You can put all the downloaded data in the `data` folder.

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

To facilitate the use of our model, we provide a demo for inference based on Gradio. It is hosted at Hugging Face Spaces, and you can access it by the following [link](https://huggingface.co/spaces/GiftOYS/M3Site).
