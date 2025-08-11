from pathlib import Path
def get_config():
    return{
        "batch_size":8,
        "num_epochs":20,
        "lr":1e-4,
        "dmodel":512,
        "seq_len":128,
        "lang_src":"en",
        "lang_tgt":"zh",
        "model_basename": "tmodel_",
        "model_folder": "/content/drive/MyDrive/weights",
        "tokenizer_file": "/content/drive/MyDrive/tokenizer_{0}.json",
        "experiment_name": "/content/drive/MyDrive/runs/tmodel",
        "preload": None,
    }
def get_weights_file_path(config,_epoch:str):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{_epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
