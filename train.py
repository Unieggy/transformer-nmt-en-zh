import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from dataset import BilingualDataset,causal_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import sentencepiece as spm
from model import build_transfomer
from config import get_config,get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import torchmetrics
from torch.utils.data._utils.collate import default_collate
import os, re, tempfile
def find_latest_checkpoint(config):
    folder = Path(config["model_folder"])
    base = config["model_basename"]
    latest_num, latest_file = -1, None
    if folder.exists():
        for fn in os.listdir(folder):
            m = re.match(rf"{re.escape(base)}(\d+)\.pt$", fn)
            if m:
                num = int(m.group(1))
                if num > latest_num:
                    latest_num, latest_file = num, folder / fn
    return latest_file

def save_checkpoint_atomic(state, path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp.flush(); os.fsync(tmp.fileno())
    os.replace(tmp.name, path)

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None
def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx= tokenizer_tgt.piece_to_id('[SOS]')
    eos_idx= tokenizer_tgt.piece_to_id('[EOS]')
    encoder_output=model.encode(source,source_mask)
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out= model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        prob=model.projection(out[:,-1])
        _, next_word=prob.max(dim=-1)
        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)
def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,device,print_msg,max_len,global_step,writer,num_examples=2):
    model.eval()
    count =0
    with torch.no_grad():
        for batch in validation_ds:
            if batch is None:
                continue
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            model_output=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            source_text=batch['src_text'][0]
            target_text=batch['tgt_text'][0]
            model_out_text=tokenizer_tgt.decode(model_output.detach().cpu().numpy().tolist())


            print_msg(f"Source: {source_text}\nExpected: {target_text}\nPredicted: {model_out_text}\n")
            if count >= num_examples:
                break
    if writer:
        metric=torchmetrics.CharErrorRate()
        cer=metric(model_out_text,target_text)
        writer.add_scalar('validation/cer', cer, global_step)
        writer.flush()

        metric=torchmetrics.WordErrorRate()
        wer=metric(model_out_text,target_text)
        writer.add_scalar('validation/wer', wer, global_step)
        writer.flush()

        metric=torchmetrics.BLEUScore()
        bleu=metric([model_out_text], [[target_text]])
        writer.add_scalar('validation/bleu', bleu, global_step)
        writer.flush()
   
def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

from pathlib import Path

SPECIALS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

def get_or_build_tokenizer(config, ds, lang):
    model_path = Path(config["tokenizer_file"].format(lang)).with_suffix(".model")
    if not model_path.exists():
        txt_path = model_path.with_suffix(".txt")
        with txt_path.open("w", encoding="utf‑8") as f:
            for item in ds:
                f.write(item["translation"][lang].replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.Train(
            input=str(txt_path),
            model_prefix=str(model_path).rstrip(".model"),
            vocab_size=32000,
            model_type="bpe",
            character_coverage=1.0,         
            user_defined_symbols=",".join(SPECIALS)
        )

    # 2.load model
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    return sp


def get_ds(config):
    ds_raw=load_dataset('Helsinki-NLP/opus-100',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    train_ds_size=int(0.9 * len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']],out_type=int)
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']],out_type=int)
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f'Max len of src sentence:{max_len_src}')
    print(f'Max len of tgt sentence:{max_len_tgt}')

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True,collate_fn=safe_collate)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=True,collate_fn=safe_collate)
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_Len, vocab_tgt_Len):
    model=build_transfomer(vocab_src_Len,vocab_tgt_Len,config['seq_len'],config['seq_len'],config['dmodel'])
    return model

def train_model(config):
    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f'Using device:{device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_piece_size(),tokenizer_tgt.get_piece_size())
    model.to(device)

    writer=SummaryWriter(config['experiment_name'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    initial_epoch = 0
    global_step = 0

    ckpt_path = None
    if config.get('preload'):
        ckpt_path = get_weights_file_path(config, config['preload'])
    else:
        latest = find_latest_checkpoint(config)
        if latest:
            ckpt_path = str(latest)

    if ckpt_path and Path(ckpt_path).exists():
        print(f'Loading model weights from {ckpt_path}')
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])             
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step  = int(state.get('global_step', 0))
        initial_epoch = int(state.get('epoch', -1)) + 1           
    else:
        print('No checkpoint found, training from scratch.')

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.piece_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        batch_iterator=tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}', unit='batch')
        for batch in batch_iterator:
            if batch is None: 
                continue
            model.train()
            encoder_input=batch['encoder_input'].to(device)
            decoder_input=batch['decoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            decoder_mask=batch['decoder_mask'].to(device)

            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output=model.projection(decoder_output)
            label=batch['label'].to(device)
            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_piece_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': loss.item()})
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if global_step % 500 == 0:
                run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,device,lambda msg:batch_iterator.write(msg),config['seq_len'],global_step,writer)
            global_step += 1

        model_filename=get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
if __name__ == '__main__':
    warnings.filterwarnings
    config=get_config()
    train_model(config)


