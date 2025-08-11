import torch
from torch.utils.data import Dataset
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt,
                 src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_sos = torch.tensor([tokenizer_src.piece_to_id('[SOS]')], dtype=torch.int64)
        self.src_eos = torch.tensor([tokenizer_src.piece_to_id('[EOS]')], dtype=torch.int64)
        self.src_pad = torch.tensor([tokenizer_src.piece_to_id('[PAD]')], dtype=torch.int64)

        
        self.tgt_sos = torch.tensor([tokenizer_tgt.piece_to_id('[SOS]')], dtype=torch.int64)
        self.tgt_eos = torch.tensor([tokenizer_tgt.piece_to_id('[EOS]')], dtype=torch.int64)
        self.tgt_pad = torch.tensor([tokenizer_tgt.piece_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair = self.ds[idx]
        src_text = pair['translation'][self.src_lang]
        tgt_text = pair['translation'][self.tgt_lang]

        enc_ids = self.tokenizer_src.encode(src_text, out_type=int)
        dec_ids = self.tokenizer_tgt.encode(tgt_text, out_type=int)

        enc_pad = self.seq_len - len(enc_ids) - 2  # SOS + EOS
        dec_pad = self.seq_len - len(dec_ids) - 1  # SOS or EOS
        if enc_pad < 0 or dec_pad < 0:
            return None  # sentence too long, drop

        # -------- encoder input --------
        encoder_input = torch.cat([
            self.src_sos,
            torch.tensor(enc_ids, dtype=torch.int64),
            self.src_eos,
            torch.tensor([self.src_pad] * enc_pad, dtype=torch.int64)
        ])

        # -------- decoder input (shifted right) --------
        decoder_input = torch.cat([
            self.tgt_sos,
            torch.tensor(dec_ids, dtype=torch.int64),
            torch.tensor([self.tgt_pad] * dec_pad, dtype=torch.int64)
        ])

        # -------- label (shifted left) --------
        label = torch.cat([
            torch.tensor(dec_ids, dtype=torch.int64),
            self.tgt_eos,
            torch.tensor([self.tgt_pad] * dec_pad, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (
                (decoder_input != self.tgt_pad).unsqueeze(0).unsqueeze(0).int()
                & causal_mask(decoder_input.size(0))
            ),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size: int):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).int()
    return mask == 0
