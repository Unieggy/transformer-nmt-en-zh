# app.py — EN → ZH translator (SentencePiece + Gradio, Colab-aware)

import os
from pathlib import Path
import torch
import gradio as gr
import sentencepiece as spm
from config import get_config
try:
    from model import build_transformer as _build
except Exception:
    from model import build_transfomer as _build  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- helpers ---------------- #

def sp_load(path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    ok = sp.load(path)
    if not ok:
        raise FileNotFoundError(f"Failed to load SentencePiece model: {path}")
    return sp

def sp_ids(sp: spm.SentencePieceProcessor):
    """Return (pad_id, sos_id, eos_id); -1 if token not present in vocab."""
    return (
        sp.piece_to_id("[PAD]"),
        sp.piece_to_id("[SOS]"),
        sp.piece_to_id("[EOS]"),
    )

def strip_special(ids, sos_id: int, eos_id: int):
    out = list(ids)
    if out and sos_id >= 0 and out[0] == sos_id:
        out = out[1:]
    if eos_id >= 0 and eos_id in out:
        out = out[: out.index(eos_id)]
    return out


def load(weights_path: str, src_spm_path: str, tgt_spm_path: str):
    """Build model, load weights, and load tokenizers. Returns (model, src_sp, tgt_sp)."""
    cfg = get_config()
    d_model = int(cfg.get("dmodel", 512))
    seq_len = int(cfg.get("seq_len", 128))

    # tokenizers
    src_sp = sp_load(src_spm_path)
    tgt_sp = sp_load(tgt_spm_path)

    # model
    model = _build(
        src_vocab_size=src_sp.get_piece_size(),
        tgt_vocab_size=tgt_sp.get_piece_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
    ).to(DEVICE)

    # weights
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {missing}\n[warn] unexpected keys: {unexpected}")

    model.eval()
    return model, src_sp, tgt_sp


@torch.no_grad()
@torch.no_grad()
def greedy(model, src_sp, tgt_sp, text: str, max_len: int):
    # ----- special token ids -----
    src_pad, src_sos, src_eos = (src_sp.piece_to_id("[PAD]"),
                                 src_sp.piece_to_id("[SOS]"),
                                 src_sp.piece_to_id("[EOS]"))
    tgt_pad, tgt_sos, tgt_eos = (tgt_sp.piece_to_id("[PAD]"),
                                 tgt_sp.piece_to_id("[SOS]"),
                                 tgt_sp.piece_to_id("[EOS]"))

    def enc_pad_mask(x, pad_id):
        # (B, 1, 1, S) True=keep, False=mask-out
        if pad_id is None or pad_id < 0:
            return torch.ones((x.size(0), 1, 1, x.size(1)), dtype=torch.bool, device=x.device)
        return (x != pad_id).unsqueeze(1).unsqueeze(1)

    def causal_mask(T):
        # (1, 1, T, T) lower triangular
        m = torch.tril(torch.ones((T, T), dtype=torch.bool, device=DEVICE))
        return m.unsqueeze(0).unsqueeze(0)

    def dec_mask(y, pad_id):
        # (B, 1, T, T) = causal AND not-pad at key positions
        T = y.size(1)
        cm = causal_mask(T)                                   # (1,1,T,T)
        if pad_id is None or pad_id < 0:
            return cm.expand(y.size(0), -1, -1, -1)
        key_ok = (y != pad_id).unsqueeze(1).unsqueeze(2)      # (B,1,1,T)
        return cm & key_ok                                    # broadcast to (B,1,T,T)

    # ----- encode source -----
    src_ids = src_sp.encode(text, out_type=int)
    if src_eos is not None and src_eos >= 0:
        src_ids.append(src_eos)
    src = torch.tensor(src_ids, device=DEVICE, dtype=torch.long).unsqueeze(0)  # (1,S)

    src_mask = enc_pad_mask(src, src_pad)                      # (1,1,1,S)
    memory = model.encode(src, src_mask)                       # (1,S,d)

    # ----- start target -----
    if tgt_sos is None or tgt_sos < 0:
        tgt = torch.empty((1, 0), device=DEVICE, dtype=torch.long)
    else:
        tgt = torch.tensor([[tgt_sos]], device=DEVICE, dtype=torch.long)

    steps = max(1, int(max_len))
    for _ in range(steps):
        tgt_mask = dec_mask(tgt, tgt_pad)                      # (1,1,T,T)
        dec_out = model.decode(memory, src_mask, tgt, tgt_mask) # (1,T,d)
        logits = model.projection(dec_out)                      # (1,T,V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True) # (1,1)
        tgt = torch.cat([tgt, next_id], dim=1)

        if tgt_eos is not None and tgt_eos >= 0 and next_id.item() == tgt_eos:
            break
        if tgt.size(1) >= max_len:
            break

    # strip SOS/EOS and decode
    out_ids = tgt.squeeze(0).tolist()
    if tgt_sos is not None and tgt_sos >= 0 and out_ids and out_ids[0] == tgt_sos:
        out_ids = out_ids[1:]
    if tgt_eos is not None and tgt_eos >= 0 and tgt_eos in out_ids:
        out_ids = out_ids[: out_ids.index(tgt_eos)]
    return tgt_sp.decode(out_ids)



# ---------------- UI (Colab-aware defaults) ---------------- #

def _choose_default(local_path: str, colab_path: str) -> str:
    """Prefer Colab path if it exists, else local path."""
    return colab_path if os.path.exists(colab_path) else local_path

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("##  EN → ZH NMT (Greedy · SentencePiece)")

    # set your local defaults here (when running on your own machine)
    local_weights = "weights/tmodel_19.pt"
    local_src_tok = "tokenizers/tokenizer_en.model"
    local_tgt_tok = "tokenizers/tokenizer_zh.model"

    # and Colab (Drive) defaults here
    colab_weights = "/content/drive/MyDrive/weights/tmodel_19.pt"
    colab_src_tok = "/content/drive/MyDrive/tokenizer_en.model"
    colab_tgt_tok = "/content/drive/MyDrive/tokenizer_zh.model"

    weights_default = _choose_default(local_weights, colab_weights)
    src_tok_default = _choose_default(local_src_tok, colab_src_tok)
    tgt_tok_default = _choose_default(local_tgt_tok, colab_tgt_tok)

    with gr.Row():
        weights_tb = gr.Textbox(label="Weights (.pt)", value=weights_default)
        src_tok_tb = gr.Textbox(label="Source tokenizer (.model, EN)", value=src_tok_default)
        tgt_tok_tb = gr.Textbox(label="Target tokenizer (.model, ZH)", value=tgt_tok_default)
        max_len = gr.Slider(16, 512, value=128, step=1, label="Max decode length")

    input_en = gr.Textbox(
        label="English input",
        value="The weather is nice today; shall we take a walk by the beach?",
        lines=3,
    )
    output_zh = gr.Textbox(label="Chinese translation", lines=3)
    status = gr.Textbox(label="Status", interactive=False)

    state = gr.State(value=None)
    load_btn = gr.Button("Load model")
    translate_btn = gr.Button("Translate", variant="primary", interactive=False)

    def on_load(w, s, t):
        import traceback
        # show missing paths clearly
        missing = [p for p in (w, s, t) if not Path(p).exists()]
        if missing:
            msg = "❌ Missing paths:\n" + "\n".join(f"- {p}" for p in missing)
            print(msg)
            return None, gr.update(value=msg), gr.update(interactive=False)
        try:
            model, sp_src, sp_tgt = load(w, s, t)
            print("✅ Loaded:", w, s, t)
            return (model, sp_src, sp_tgt), gr.update(value="✅ Model loaded"), gr.update(interactive=True)
        except Exception:
            err = "❌ Load failed:\n" + traceback.format_exc()
            print(err)
            return None, gr.update(value=err), gr.update(interactive=False)

    load_btn.click(
        fn=on_load,
        inputs=[weights_tb, src_tok_tb, tgt_tok_tb],
        outputs=[state, status, translate_btn],
    )

    def on_translate(state_tuple, text, mlen):
        import traceback
        if not state_tuple:
            return "Please click ‘Load model’ first."
        try:
            model, sp_src, sp_tgt = state_tuple
            return greedy(model, sp_src, sp_tgt, text, int(mlen))
        except Exception:
            err = "❌ Translate failed:\n" + traceback.format_exc()
            print(err)
            return err

    translate_btn.click(on_translate, [state, input_en, max_len], outputs=output_zh)

    # auto-load if all guessed defaults exist
    def _maybe_autoload(w, s, t):
        if all(Path(p).exists() for p in (w, s, t)):
            st, msg, btn = on_load(w, s, t)
            return st, msg, btn
        return None, gr.update(value="ℹ️ Set paths above, then click ‘Load model’."), gr.update(interactive=False)

    demo.load(_maybe_autoload, [weights_tb, src_tok_tb, tgt_tok_tb], [state, status, translate_btn])

if __name__ == "__main__":
    # share=True is required to get a public link on Colab
    demo.launch(share=True)
