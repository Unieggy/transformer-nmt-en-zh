import torch
import gradio as gr
import sentencepiece as spm
from pathlib import Path
from config import get_config
from model import build_transfomer  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def special_ids(sp: spm.SentencePieceProcessor):
    """Return (pad_id, sos_id, eos_id) from a SentencePieceProcessor."""
    pad_id = sp.piece_to_id("[PAD]")
    sos_id = sp.piece_to_id("[SOS]")
    eos_id = sp.piece_to_id("[EOS]")
    return pad_id, sos_id, eos_id


def load(weights_path, src_tok_path, tgt_tok_path):
    """Load model weights and SP tokenizers."""
    cfg = get_config()
    d_model = cfg.get("dmodel", 512)
    seq_len = cfg.get("seq_len", 128)

    # Load SP tokenizers
    src_sp = spm.SentencePieceProcessor()
    tgt_sp = spm.SentencePieceProcessor()
    src_sp.load(src_tok_path)
    tgt_sp.load(tgt_tok_path)

    # Build model
    model = build_transfomer(
        src_vocab_size=src_sp.get_piece_size(),
        tgt_vocab_size=tgt_sp.get_piece_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model
    ).to(DEVICE)

    # Load weights
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, src_sp, tgt_sp


@torch.no_grad()
def greedy(model, src_sp, tgt_sp, text, max_len):
    """Greedy decode with SentencePiece tokenizers."""
    src_pad, src_sos, src_eos = special_ids(src_sp)
    tgt_pad, tgt_sos, tgt_eos = special_ids(tgt_sp)

    # Encode source + EOS
    src_ids = src_sp.encode(text, out_type=int)
    if src_eos != -1:
        src_ids.append(src_eos)
    src = torch.tensor(src_ids, device=DEVICE).unsqueeze(0)

    # Start target with SOS
    tgt = torch.tensor([[tgt_sos]], device=DEVICE)

    for _ in range(max_len - 1):
        out = model(src, tgt)  # adjust if your model requires masks
        next_id = out[:, -1, :].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_id], dim=1)
        if next_id.item() == tgt_eos:
            break

    # Strip SOS/EOS
    ids = tgt.squeeze(0).tolist()
    if ids and ids[0] == tgt_sos:
        ids = ids[1:]
    if tgt_eos in ids:
        ids = ids[:ids.index(tgt_eos)]

    return tgt_sp.decode(ids)


# --------- Gradio UI --------- #
with gr.Blocks() as demo:
    gr.Markdown("# 🈶 NMT Translator (Greedy, SentencePiece)")

    with gr.Row():
        weights = gr.Textbox(label="Weights path", value="weights/tmodel_19.pt")
        src_tok = gr.Textbox(label="Source tokenizer (.model)", value="tokenizers/tokenizer_en.model")
        tgt_tok = gr.Textbox(label="Target tokenizer (.model)", value="tokenizers/tokenizer_zh.model")
        max_len = gr.Slider(16, 256, value=128, step=1, label="Max decode length")

    input_text = gr.Textbox(label="English input", value="The weather is nice today, let's go to the beach.")
    output = gr.Textbox(label="Chinese translation")
    state = gr.State()

    def on_load(w, s, t):
        model, src, tgt = load(w, s, t)
        return (model, src, tgt)

    gr.Button("Load model").click(on_load, [weights, src_tok, tgt_tok], state)

    def on_translate(state_tuple, text, max_len):
        model, src, tgt = state_tuple
        return greedy(model, src, tgt, text, max_len)

    gr.Button("Translate").click(on_translate, [state, input_text, max_len], output)

if __name__ == "__main__":
    demo.launch(share=True)
