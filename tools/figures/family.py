"""Planned-architecture figures for the new family models (Tossana style) → build/family/<Repo>.png

    python family.py            # all
    python family.py Mano-MT-Khmer
"""
import sys
from common import Fig, SAN, KM, BL, BLs, BT, BTs, GR, GRs, GB, GBs, PU, PUs, PT, PTs, GY, GYs

AMB, AMBs, AMT = "#fffaf0", "#e0b04a", "#fbe7b8"
COL = {"blue": (BL, BLs, BT, BTs, "#2448b8"), "green": (GR, GRs, GB, GBs, "#2f7a24"),
       "purple": (PU, PUs, PT, PTs, "#6b3fa8"), "amber": (AMB, AMBs, AMT, AMBs, "#a15c07")}

# Each model: title, input (label, lines), panels [color, title, subtitle, boxes [(label, sub)], foot], chips, params
SPECS = {
 "Mano-MT-Khmer": dict(
    name="Mano មនោ", what="real-time speech translation · km · en · fr · zh · ja", params="≈65M",
    inp=("Microphone", ["16 kHz live audio", "user picks from → to", "ខ្មែរ → English", "or auto-detect"]),
    panels=[
     ("green", "1. Listen", "(voice detector + chunks)", [("Silero VAD", "speech / silence · 30 ms frames"), ("0.5 s Audio Chunks", "streamed while they talk"), ("Language ID", "the user's pick, or Whisper detects it")], "sentence ends after ≈0.6 s of silence"),
     ("blue", "2. Speech → Text", "(ASR, reused)", [("Sota សោតៈ", "Khmer · Conformer CTC + attention"), ("Whisper large-v3-turbo", "French · Chinese · Japanese · English"), ("Partial Transcript", "updates on every chunk")], "no new ASR training for v1"),
     ("purple", "3. Translate", "(Mano · Transformer enc-dec)", [("Target Tag First", "&lt;2km&gt; &lt;2en&gt; &lt;2fr&gt; &lt;2zh&gt; &lt;2ja&gt;"), ("Encoder × 8 · Decoder × 4", "d = 512 · 8 heads · shared 48k vocab"), ("Distilled from MADLAD-400", "Apache-2.0 teacher → small Mano")], "≈65M · beam 4 · re-translates as text grows"),
     ("amber", "4. Show", "(live subtitles)", [("ខ្ញុំសប្បាយចិត្តណាស់", "what was said · grey, live"), ("I am very happy", "the translation · big"), ("Stable Prefix", "words stop changing once agreed")], "web + phone · ≈1–2 s behind speech")],
    chips=["any pair of km · en · fr · zh · ja", "ASR reused: Sota + Whisper (MIT)", "open-license data + teacher only", "text only · no TTS · ONNX int8"]),
 "Jivha-Pronunciation-Khmer": dict(
    name="Jivhā ជិវ្ហា", what="Khmer pronunciation coach", params="≈20M",
    inp=("Learner reads aloud", ["expected text:", "ស្រុកខ្មែរ", "+ 16 kHz audio", "from the phone mic"]),
    panels=[
     ("blue", "1. Acoustic Encoder", "(Sota encoder, reused)", [("Log-Mel 80 + Conv2D ×4", "40 ms frames"), ("Conformer Blocks × 12", "d = 256 · 4 heads"), ("CTC Posteriors", "(T, 3,402 clusters + blank)")], "fine-tuned from Sota · ≈20M"),
     ("green", "2. Align + Score", "(goodness of pronunciation)", [("Forced Alignment", "Viterbi on the expected clusters"), ("GOP per cluster", "log P(expected) − log max P(any)"), ("Scoring Head", "2-layer MLP · ≈0.1M → ok / wrong"), ("Heard-as", "top competing cluster, e.g. ស instead of ស្រ")], "trained on learner clips + simulated mistakes"),
     ("amber", "3. Feedback", "(app)", [("ស្រុក ខ្មែរ", "wrong clusters in red"), ("\"ស្រ → you said ស\"", "short tip per mistake"), ("Play the right sound", "VoxPocket TTS"), ("Score 0–100", "per word and per sentence")], "ONNX Runtime · phone · offline")],
    chips=["reuses the Sota encoder", "per-cluster scores, not just pass/fail", "simulated mistakes from TTS + edits", "int8 ONNX · on the phone"]),
 "Sodhana-Spell-Khmer": dict(
    name="Sodhana សោធន៍", what="Khmer spell checker + autocomplete", params="≈7M",
    inp=("Typed text", ["ខ្ញុំទៅសាលារៀន", "(with typos)", "from the Akkhara", "keyboard"]),
    panels=[
     ("blue", "1. Encoder", "(Transformer encoder × 4)", [("Cluster Embedding", "3,402 Khmer clusters"), ("Encoder Layers × 4", "self-attn · FFN"), ("Error Detector", "per cluster: ok / wrong")], "d = 256 · 4 heads · ≈4M"),
     ("purple", "2. Corrector", "(Transformer decoder × 2)", [("Decoder Layers × 2", "cross-attn to the typed text"), ("Copy-biased Beam", "keeps correct words unchanged"), ("Next-word Mode", "same decoder → autocomplete")], "d = 256 · ≈3M"),
     ("amber", "3. Output", "(in the keyboard)", [("Dictionary Check", "44k-word lexicon"), ("Underline + 3 suggestions", "tap to replace"), ("Next-word chips", "above the keys")], "ONNX Runtime · phone · offline")],
    chips=["training errors: coeng swaps, vowel order, ឲ/ឱ, key neighbours", "never changes correct text", "same tokens as Chakkhu & Sota", "int8 ONNX ≈7 MB"]),
 "Vacana-TextNorm-Khmer": dict(
    name="Vacana វចនៈ", what="Khmer text normalization", params="≈4M",
    inp=("Two directions", ["for speech:", "២៥ ដុល្លារ", "for reading:", "Sota output (no punct.)"]),
    panels=[
     ("green", "1. Words for speech", "(text → spoken words)", [("Rule Speller", "numbers · dates · money · units"), ("Ambiguity Tagger", "year / phone no. / amount"), ("២៥ ដុល្លារ → ម្ភៃប្រាំដុល្លារ", "feeds VoxPocket")], "rules from the Sota data tools"),
     ("blue", "2. Tagger", "(Transformer encoder × 4)", [("Cluster Embedding", "3,402 clusters + Latin"), ("Encoder Layers × 4", "d = 256 · 4 heads"), ("Token Heads", "punctuation · digit span · English case")], "shared by both directions · ≈4M"),
     ("amber", "3. Text for reading", "(spoken words → text)", [("Insert ។ ? ,", "from the punctuation head"), ("Words → digits", "ម្ភៃប្រាំដុល្លារ → ២៥ ដុល្លារ"), ("Readable transcript", "for Sota output")], "ONNX Runtime · CPU · offline")],
    chips=["one small tagger, two jobs", "rules first, model only for ambiguity", "Sota → readable text", "int8 ONNX ≈4 MB"]),
 "Sati-KWS-Khmer": dict(
    name="Sati សតិ", what="Khmer wake word + voice commands", params="≈0.2M",
    inp=("Live microphone", ["16 kHz mono", "1 s sliding window", "every 100 ms", "always on"]),
    panels=[
     ("blue", "1. Features", "(tiny front end)", [("Log-Mel 40", "25 ms window · 10 ms hop"), ("Per-channel norm", "PCEN, robust to loudness")], "same audio pipeline as Sota"),
     ("green", "2. DS-CNN", "(depthwise-separable CNN)", [("Conv 3×3 · 64 ch", "stride 2"), ("DS Blocks × 5", "depthwise 3×3 + pointwise 1×1"), ("Global Avg Pool", "(B, 64)"), ("Softmax", "wake word · ≈30 commands · unknown · silence")], "≈0.2M params · ≈1 ms per window"),
     ("amber", "3. Decision", "(smoothing)", [("Posterior Smoothing", "average over 5 windows"), ("Threshold + Cooldown", "≈1 false wake per day target"), ("\"បើកភ្លើង\" → turn_on_light", "command id to the app")], "ONNX Runtime · phone / MCU")],
    chips=["real clips + VoxPocket voices", "noise + room + phone augmentation", "unknown-word class against false wakes", "int8 ≈0.2 MB"]),
 "Nama-Diarization-Khmer": dict(
    name="Nāma នាម", what="Khmer speaker diarization", params="≈6M",
    inp=("Long recording", ["meeting · call", "podcast", "16 kHz mono", "any length"]),
    panels=[
     ("blue", "1. Speech Detection", "(VAD)", [("Voice Activity Detector", "small CNN · 10 ms frames"), ("Speech Segments", "drop silence and music")], "≈0.3M"),
     ("green", "2. Speaker Embeddings", "(ECAPA-TDNN, small)", [("1.5 s windows", "hop 0.75 s"), ("SE-Res2Net Blocks × 3", "channels 512"), ("Attentive Stat Pool", "→ 192-dim embedding")], "≈6M · trained on many Khmer voices"),
     ("purple", "3. Clustering", "(who is who)", [("Cosine Similarity", "between all windows"), ("Spectral Clustering", "number of speakers found automatically"), ("Resegmentation", "smooth speaker changes")], "no training needed"),
     ("amber", "4. Output", "(with Sota)", [("Speaker 1 · 0:00–0:12", "ជម្រាបសួរ ..."), ("Speaker 2 · 0:12–0:20", "បាទ ..."), ("who said what", "Sota timestamps + Nāma speakers")], "ONNX Runtime · CPU · offline")],
    chips=["works with any number of speakers", "pairs with Sota timestamps", "helps label new podcast data", "int8 ONNX ≈6 MB"]),
 "Lekha-HTR-Khmer": dict(
    name="Lekhā លេខា", what="Khmer handwriting recognition", params="≈14M",
    inp=("Handwritten line", ["grayscale photo", "resized to 64 × W", "(from Chakkhu's", "line detector)"]),
    panels=[
     ("blue", "1. Visual Encoder", "(CNN + Transformer)", [("CNN Stem", "6 residual blocks → (W/4, 256)"), ("Transformer Layers × 6", "d = 256 · 4 heads"), ("Line Features", "(B, W/4, 256)")], "≈8M"),
     ("green", "2. CTC Head", "(fast draft)", [("Linear → 3,402 + blank", "per column"), ("Greedy Draft", "cluster count + text")], "≈0.9M"),
     ("purple", "3. Attention Decoder", "(Transformer decoder × 3)", [("Decoder Layers × 3", "cross-attn to line features"), ("LM Head", "3,402 Khmer clusters"), ("Beam 5 · CTC + decoder score", "like Sota")], "≈5M"),
     ("amber", "4. Output", "(in Chakkhu)", [("Recognized text", "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា"), ("Confidence per word", "low ones highlighted")], "ONNX Runtime · phone · offline")],
    chips=["real lines + synthetic handwriting fonts", "elastic, pen-width and paper augmentation", "same tokens & decoding as Sota", "int8 ONNX ≈14 MB"]),
}


def draw(repo, s, out):
    W = 2000
    maxb = max(len(b[3]) for b in s["panels"])
    PB = 250 + (maxb - 1) * 140 + 92 + 75          # panel bottom
    MY = (120 + PB) // 2                            # arrow height
    H = PB + 230
    f = Fig(W, H)
    f.t(40, 72, f"{s['name']} · {s['what']} · {s['params']} parameters (planned)", 38, 600, "#111827", "start", SAN)
    # input column
    lab, lines = s["inp"]
    f.R(30, MY - 130, 210, 260, "#f8fafc", "#9ca3af", 1.6, 14)
    for i, l in enumerate(lines):
        km = any("\u1780" <= ch <= "\u17ff" for ch in l)
        f.t(135, MY - 65 + i * 48, l, 20 if i % 2 == 0 else 17, fill="#1f2937" if i % 2 == 0 else "#4b5563", fam=KM if km else SAN, italic=i % 2 == 1)
    f.t(135, MY + 170, lab, 22, fam=SAN)
    n = len(s["panels"])
    gap = 40
    x0 = 280
    pw = (W - 30 - x0 - gap * (n - 1)) / n
    f.A([(245, MY), (x0 - 6, MY)])
    for i, (col, title, sub, boxes, foot) in enumerate(s["panels"]):
        bg, bs, bt, bts, tc = COL[col]
        x = x0 + i * (pw + gap)
        f.R(x, 120, pw, PB - 120, bg, bs, 2, 22)
        f.t(x + pw / 2, 172, title, 28, 600, tc, fam=SAN)
        f.t(x + pw / 2, 206, sub, 20, 400, tc, fam=SAN, italic=True)
        bh = 92
        top = 250
        step = 140
        for j, (l1, l2) in enumerate(boxes):
            y = top + j * step
            f.R(x + 24, y, pw - 48, bh, bt if col != "amber" else "#fff", bts, 1.5, 10)
            f.t(x + pw / 2, y + 40, l1, 21, fam=KM if any("ក" <= ch <= "៿" for ch in l1) else SAN)
            f.t(x + pw / 2, y + 70, l2, 16, fill="#374151", fam=KM if any("ក" <= ch <= "៿" for ch in l2) else SAN, italic=True)
            if j < len(boxes) - 1:
                f.A([(x + pw / 2, y + bh + 2), (x + pw / 2, y + step - 4)], sw=1.8)
        f.t(x + pw / 2, PB - 28, foot, 17, fill="#374151", fam=SAN, italic=True)
        if i < n - 1:
            f.A([(x + pw + 2, MY), (x + pw + gap - 4, MY)])
    # chips
    cw = (W - 80 - 3 * 20) / 4
    dots = ["#2f5fd0", "#e0a020", "#2fa36b", "#8b5cf6"]
    for i, c in enumerate(s["chips"]):
        x = 40 + i * (cw + 20)
        f.R(x, PB + 40, cw, 62, "#fff", "#d1d5db", 1.4, 31)
        f.a(f'<circle cx="{x + 30}" cy="{PB + 71}" r="8" fill="{dots[i]}"/>')
        f.t(x + cw / 2 + 12, PB + 78, c, 17 if len(c) < 40 else 15, fam=KM if any("ក" <= ch <= "៿" for ch in c) else SAN)
    f.R(40, PB + 130, W - 80, 80, "none", "#9ca3af", 1.4, 14, "7 5")
    f.t(70, PB + 178, "Planned design: sizes are estimates (≈), not measured yet. No results until the model is trained and tested.", 19, fill="#4b5563", anchor="start", fam=SAN, italic=True)
    f.t(W - 70, PB + 178, f"{s['name']} · {s['params']} params", 20, 600, "#374151", "end", SAN)
    f.save(out, f"build/{repo}.html")


if __name__ == "__main__":
    import os
    os.makedirs("build/family", exist_ok=True)
    for repo in sys.argv[1:] or SPECS:
        draw(repo, SPECS[repo], f"build/family/{repo}.png")
        print("ok", repo)
