"""Mano មនោ detailed architecture (Sota fig1 style) → build/family/Mano-MT-Khmer.png
    uv run --python 3.12 --with numpy --with playwright python mano.py
"""
import math, numpy as np
from common import Fig, SER, SAN, KM, BL, BLs, BT, BTs, GR, GRs, GB, GBs, PU, PUs, PT, PTs, PL, GY, GYs

rng = np.random.default_rng(5)
W, H = 2000, 1230
f = Fig(W, H)
t, R, A = f.t, f.R, f.A
AMB, AMBs, AMT, AMC = "#fffaf0", "#e0b04a", "#fbe7b8", "#a15c07"
iskm = lambda s: any("ក" <= c <= "៿" for c in s)


def tok(x, y, w, h, fill, stroke, label="", size=16):
    R(x, y, w, h, fill, stroke, 1.4, 5)
    if label:
        t(x + w / 2, y + h / 2 + 6, label, size, fam=KM if iskm(label) else SER)


t(40, 60, "Mano មនោ · real-time speech translation · Khmer · English · French · Chinese · Japanese", 30, 600, "#111827", "start", SAN)

# ---------------- input
cy = 390; n = 110; xs = np.linspace(40, 245, n)
env = np.concatenate([np.hanning(45), np.zeros(10), np.hanning(55)])[:n]
for i, x in enumerate(xs):
    v = env[i] * (0.5 + 0.5 * abs(math.sin(i * 0.9))) * (0.6 + 0.4 * rng.random())
    hh = max(2, v * 55)
    f.a(f'<line x1="{x:.1f}" y1="{cy-hh:.1f}" x2="{x:.1f}" y2="{cy+hh:.1f}" stroke="#2f7a24" stroke-width="2"/>')
t(142, 478, "Microphone", 22)
t(142, 503, "16 kHz mono, live", 17, fill="#4b5563")
R(40, 540, 205, 150, "#f8fafc", "#9ca3af", 1.4, 12)
t(142, 570, "user picks", 17, fill="#4b5563", italic=True)
R(58, 584, 168, 36, "#fff", "#9ca3af", 1.2, 8); t(142, 608, "from: ខ្មែរ ▾", 16, fam=KM)
R(58, 630, 168, 36, "#fff", "#9ca3af", 1.2, 8); t(142, 654, "to: English ▾", 16)
t(142, 716, "or auto-detect", 16, fill="#4b5563", italic=True)
A([(252, 440), (282, 440)])

# ---------------- panel 1: streaming front end
x1, w1 = 290, 280
t(x1 + w1 / 2, 118, "1. Listen", 28, 500, "#2f7a24", fam=SAN)
t(x1 + w1 / 2, 150, "(streaming front end)", 20, 400, "#2f7a24", fam=SAN)
R(x1, 175, w1, 760, GR, GRs, 2, 22)
t(x1 + w1 / 2, 222, "0.5 s audio chunks", 19)
for i in range(6):
    tok(x1 + 28 + i * 38, 240, 32, 40, GB, GBs)
t(x1 + w1 / 2, 306, "(B, 8000) each", 17, italic=True)
A([(x1 + w1 / 2, 316), (x1 + w1 / 2, 340)])
f.box(x1 + 25, 345, w1 - 50, 78, GB, GBs, "Silero VAD", "speech prob · 30 ms frames", 20)
A([(x1 + w1 / 2, 427), (x1 + w1 / 2, 452)])
f.box(x1 + 25, 457, w1 - 50, 78, GB, GBs, "Rolling Buffer", "≤ 20 s of the sentence", 20)
A([(x1 + w1 / 2, 539), (x1 + w1 / 2, 564)])
f.box(x1 + 25, 569, w1 - 50, 78, GB, GBs, "Endpoint Rule", "silence ≥ 0.6 s → sentence end", 20)
A([(x1 + w1 / 2, 651), (x1 + w1 / 2, 676)])
f.box(x1 + 25, 681, w1 - 50, 78, GB, GBs, "Language ID", "user pick, or Whisper detect", 20)
t(x1 + w1 / 2, 800, "every 0.5 s: send the", 17, fill="#374151", italic=True)
t(x1 + w1 / 2, 824, "buffer so far to ASR", 17, fill="#374151", italic=True)
t(x1 + w1 / 2, 900, "no training · ≈2M (VAD)", 17, fill="#374151", italic=True)
A([(x1 + w1 + 2, 440), (x1 + w1 + 26, 440)])

# ---------------- panel 2: ASR
x2, w2 = 600, 370
t(x2 + w2 / 2, 118, "2. Speech → Text", 28, 500, "#2448b8", fam=SAN)
t(x2 + w2 / 2, 150, "(ASR, reused models)", 20, 400, "#2448b8", fam=SAN)
R(x2, 175, w2, 760, BL, BLs, 2, 22)
R(x2 + 85, 205, 200, 50, "#fff", BTs, 1.4, 25)
t(x2 + w2 / 2, 237, "language = km ?", 18)
A([(x2 + 120, 258), (x2 + 95, 292)]); A([(x2 + 250, 258), (x2 + 275, 292)])
t(x2 + 78, 284, "yes", 15, fill="#4b5563", italic=True); t(x2 + 296, 284, "no", 15, fill="#4b5563", italic=True)
# Sota
R(x2 + 18, 298, 160, 250, BT, BTs, 1.6, 10)
t(x2 + 98, 330, "Sota សោតៈ", 20, fam=SAN)
for k, s in enumerate(["Conv2D ×4 subsample", "Conformer × 12", "d 256 · 4 heads", "CTC + attn decoder", "3,402 Khmer clusters", "beam 5"]):
    t(x2 + 98, 362 + k * 28, s, 15, fill="#1f2937")
t(x2 + 98, 538, "≈26M · ours", 15, fill="#2448b8", italic=True)
# Whisper
R(x2 + 192, 298, 160, 250, BT, BTs, 1.6, 10)
t(x2 + 272, 324, "Whisper", 20, fam=SAN); t(x2 + 272, 346, "large-v3-turbo", 15, fam=SAN)
for k, s in enumerate(["log-Mel 128", "encoder × 32", "decoder × 4", "d 1280 · 20 heads", "fr · zh · ja · en"]):
    t(x2 + 272, 376 + k * 28, s, 15, fill="#1f2937")
t(x2 + 272, 538, "≈809M · MIT", 15, fill="#2448b8", italic=True)
A([(x2 + 98, 552), (x2 + 98, 578), (x2 + w2 / 2 - 8, 578), (x2 + w2 / 2 - 8, 596)])
A([(x2 + 272, 552), (x2 + 272, 578), (x2 + w2 / 2 + 8, 578), (x2 + w2 / 2 + 8, 596)])
t(x2 + w2 / 2, 622, "Partial transcript (updates every 0.5 s)", 17)
xx = x2 + 22
for s, st in [("ខ្ញុំ", 1), ("សប្បាយ", 1), ("ចិត្ត", 1), ("ណាស់", 0), ("ដែល", 0)]:
    w = 60
    tok(xx, 640, w, 42, PL if st else "#f3f4f6", PTs if st else GYs)
    t(xx + w / 2, 668, s, 16, fill="#1f2937" if st else "#9ca3af", fam=KM)
    xx += w + 6
t(x2 + w2 / 2, 712, "stable  ·  still changing", 15, fill="#4b5563", italic=True)
R(x2 + 25, 735, w2 - 50, 80, "#fff", BTs, 1.4, 10, "6 4")
t(x2 + w2 / 2, 767, "Local agreement", 18, 500, fam=SAN)
t(x2 + w2 / 2, 795, "a word is stable once 2 updates agree", 16, fill="#374151")
t(x2 + w2 / 2, 845, "source text: (B, S) tokens", 17, italic=True)
t(x2 + w2 / 2, 900, "no new ASR training for v1", 17, fill="#374151", italic=True)
A([(x2 + w2 + 2, 440), (x2 + w2 + 26, 440)])

# ---------------- panel 3: Mano translator
x3, w3 = 1000, 650
t(x3 + w3 / 2, 118, "3. Mano Translator", 28, 500, "#6b3fa8", fam=SAN)
t(x3 + w3 / 2, 150, "(Transformer encoder 8 + decoder 4)", 20, 400, "#6b3fa8", fam=SAN)
R(x3, 175, w3, 760, PU, PUs, 2, 22)
# encoder column
ex, ew = x3 + 25, 290
t(ex + ew / 2, 214, "Source tokens + target tag", 17)
tok(ex, 228, 58, 38, GY, GYs, "&lt;2en&gt;", 15)
xx = ex + 64
for s in ["ខ្ញុំ", "សប្បាយ", "ចិត្ត", "⋯"]:
    tok(xx, 228, 52, 38, PL, PTs, s, 15); xx += 57
A([(ex + ew / 2, 272), (ex + ew / 2, 296)])
f.box(ex, 300, ew, 64, PT, PTs, "Shared Embedding", "48k × 512 · + sinusoidal position", 19)
A([(ex + ew / 2, 368), (ex + ew / 2, 392)])
R(ex, 396, ew, 214, PT, PTs, 1.8, 10)
t(ex + ew / 2, 428, "Encoder Layers × 8", 21, fam=SAN)
for k, (c, wd) in enumerate([("LN", 40), ("Self-attn", 92), ("LN", 40), ("FFN", 56)]):
    pass
cx = ex + 16
for k, (c, wd) in enumerate([("LN", 32), ("MHSA", 54), ("⊕", 22), ("LN", 32), ("FFN", 42), ("⊕", 22)]):
    R(cx, 448, wd, 34, PL, PTs, 1.2, 6); t(cx + wd / 2, 471, c, 15)
    cx += wd
    if k < 5:
        A([(cx + 1, 465), (cx + 8, 465)], sw=1.3); cx += 10
t(ex + ew / 2, 512, "pre-LN · 8 heads · d 512", 16, fill="#4b2a85")
t(ex + ew / 2, 538, "FFN 2048 · GELU · dropout 0.1", 16, fill="#4b2a85")
t(ex + ew / 2, 590, "≈25M", 16, fill="#4b2a85", italic=True)
A([(ex + ew / 2, 614), (ex + ew / 2, 640)])
for i in range(5):
    tok(ex + 70 + i * 32, 646, 26, 44, PT, PTs)
t(ex + ew / 2, 716, "memory (B, S, 512)", 17, italic=True)
# decoder column
dx, dw = x3 + 335, 290
t(dx + dw / 2, 214, "Target so far (teacher forcing)", 17)
tok(dx, 228, 48, 38, GY, GYs, "&lt;s&gt;", 15)
xx = dx + 54
for s in ["I", "am", "very", "happy"]:
    w = 22 + 10 * len(s)
    tok(xx, 228, w, 38, "#e8f0fb", BTs, s, 15); xx += w + 5
A([(dx + dw / 2, 272), (dx + dw / 2, 296)])
f.box(dx, 300, dw, 64, PT, PTs, "Shared Embedding", "same table as the encoder", 19)
A([(dx + dw / 2, 368), (dx + dw / 2, 392)])
R(dx, 396, dw, 214, PT, PTs, 1.8, 10)
t(dx + dw / 2, 428, "Decoder Layers × 4", 21, fam=SAN)
cx = dx + 14
for k, (c, wd) in enumerate([("Masked SA", 92), ("Cross-attn", 92), ("FFN", 50)]):
    R(cx, 448, wd, 34, PL, PTs, 1.2, 6); t(cx + wd / 2, 471, c, 15)
    cx += wd
    if k < 2:
        A([(cx + 1, 465), (cx + 12, 465)], sw=1.3); cx += 15
t(dx + dw / 2, 512, "each: pre-LN + residual ⊕", 16, fill="#4b2a85")
t(dx + dw / 2, 538, "8 heads · d 512 · FFN 2048", 16, fill="#4b2a85")
t(dx + dw / 2, 590, "≈17M", 16, fill="#4b2a85", italic=True)
A([(dx + dw / 2, 614), (dx + dw / 2, 638)])
R(dx + 25, 642, dw - 50, 40, PL, PTs, 1.5, 8); t(dx + dw / 2, 668, "LM Head (48k, tied)", 17, fam=SAN)
A([(dx + dw / 2, 686), (dx + dw / 2, 706)])
ax, ay = dx + 40, 790
f.a(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay-72}" stroke="#111" stroke-width="1.5"/>')
A([(ax, ay), (ax + 215, ay)], sw=1.5)
hs = list(rng.random(9) * 25 + 6); hs[4] = 66
bx = ax + 12
for g in range(3):
    for k in range(3):
        hg = hs[g * 3 + k]; f.a(f'<rect x="{bx}" y="{ay-hg}" width="10" height="{hg}" fill="{PT}" stroke="{PTs}"/>'); bx += 12
    if g < 2:
        t(bx + 18, ay - 6, "⋯", 18); bx += 38
t(dx + dw / 2, 814, "next token: “happy”", 16, italic=True)
# cross attention arrow memory -> decoder
A([(ex + ew / 2 + 85, 668), (ex + ew + 12, 668), (ex + ew + 12, 470), (dx + 10, 470)], "#7c4fc0", 2, "7 4")
t(ex + ew / 2, 740, "→ cross-attention in every decoder layer", 15, fill="#6b3fa8", italic=True)
# inference box
R(x3 + 25, 842, w3 - 50, 76, "#fff", PUs, 1.4, 10, "6 4")
t(x3 + w3 / 2, 872, "Inference: beam 4 · length penalty 0.6 · re-translate when the stable text grows", 16, fill="#374151")
t(x3 + w3 / 2, 898, "copy check: numbers and names must appear in the output", 16, fill="#374151")
A([(x3 + w3 + 2, 440), (x3 + w3 + 26, 440)])

# ---------------- panel 4: live subtitles
x4, w4 = 1680, 290
t(x4 + w4 / 2, 118, "4. Show", 28, 500, AMC, fam=SAN)
t(x4 + w4 / 2, 150, "(live subtitles)", 20, 400, AMC, fam=SAN)
R(x4, 175, w4, 760, AMB, AMBs, 2, 22)
R(x4 + 20, 210, w4 - 40, 300, "#111827", "#374151", 2, 14)
t(x4 + w4 / 2, 250, "ខ្មែរ → English", 15, fill="#9ca3af", fam=KM)
t(x4 + w4 / 2, 315, "ខ្ញុំសប្បាយចិត្តណាស់", 17, fill="#9ca3af", fam=KM)
t(x4 + w4 / 2, 340, "ដែលបានជួបអ្នក", 17, fill="#6b7280", fam=KM)
t(x4 + w4 / 2, 400, "I am very happy", 23, 600, "#fff", fam=SAN)
t(x4 + w4 / 2, 432, "to meet you.", 23, 600, "#fff", fam=SAN)
f.a(f'<circle cx="{x4 + w4/2}" cy="{482}" r="14" fill="#ef4444"/>')
t(x4 + w4 / 2, 540, "grey: what was said (live)", 16, fill="#374151", italic=True)
t(x4 + w4 / 2, 564, "white: the translation", 16, fill="#374151", italic=True)
f.box(x4 + 20, 600, w4 - 40, 70, AMT, AMBs, "WebSocket stream", "server → page, each update", 18)
f.box(x4 + 20, 690, w4 - 40, 70, AMT, AMBs, "≈1–2 s behind", "speech → shown (target)", 18)
f.box(x4 + 20, 780, w4 - 40, 70, AMT, AMBs, "Text only", "no voice output (no TTS)", 18)
t(x4 + w4 / 2, 900, "web first, then phone", 17, fill="#374151", italic=True)

# ---------------- training loss + legend
R(260, 965, 1480, 92, "#fffbea", "#a15c07", 1.6, 12, "7 5")
t(1000, 1000, "Training loss:  L = 0.5 · L<tspan baseline-shift='sub' font-size='14'>CE</tspan> (label smoothing 0.1)  +  0.5 · L<tspan baseline-shift='sub' font-size='14'>KD</tspan> (match the teacher MADLAD-400, Apache-2.0)", 20, fill="#7a4505")
t(1000, 1034, "data: open-license sentence pairs in all 20 directions + teacher translations of Khmer text · Khmer test sets never used for training", 17, fill="#7a4505", italic=True)
R(40, 1085, 1920, 118, "none", "#9ca3af", 1.4, 14, "7 5")
lx = 70
for fill, st, lab in [(GB, GBs, "Front end"), (BT, BTs, "ASR"), (PT, PTs, "Mano"), (AMT, AMBs, "Output"), (GY, GYs, "Special token"), (PL, PTs, "Source token")]:
    R(lx, 1106, 28, 28, fill, st, 1.4, 5); t(lx + 38, 1127, lab, 18, anchor="start"); lx += 62 + len(lab) * 10
t(lx + 20, 1118, "&lt;2en&gt; &lt;2km&gt; &lt;2fr&gt; &lt;2zh&gt; &lt;2ja&gt; : target language   ·   &lt;s&gt; : start", 17, anchor="start")
t(lx + 20, 1142, "48k shared SentencePiece vocab (km · en · fr · zh · ja)", 17, anchor="start")
t(70, 1182, "Planned design: sizes are estimates (≈), not measured yet. No results until the model is trained and tested.", 17, fill="#4b5563", anchor="start", fam=SAN, italic=True)
t(1930, 1182, "Mano មនោ · translator ≈67M params (+ reused ASR)", 19, 600, "#374151", "end", SAN)

import os
os.makedirs("build/family", exist_ok=True)
f.save("build/family/Mano-MT-Khmer.png", "build/Mano-MT-Khmer.html")
print("ok")
