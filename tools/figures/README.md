# Sota figures

Regenerates the three architecture figures in `assets/` (2x PNG).

```
pip install numpy playwright && python -m playwright install chromium
cd tools/figures
python fig1.py   # assets/architecture.png      (overview)
python fig2.py   # assets/conformer-block.png   (inside one Conformer block)
python fig3.py   # assets/training-decoding.png (CTC, joint loss, inference)
```
Fonts needed on your computer: Inter, Noto Sans Khmer, Latin Modern Roman (LM Roman 10).
Edit numbers or labels in the .py files, rerun, then commit the new PNGs.
