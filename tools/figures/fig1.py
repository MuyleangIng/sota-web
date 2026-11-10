import numpy as np, math
rng=np.random.default_rng(3)
W,H=2000,1160
o=[];a=o.append
SER="'LM Roman 10','Latin Modern Roman','Noto Sans Khmer',serif"
SAN="Inter,'Noto Sans Khmer',sans-serif"
def t(x,y,s,size=20,w=400,fill="#1f2937",anchor="middle",fam=SER,italic=False):
    st=' font-style="italic"' if italic else ''
    a(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')
def R(x,y,w,h,fill,stroke,sw=1.5,rx=8,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def A(pts,c="#111827",sw=2,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    a('<polyline points="'+" ".join(f"{x},{y}" for x,y in pts)+f'" fill="none" stroke="{c}" stroke-width="{sw}"{d} marker-end="url(#ah)" style="color:{c}"/>')
BL,BLs,BT,BTs="#edf3fc","#6f9be6","#bcd4f6","#4a7fd8"
GR,GRs,GB,GBs="#eef7ea","#8cc483","#c9e6be","#5aa04e"
PU,PUs,PT,PTs,PL="#f7f2fc","#b79bd6","#dccaf0","#9b74c9","#efe6f8"
GY,GYs="#eceef1","#9ca3af"
def tok(x,y,w,h,fill,stroke,label="",size=17,km=False):
    R(x,y,w,h,fill,stroke,1.4,5)
    if label: t(x+w/2,y+h/2+6,label,size)
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
a('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L9 5L1 9z" fill="currentColor"/></marker></defs>')
a(f'<rect width="{W}" height="{H}" fill="#fff"/>')

# ---- inputs
cy=380; n=120; xs=np.linspace(40,250,n)
env=np.concatenate([np.hanning(50),np.zeros(10),np.hanning(60)])[:n]
for i,x in enumerate(xs):
    v=env[i]*(0.5+0.5*abs(math.sin(i*0.9)))*(0.6+0.4*rng.random())
    hh=max(2,v*60); a(f'<line x1="{x:.1f}" y1="{cy-hh:.1f}" x2="{x:.1f}" y2="{cy+hh:.1f}" stroke="#2f5fc4" stroke-width="2"/>')
t(145,475,"Audio Waveform",22)
t(145,500,"16 kHz mono",18,fill="#4b5563")
inf=[(0,0,4),(87,16,110),(188,55,84),(249,142,9),(252,255,164)]
def cm(v):
    v=min(max(v,0),.999)*(len(inf)-1); i=int(v); f=v-i
    return "#%02x%02x%02x"%tuple(int(inf[i][k]*(1-f)+inf[i+1][k]*f) for k in range(3))
sx,sy,sw,sh,C,Rr=40,540,210,130,42,22
envs=np.interp(np.linspace(0,1,C),np.linspace(0,1,n),env)
for c in range(C):
    f0=3.2+1.5*math.sin(c*.3)
    for r in range(Rr):
        fr=Rr-1-r
        v=envs[c]*(.6*math.exp(-fr/8)+.4*(.5+.5*math.cos(2*math.pi*fr/f0))*math.exp(-fr/14))+.05+.06*rng.random()
        a(f'<rect x="{sx+c*sw/C:.2f}" y="{sy+r*sh/Rr:.2f}" width="{sw/C+.4:.2f}" height="{sh/Rr+.4:.2f}" fill="{cm(v*1.2)}"/>')
t(145,705,"Log-Mel Spectrogram",22)
t(145,733,"(B, T, 80)",20,italic=True)
A([(265,520),(300,520)])

# ---- panel 1
t(560,110,"1. Acoustic Encoder",30,500,"#2448b8",fam=SAN)
t(560,146,"(Conv2D Subsampling + Conformer)",24,400,"#2448b8",fam=SAN)
R(310,175,520,730,BL,BLs,2,22)
t(415,345,"2× Conv2D",21); t(415,372,"Subsample (×4)",21)
def slab(x,y,w,h,d):
    a(f'<polygon points="{x},{y} {x+d},{y-d} {x+w+d},{y-d} {x+w},{y}" fill="#dce8fa" stroke="{BTs}" stroke-width="1.2"/>')
    a(f'<polygon points="{x+w},{y} {x+w+d},{y-d} {x+w+d},{y+h-d} {x+w},{y+h}" fill="#a9c3ec" stroke="{BTs}" stroke-width="1.2"/>')
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{BT}" stroke="{BTs}" stroke-width="1.2"/>')
slab(340,440,40,150,26); t(412,520,"⋯",22); slab(430,440,40,150,26)
A([(500,515),(528,515)])
t(675,240,"Conformer Blocks",21); t(675,267,"× 12",21,italic=True)
R(538,290,272,370,"none",BTs,1.5,14,"7 5")
for x in [562,622,752]:
    R(x,315,44,320,BT,BTs,1.4,8)
t(694,480,"⋯",24)
R(335,690,470,195,"#f8fbff",BTs,1.4,14,"7 5")
t(570,722,"One block (macaron style)",19)
chips=[("½ FFN",85),("MHSA",72),("Conv",62),("½ FFN",85),("LN",44)]
x=355
for i,(c,wd) in enumerate(chips):
    R(x,742,wd,40,BT,BTs,1.2,6); t(x+wd/2,769,c,18)
    x+=wd
    if i<4: A([(x+2,762),(x+15,762)],sw=1.5); x+=18
t(570,818,"rel-pos self-attention · depthwise conv (k = 31)",17,fill="#374151")
t(570,846,"d = 256 · 4 heads · FFN 1024 · residual ⊕",17,fill="#374151")
t(570,872,"≈20M params (incl. subsampling)",17,fill="#374151",italic=True)

# encoder output tokens
A([(830,465),(865,465),(865,505)])
for i in range(4): R(845+i*30,515,24,48,BT,BTs,1.3,4)
t(905,598,"(B, T/4, 256)",19,italic=True)
A([(905,512),(905,440),(990,440)])

# ---- panel 2 CTC
t(1135,110,"2. CTC Head",30,500,"#2f7d23",fam=SAN)
t(1135,146,"(frame-level)",24,400,"#2f7d23",fam=SAN)
R(1000,175,280,730,GR,GRs,2,22)
t(1140,230,"Encoder frames",20)
for i in range(6): R(1043+i*33,262,26,42,BT,BTs,1.3,4)
A([(1140,312),(1140,340)])
R(1030,345,220,80,GB,GBs,1.8,10)
t(1140,380,"Linear + Softmax",22,fill="#1f4f17",fam=SAN)
t(1140,405,"V + 1 outputs",17,fill="#2f6b24")
A([(1140,430),(1140,465)])
t(1140,462+0,"",1)
labs=["ε","សួ","សួ","ε","ស្តី","␣","h"]
for i,l in enumerate(labs):
    tok(1022+i*34,472,30,40,"#e4f2dd",GBs)
    t(1037+i*34,499,l,15,fam="'Noto Sans Khmer',serif" if l not in "εh␣" else SER)
t(1140,540,"best label per frame",17,fill="#374151",italic=True)
A([(1140,552),(1140,585)])
t(1140,612,"collapse repeats, drop ε",18)
for i,l in enumerate(["សួ","ស្តី","␣","h","⋯"]):
    tok(1055+i*34,632,30,40,PL,PTs)
    t(1070+i*34,659,l,15,fam="'Noto Sans Khmer',serif" if l not in "␣h⋯" else SER)
t(1140,712,"(B, T/4, V+1)",19,italic=True)
R(1050,745,180,80,"#fff","#a15c07",1.5,10,"6 4")
t(1140,780,"Loss",21); t(1140,806,"(CTC)",19)
t(1140,860,"fast greedy draft",17,fill="#374151",italic=True)
t(1140,882,"+ cluster count",17,fill="#374151",italic=True)

# ---- panel 3 decoder
t(1640,110,"3. Attention Decoder",30,500,"#6b3fb0",fam=SAN)
t(1640,146,"(Transformer Decoder × 3)",24,400,"#6b3fb0",fam=SAN)
R(1310,175,660,730,PU,PUs,2,22)
t(1640,222,"Input Cluster Sequence (teacher forcing)",19)
tok(1345,240,74,40,GY,GYs,"&lt;sos&gt;",16)
x=1432
for l in ["សួ","ស្តី","␣","h","⋯","o"]:
    tok(x,240,44,40,PL,PTs); t(x+22,267,l,16,fam="'Noto Sans Khmer',serif" if l not in "␣h⋯o" else SER); x+=52
R(1345,300,450,46,PT,PTs,1.5,8); t(1570,330,"Token Embedding",21,fam=SAN)
A([(1570,350),(1570,385)])
R(1345,390,450,100,PT,PTs,1.8,10)
t(1570,428,"Decoder Layers × 3",22,fam=SAN)
t(1570,458,"masked self-attn · cross-attn · FFN",17,fill="#4b2a85")
A([(1570,494),(1570,522)])
R(1400,526,340,40,PL,PTs,1.5,8); t(1570,553,"LM Head (3,402 clusters)",18,fam=SAN)
A([(1570,570),(1570,598)])
t(1570,622,"Next Cluster Logits",19)
ax,ay=1405,730
a(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay-95}" stroke="#111" stroke-width="1.6"/>')
A([(ax,ay),(ax+335,ay)],sw=1.6)
hs=list(rng.random(9)*35+8); hs[4]=88
bx=ax+15
for g in range(3):
    for k in range(3):
        hgt=hs[g*3+k]; a(f'<rect x="{bx}" y="{ay-hgt}" width="11" height="{hgt}" fill="{PT}" stroke="{PTs}"/>'); bx+=13
    if g<2: t(bx+28,ay-8,"⋯",20); bx+=62
t(1570,758,"Cluster vocabulary",18)
R(1350,785,440,100,"#fff",PTs,1.4,10,"6 4")
t(1570,818,"Inference",19,500,fam=SAN)
t(1570,846,"beam search (5) · CTC + decoder scores",17,fill="#374151")
t(1570,870,"runaway guard → fall back to CTC",17,fill="#374151")
# training target
R(1812,390,140,345,"#fff",PUs,1.4,12,"6 4")
t(1882,422,"Training",19); t(1882,446,"Target",19)
yy=470
for l in ["សួ","ស្តី","⋯","&lt;/s&gt;"]:
    tok(1847,yy,70,36,GY if "s&gt;" in l else PL,GYs if "s&gt;" in l else PTs)
    t(1882,yy+24,l,15,fam="'Noto Sans Khmer',serif"); yy+=44
A([(1882,650),(1882,672)],sw=1.5,dash="4 3")
t(1882,698,"Loss (CE)",18)
# cross-attention route
A([(905,600+12),(905,935),(1328,935),(1328,440),(1343,440)],"#4a7fd8",2.2,"8 5")
t(1115,960,"encoder memory → cross-attention",18,fill="#2448b8",italic=True)

# joint loss
R(640,990,720,58,"#fffbea","#a15c07",1.6,12,"7 5")
t(1000,1027,"Joint training loss:  L = 0.3 · L<tspan baseline-shift='sub' font-size='15'>CTC</tspan> + 0.7 · L<tspan baseline-shift='sub' font-size='15'>CE</tspan>",22,fill="#7a4505")

# legend
R(60,1075,1880,70,"none","#9ca3af",1.4,14,"7 5")
lx=90
for fill,st,lab in [(BT,BTs,"Audio / encoder"),(GB,GBs,"CTC head"),(PT,PTs,"Decoder"),(GY,GYs,"Special token"),(PL,PTs,"Khmer cluster token")]:
    R(lx,1094,30,30,fill,st,1.4,5); t(lx+42,1116,lab,19,anchor="start"); lx+=60+len(lab)*10
t(lx+20,1104,"ε : CTC blank   ·   &lt;sos&gt; / &lt;/s&gt; : start / end",18,anchor="start")
t(lx+20,1130,"V = 3,402 Khmer clusters (shared with Chakkhu)",18,anchor="start")
t(1915,1116,"Sota សោតៈ · ≈26M params",19,500,"#374151","end",fam=SAN)
a('</svg>')
svg="\n".join(o)
import os; os.makedirs("build",exist_ok=True); open("build/fig1.html","w").write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{{margin:0}}</style></head><body>{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":W,"height":H},device_scale_factor=2)
    pg.goto("file://"+os.path.abspath("build/fig1.html")); pg.wait_for_timeout(600)
    pg.screenshot(path="../../assets/architecture.png",full_page=True); b.close()
