from common import *
rng=np.random.default_rng(5)
F=Fig(2000,990); t,R,A,box=F.t,F.R,F.A,F.box
t(1000,58,"Figure 2. Inside one Conformer block of Sota (repeated × 12)",30,500,"#2448b8",fam=SAN)
t(1000,92,"input and output keep the same shape: (B, T/4, 256) · one frame = 40 ms",20,fill="#4b5563",italic=True)
# top macaron strip
R(40,120,1920,230,BL,BLs,2,20)
y=235
t(110,y-8,"x",24,italic=True); t(110,y+20,"(B, T/4, 256)",15,italic=True,fill="#4b5563")
mods=[("½ FFN",PT if False else "#fde2e9","#d9708f","(a)"),("MHSA",BT,BTs,"(b)"),("Conv module","#d6f0ec","#2f9e8f","(c)"),("½ FFN","#fde2e9","#d9708f","(a)")]
x=190; prev=150
xs=[]
for name,f,s,tag in mods:
    A([(prev,y),(x-4,y)])
    box(x,y-38,200,76,f,s,name,tag,size=21)
    px=x+250; F.plus(px,y)
    # residual arc
    F.A([(x-22,y),(x-22,y-70),(px,y-70),(px,y-15)],"#6b7280",1.6)
    A([(x+200,y),(px-15,y)])
    if name=="½ FFN": t(x+225,y-14,"×½",17,italic=True)
    prev=px+13; x=px+60; xs.append(px)
A([(prev,y),(x-4,y)])
box(x,y-30,150,60,GY,GYs,"LayerNorm",size=19)
A([(x+150,y),(x+190,y)])
t(x+215,y-8,"y",24,italic=True); t(x+215,y+20,"(B, T/4, 256)",15,italic=True,fill="#4b5563")
t(1000,335,"macaron design: two half-step feed-forward layers sandwich attention and convolution · every module has a residual ⊕",17,fill="#374151",italic=True)

# panel helper
def panel(x,w,title,col,fill,stroke):
    R(x,380,w,580,fill,stroke,2,20); t(x+w/2,418,title,23,500,col,fam=SAN)
def vstack(cx,items,y0,gap=16,wd=330):
    y=y0
    for i,(l1,l2,f,s,h) in enumerate(items):
        box(cx-wd/2,y,wd,h,f,s,l1,l2,size=18)
        if i<len(items)-1: A([(cx,y+h+1),(cx,y+h+gap-2)],sw=1.6)
        y+=h+gap
    return y-gap
# (a) FFN
panel(40,560,"(a) Feed-forward module","#a3355b","#fdf3f6","#d9708f")
cx=320
end=vstack(cx,[("LayerNorm",None,GY,GYs,46),("Linear 256 → 1024",None,"#fde2e9","#d9708f",50),("Swish  x·σ(x)",None,"#fff","#d9708f",46),("Dropout 0.1",None,"#fff",GYs,40),("Linear 1024 → 256",None,"#fde2e9","#d9708f",50),("× ½  (half step)",None,"#fff","#d9708f",44)],450)
A([(cx,end+2),(cx,end+30)]); F.plus(cx,end+45)
A([(cx-200,450+23),(cx-200,end+45),(cx-15,end+45)],"#6b7280",1.6)
t(cx-210,700,"residual",15,anchor="end",italic=True,fill="#6b7280")
t(cx,end+92,"≈0.53M params per FFN · widens, mixes, shrinks",16,fill="#374151",italic=True)
t(cx,end+116,"applied twice per block (before and after)",16,fill="#374151",italic=True)
# (b) MHSA
panel(620,640,"(b) Multi-head self-attention","#2448b8",BL,BLs)
cx=940
box(cx-160,450,320,46,GY,GYs,"LayerNorm",size=18)
A([(cx,498),(cx,520)],sw=1.6)
for i,l in enumerate(["Q","K","V"]):
    box(cx-150+i*110,524,80,44,BT,BTs,l,size=20)
    A([(cx-110+i*110,570),(cx-110+i*110,592)],sw=1.4)
box(cx-160,596,320,62,BT,BTs,"4 heads × 64 dims","softmax(QKᵀ/√d + rel-pos) · V",size=18)
A([(cx,660),(cx,682)],sw=1.6)
box(cx-160,686,320,44,"#fff",BTs,"Concat + Linear 256 → 256",size=17)
A([(cx,732),(cx,752)],sw=1.6); box(cx-160,756,320,40,"#fff",GYs,"Dropout 0.1",size=17)
A([(cx,798),(cx,820)]); F.plus(cx,835)
A([(cx-190,473),(cx-190,835),(cx-15,835)],"#6b7280",1.6)
# attention map
mx,my,ms=cx+180,540,14
for i in range(8):
    for j in range(8):
        v=math.exp(-abs(i-j)/1.8)*(0.75+0.25*rng.random())
        c="#%02x%02x%02x"%(int(237-200*v),int(243-160*v),int(252-60*v))
        R(mx+j*ms,my+i*ms,ms,ms,c,"#fff",0.5,0)
t(mx+56,my+135,"attention map",15,italic=True,fill="#4b5563")
t(mx+56,my+155,"(frames × frames)",15,italic=True,fill="#4b5563")
t(cx,880,"each 40 ms frame looks at every other frame:",16,fill="#374151",italic=True)
t(cx,902,"links a syllable to the whole sentence · ≈0.26M params",16,fill="#374151",italic=True)
t(cx,936,"relative positions → works for any clip length",16,fill="#374151",italic=True)
# (c) Conv
panel(1280,680,"(c) Convolution module","#1f7a6d","#effaf8","#2f9e8f")
cx=1500
end=vstack(cx,[("LayerNorm",None,GY,GYs,42),("Pointwise 256 → 512",None,"#d6f0ec","#2f9e8f",44),("GLU → 256",None,"#fff","#2f9e8f",40),("Depthwise Conv1D, k = 31",None,"#d6f0ec","#2f9e8f",48),("BatchNorm + Swish",None,"#fff","#2f9e8f",40),("Pointwise 256 → 256",None,"#d6f0ec","#2f9e8f",44),("Dropout 0.1",None,"#fff",GYs,38)],450,gap=14,wd=290)
A([(cx,end+2),(cx,end+24)]); F.plus(cx,end+39)
A([(cx-170,471),(cx-170,end+39),(cx-15,end+39)],"#6b7280",1.6)
# kernel sliding visual
kx,ky=1690,600
t(1820,560,"depthwise kernel over time",17,italic=True,fill="#1f7a6d")
for i in range(9):
    on=3<=i<=5
    R(kx+i*28,ky,22,60,"#2f9e8f" if on else "#d6f0ec","#1f7a6d",1,3)
F.A([(kx+3*28-4,ky+72),(kx+3*28-4,ky+80),(kx+6*28-2,ky+80),(kx+6*28-2,ky+72)],"#1f7a6d",1.8,head=False)
t(1820,ky+108,"k = 31 frames ≈ 1.24 s window",16,fill="#1f7a6d")
F.A([(kx+3*28-4,ky-14),(kx+8*28,ky-14)],"#1f7a6d",1.6)
t(1820,ky+150,"each channel has its own kernel:",16,italic=True,fill="#374151")
t(1820,ky+172,"cheap, local sound patterns",16,italic=True,fill="#374151")
t(1820,ky+194,"(consonant clusters, vowel sounds)",16,italic=True,fill="#374151")
t(1820,ky+230,"≈0.2M params",16,italic=True,fill="#374151")
F.save("../../assets/conformer-block.png","build/fig2.html")
