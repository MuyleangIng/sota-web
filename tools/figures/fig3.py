from common import *
F=Fig(2000,1000); t,R,A,box=F.t,F.R,F.A,F.box
t(1000,58,"Figure 3. How Sota learns and reads: hybrid CTC / attention",30,500,"#6b3fb0",fam=SAN)
t(1000,92,"training uses both heads together · inference combines them and guards against runaway output",20,fill="#4b5563",italic=True)
# (a) CTC alignment
R(40,125,660,840,GR,GRs,2,20); t(370,165,"(a) CTC alignment",23,500,"#2f7d23",fam=SAN)
t(370,195,"which cluster does each 40 ms frame hear?",17,italic=True,fill="#374151")
toks=["ε","សួ","ស្តី","␣","h","e","l","o"]
path=[0,1,1,0,2,2,3,4,0,5,6,0,6,7,7,0]
gx,gy,cs=170,240,28
for r,tk in enumerate(toks):
    t(gx-14,gy+r*cs+20,tk,17,anchor="end",fam=KM if r in (1,2) else SER)
for c in range(len(path)):
    for r in range(len(toks)):
        on=path[c]==r
        R(gx+c*cs,gy+r*cs,cs-4,cs-4,"#5aa04e" if on else "#e1efda","none",0,4)
pts=[(gx+c*cs+12,gy+path[c]*cs+12) for c in range(len(path))]
F.A(pts,"#e8781a",2,head=False)
t(gx+len(path)*cs/2,gy+len(toks)*cs+26,"frames (time) →",17,italic=True,fill="#4b5563")
y1=gy+len(toks)*cs+50
t(370,y1+16,"best label per frame",17,italic=True)
for c,p in enumerate(path):
    R(gx+c*cs,y1+28,cs-4,32,"#e4f2dd",GBs,1,4); t(gx+c*cs+12,y1+50,toks[p],13,fam=KM if p in(1,2) else SER)
A([(370,y1+68),(370,y1+100)])
t(370,y1+124,"collapse repeats, then drop ε",18)
out=["សួ","ស្តី","␣","h","e","l","l","o"]
ox=370-len(out)*40/2
for i,l in enumerate(out):
    R(ox+i*40,y1+140,34,38,PL,PTs,1.3,5); t(ox+i*40+17,y1+166,l,16,fam=KM if i<2 else SER)
t(370,y1+218,"the ε between “l l” keeps the double letter",16,italic=True,fill="#374151")
t(370,y1+252,"P_CTC(y | x) = sum over all paths that collapse to y",17)
t(370,y1+276,"→ alignment is learned, never labelled by hand",16,italic=True,fill="#374151")
# (b) joint training
R(720,125,560,840,AM,"#d9a54a",2,20); t(1000,165,"(b) Joint training",23,500,"#8a5306",fam=SAN)
t(1000,195,"one encoder, two teachers",17,italic=True,fill="#374151")
box(840,800,320,70,BT,BTs,"Conformer encoder","shared · ≈20M",size=20)
box(750,560,210,70,GB,GBs,"CTC head","per frame",size=19)
box(1040,560,210,70,PT,PTs,"Attention decoder","teacher forcing",size=19)
A([(930,798),(930,700),(855,700),(855,634)])
A([(1070,798),(1070,700),(1145,700),(1145,634)],BTs,1.8)
t(1105,690,"cross-attn",14,italic=True,fill=BTs,anchor="start")
box(775,420,160,56,"#fff",AMs,"L_CTC".replace("L_CTC","L<tspan baseline-shift='sub' font-size='13'>CTC</tspan>"),size=20)
box(1065,420,160,56,"#fff",AMs,"L<tspan baseline-shift='sub' font-size='13'>CE</tspan>",size=20)
A([(855,558),(855,480)]); A([(1145,558),(1145,480)])
A([(855,418),(855,330),(975,330),(975,300)]); A([(1145,418),(1145,330),(1025,330),(1025,300)])
t(880,370,"× 0.3",19,anchor="start",italic=True); t(1120,370,"× 0.7",19,anchor="end",italic=True)
R(860,235,280,62,"#fff",AMs,1.8,10)
t(1000,274,"L = 0.3 · L<tspan baseline-shift='sub' font-size='13'>CTC</tspan> + 0.7 · L<tspan baseline-shift='sub' font-size='13'>CE</tspan>",20)
F.A([(1258,300),(1258,835),(1165,835)],"#a15c07",1.8,"7 5")
t(1250,580,"",1)
t(1000,900,"gradients from both losses update the encoder",16,italic=True,fill="#374151")
t(1000,924,"CTC → clean, monotonic alignment · decoder → Khmer spelling",16,italic=True,fill="#374151")
t(1268,560,"backprop",15,italic=True,fill="#a15c07",anchor="end")
# (c) inference
R(1300,125,660,840,PU,PUs,2,20); t(1630,165,"(c) Inference on device",23,500,"#6b3fb0",fam=SAN)
t(1630,195,"ONNX Runtime · CPU · offline",17,italic=True,fill="#374151")
cx=1500
box(cx-150,225,300,50,GY,GYs,"Audio chunk ≤ 10 s",size=18)
A([(cx,277),(cx,300)])
box(cx-150,304,300,56,BT,BTs,"Encoder (runs once)",size=18)
A([(cx,362),(cx,385)])
box(cx-150,389,300,62,GB,GBs,"CTC greedy draft","gives cluster count n",size=18)
A([(cx,453),(cx,476)])
box(cx-150,480,300,74,PT,PTs,"Beam search · 5 beams","0.3·log P_CTC + 0.7·log P_dec",size=18)
A([(cx,556),(cx,582)])
# diamond
dy=650
F.a(f'<polygon points="{cx},{dy-64} {cx+150},{dy} {cx},{dy+64} {cx-150},{dy}" fill="#fff" stroke="{AMs}" stroke-width="1.8"/>')
t(cx,dy-6,"output length",17); t(cx,dy+18,"&gt; 1.5 × n ?",17)
A([(cx+150,dy),(cx+260,dy),(cx+260,dy+90)]); t(cx+205,dy-10,"yes",17,italic=True)
box(cx+165,dy+94,190,62,GB,GBs,"use CTC draft","runaway guard",size=17)
A([(cx,dy+66),(cx,dy+150)]); t(cx+18,dy+108,"no",17,italic=True,anchor="start")
box(cx-150,dy+154,300,50,PL,PTs,"best beam",size=18)
A([(cx,dy+206),(cx,dy+228)]); A([(cx+260,dy+158),(cx+260,dy+252),(cx+155,dy+252)])
R(cx-150,dy+232,300,66,"#efe7d6","#a8916b",1.4,8)
t(cx,dy+262,"សួស្តី hello",24,600,fam=KM); t(cx,dy+287,"+ confidence score",14,italic=True,fill="#4b5563")
# mini beam tree
bx0,by0=1710,300
nodes=[(bx0,by0+100)]
lv1=[(bx0+70,by0+40+i*30) for i in range(5)]
best=[(bx0,by0+100),(bx0+70,by0+70),(bx0+140,by0+90),(bx0+205,by0+100)]
lv2=[(bx0+140,by0+50+i*40) for i in range(4)]
lv3=[(bx0+205,by0+60+i*40) for i in range(3)]
for p in lv1: F.a(f'<line x1="{bx0}" y1="{by0+100}" x2="{p[0]}" y2="{p[1]}" stroke="#9ca3af"/>')
for a_,b_ in zip(lv1[:4],lv2): F.a(f'<line x1="{a_[0]}" y1="{a_[1]}" x2="{b_[0]}" y2="{b_[1]}" stroke="#9ca3af"/>')
for a_,b_ in zip(lv2[:3],lv3): F.a(f'<line x1="{a_[0]}" y1="{a_[1]}" x2="{b_[0]}" y2="{b_[1]}" stroke="#9ca3af"/>')
F.A(best,"#9b74c9",3,head=False)
for p in [nodes[0]]+lv1+lv2+lv3:
    on=p in best
    F.a(f'<circle cx="{p[0]}" cy="{p[1]}" r="{6 if on else 4.5}" fill="{"#9b74c9" if on else "#fff"}" stroke="#4b5563"/>')
t(bx0+105,by0+250,"5 beams, best path kept",16,italic=True,fill="#4b5563")
t(bx0+105,by0+275,"one cluster per step",16,italic=True,fill="#4b5563")
F.save("../../assets/training-decoding.png","build/fig3.html")
