import numpy as np, math
SER="'LM Roman 10','Latin Modern Roman','Noto Sans Khmer',serif"
SAN="Inter,'Noto Sans Khmer',sans-serif"
KM="'Noto Sans Khmer',serif"
BL,BLs,BT,BTs="#edf3fc","#6f9be6","#bcd4f6","#4a7fd8"
GR,GRs,GB,GBs="#eef7ea","#8cc483","#c9e6be","#5aa04e"
PU,PUs,PT,PTs,PL="#f7f2fc","#b79bd6","#dccaf0","#9b74c9","#efe6f8"
GY,GYs="#eceef1","#9ca3af"
AM,AMs="#fffbea","#a15c07"
class Fig:
    def __init__(s,W,H): s.W,s.H=W,H; s.o=[]; s.a=s.o.append
    def t(s,x,y,txt,size=20,w=400,fill="#1f2937",anchor="middle",fam=SER,italic=False):
        st=' font-style="italic"' if italic else ''
        s.a(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{st}>{txt}</text>')
    def R(s,x,y,w,h,fill,stroke,sw=1.5,rx=8,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        s.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def A(s,pts,c="#111827",sw=2,dash=None,head=True):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        m=' marker-end="url(#ah)"' if head else ''
        s.a('<polyline points="'+" ".join(f"{x},{y}" for x,y in pts)+f'" fill="none" stroke="{c}" stroke-width="{sw}"{d}{m} style="color:{c}"/>')
    def plus(s,x,y,r=13):
        s.a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="#111827" stroke-width="1.6"/>')
        s.a(f'<line x1="{x-r+5}" y1="{y}" x2="{x+r-5}" y2="{y}" stroke="#111827" stroke-width="1.6"/><line x1="{x}" y1="{y-r+5}" x2="{x}" y2="{y+r-5}" stroke="#111827" stroke-width="1.6"/>')
    def box(s,x,y,w,h,fill,stroke,l1,l2=None,size=19,fam=SAN,sw=1.5):
        s.R(x,y,w,h,fill,stroke,sw,8)
        if l2: s.t(x+w/2,y+h/2-3,l1,size,fam=fam); s.t(x+w/2,y+h/2+20,l2,15,fill="#4b5563",italic=True)
        else: s.t(x+w/2,y+h/2+6,l1,size,fam=fam)
    def save(s,path,html):
        svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" height="{s.H}" viewBox="0 0 {s.W} {s.H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L9 5L1 9z" fill="currentColor"/></marker></defs><rect width="{s.W}" height="{s.H}" fill="#fff"/>'+"\n".join(s.o)+'</svg>')
        __import__('os').makedirs('build',exist_ok=True); open(html,"w").write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{{margin:0}}</style></head><body>{svg}</body></html>')
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b=p.chromium.launch(); pg=b.new_page(viewport={"width":s.W,"height":s.H},device_scale_factor=2)
            pg.goto("file://"+__import__('os').path.abspath(html)); pg.wait_for_timeout(600)
            pg.screenshot(path=path,full_page=True); b.close()
