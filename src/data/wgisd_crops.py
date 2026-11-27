"""Recorta cada racimo de WGISD con su máscara de instancia sobre fondo blanco (como una foto de estudio)."""
import numpy as np, cv2, glob, os, sys, json
out=sys.argv[1]; rows=[]
for f in sorted(glob.glob("*.npz")):
    name=f[:-4]; img=cv2.imread(name+".jpg"); M=np.load(f)["arr_0"]>0.5; H,W=img.shape[:2]
    for k in range(M.shape[2]):
        m=M[:,:,k]; a=m.sum()
        if a < 0.004*H*W: continue                        # racimos muy pequeños o casi ocultos
        ys,xs=np.where(m); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        if x0<=2 or y0<=2 or x1>=W-3 or y1>=H-3: continue  # cortados por el borde de la foto
        pad=int(0.15*max(y1-y0,x1-x0))
        Y0,Y1,X0,X1=max(0,y0-pad),min(H,y1+pad),max(0,x0-pad),min(W,x1+pad)
        c=img[Y0:Y1,X0:X1].copy(); c[~m[Y0:Y1,X0:X1]]=255
        fn=f"{name}_{k:02d}.jpg"; cv2.imwrite(os.path.join(out,fn),c,[cv2.IMWRITE_JPEG_QUALITY,92]); rows.append([name,fn])
json.dump(rows,open(os.path.join(out,"_recortes.json"),"w"))
print(len(rows),"racimos de",len(glob.glob("*.npz")),"fotos")
