#!/usr/bin/env python3
"""Export YOLO segmentation and measure prediction agreement and wall latency.

ONNX defaults to CPU parity; TensorRT requires the target NVIDIA GPU. This does
not label a model validated unless all supplied-image comparisons pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def compare(a,b):
    aa=a.boxes.xyxy.cpu().numpy();bb=b.boxes.xyxy.cpu().numpy()
    if len(aa)!=len(bb):return {'pass':False,'reason':'detection_count','reference':len(aa),'exported':len(bb)}
    if not len(aa):return {'pass':True,'detections':0}
    low=np.maximum(aa[:,None,:2],bb[None,:,:2]);high=np.minimum(aa[:,None,2:],bb[None,:,2:])
    intersection=np.prod(np.maximum(0,high-low),axis=2)
    ar=np.prod(aa[:,2:]-aa[:,:2],axis=1);br=np.prod(bb[:,2:]-bb[:,:2],axis=1)
    iou=intersection/np.maximum(ar[:,None]+br[None,:]-intersection,1e-9)
    row,col=linear_sum_assignment(1-iou)
    min_box=float(iou[row,col].min())
    if a.masks is None or b.masks is None:return {'pass':False,'reason':'missing_masks'}
    am=a.masks.data.cpu().numpy()>.5;bm=b.masks.data.cpu().numpy()>.5
    if am.shape[1:]!=bm.shape[1:]:return {'pass':False,'reason':'mask_shape'}
    masks=[float(np.logical_and(am[i],bm[j]).sum()/max(1,np.logical_or(am[i],bm[j]).sum())) for i,j in zip(row,col)]
    confidence=float(np.max(np.abs(a.boxes.conf.cpu().numpy()[row]-b.boxes.conf.cpu().numpy()[col])))
    classes=bool(np.array_equal(a.boxes.cls.cpu().numpy()[row],b.boxes.cls.cpu().numpy()[col]))
    return {'pass':min_box>=.98 and min(masks)>=.95 and confidence<=.02 and classes,
            'detections':len(aa),'min_box_iou':min_box,'min_mask_iou':min(masks),
            'max_confidence_delta':confidence,'class_agreement':classes}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--format',choices=['onnx','engine'],default='onnx')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--limit',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Choose a new output directory')
    if args.format=='engine' and args.device=='cpu':ap.error('TensorRT requires --device 0 (or another GPU)')
    images=sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'})[:args.limit]
    if not images or args.limit<1:ap.error('No input images')
    from ultralytics import YOLO
    import torch
    reference=YOLO(str(args.weights),task='segment')
    if list(reference.names.values())!=['package']:ap.error('Expected a package segmentation checkpoint')
    args.output.mkdir(parents=True)
    # Ultralytics writes alongside checkpoint. Export a copied checkpoint to preserve source files.
    import shutil
    staged=args.output/'package.pt';shutil.copy2(args.weights,staged)
    exported=Path(YOLO(str(staged),task='segment').export(format=args.format,imgsz=640,batch=1,
                  dynamic=False,simplify=False,device=args.device,half=args.format=='engine'))
    deployed=YOLO(str(exported),task='segment')
    def infer(model,path):
        if args.device!='cpu':torch.cuda.synchronize()
        start=time.perf_counter()
        result=model.predict(str(path),device=args.device,imgsz=640,rect=False,
                             retina_masks=True,conf=.5,verbose=False)[0]
        if args.device!='cpu':torch.cuda.synchronize()
        return result,(time.perf_counter()-start)*1000
    for _ in range(3):
        infer(reference,images[0]);infer(deployed,images[0])
    rows=[];pt=[];out=[]
    for path in images:
        a,t1=infer(reference,path);b,t2=infer(deployed,path)
        rows.append({'image':str(path),**compare(a,b)});pt.append(t1);out.append(t2)
    detected=sum(row.get('detections',0) for row in rows)
    passed=all(row['pass'] for row in rows) and detected>0
    report={'pass':passed,'comparisons':rows,'detections_compared':detected,'device':args.device,
        'exported_model':str(exported),'source_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'torch':torch.__version__,'timing_scope':'whole predict call, warmed, synchronized; file decoding included',
        'pytorch_median_ms':float(np.median(pt)),'pytorch_p95_ms':float(np.percentile(pt,95)),
        'export_median_ms':float(np.median(out)),'export_p95_ms':float(np.percentile(out,95))}
    (args.output/'parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    raise SystemExit(0 if passed else 1)

if __name__=='__main__':main()
