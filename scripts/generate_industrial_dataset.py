#!/usr/bin/env python3
"""UR10e wrist-view synthetic dataset; simulator IDs are used for labels only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import yaml
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--train',type=int,default=600)
    ap.add_argument('--val',type=int,default=120)
    ap.add_argument('--test',type=int,default=120)
    ap.add_argument('--seed',type=int,default=20261006)
    args=ap.parse_args()
    if args.output.exists():ap.error('Choose a new output directory')
    if min(args.train,args.val,args.test)<1:ap.error('Every split must be nonempty')
    import pybullet as pb
    from pybullet_utils.bullet_client import BulletClient
    from palletizing.industrial.cell import CellScene
    from palletizing.industrial.motion import URController
    from palletizing.backends.bullet import WristCamera
    from palletizing.dataset import instance_polygons
    from scripts.generate_dataset import carton_texture
    c=yaml.safe_load((ROOT/'config/industrial.yaml').read_text())
    p=BulletClient(pb.DIRECT);robot=None
    args.output.mkdir(parents=True)
    seen=set();records=[];previews=[]
    try:
        scene=CellScene(p,c,ROOT,[]);robot=URController(scene,c);robot.bootstrap()
        camera=WristCamera(p,robot,c)
        with tempfile.TemporaryDirectory() as temp:
            for split_id,(split,count) in enumerate([('train',args.train),('val',args.val),('test',args.test)]):
                for directory in ['images','labels']:(args.output/directory/split).mkdir(parents=True)
                for i in range(count):
                    for attempt in range(40):
                        seed=int(np.random.SeedSequence([args.seed,split_id,i,attempt]).generate_state(1)[0])
                        rng=np.random.default_rng(seed);bodies=[]
                        negative=bool(rng.random()<.1)
                        if not negative:
                            d=rng.uniform([.18,.18,.14],[.33,.33,.30])
                            xy=np.array(c['table']['center_xy'])+rng.uniform(-.045,.045,2)
                            body=scene.box(d,[*xy,.2+d[2]/2+.002],rng.uniform([.45,.24,.08],[.86,.71,.48]).tolist()+[1],
                                           mass=1,yaw=rng.uniform(-np.pi,np.pi))
                            bodies=[body]
                            if rng.random()<.75:
                                path=Path(temp)/f'texture_{seed}.png';carton_texture(path,rng)
                                p.changeVisualShape(body,-1,textureUniqueId=p.loadTexture(str(path)))
                        p.changeVisualShape(scene.surfaces['table'],-1,rgbaColor=rng.uniform(.20,.42,3).tolist()+[1])
                        for _ in range(100):p.stepSimulation()
                        camera.capture(labels=True,lightDirection=rng.uniform([-2,-2,2],[2,2,5]).tolist(),
                                       lightAmbientCoeff=float(rng.uniform(.35,.65)),
                                       lightDiffuseCoeff=float(rng.uniform(.35,.65)))
                        rgba=camera.rgb.copy();seg=camera.segmentation.copy()
                        for body in bodies:
                            p.removeBody(body);scene.collision_objects.pop(body,None)
                        try:labels,_=instance_polygons(seg,bodies)
                        except ValueError:continue
                        digest=hashlib.sha256(rgba.tobytes()).hexdigest()
                        if digest not in seen:break
                    else:raise RuntimeError(f'Cannot produce unique valid sample {split}/{i}')
                    seen.add(digest);name=f'{split}_{i:05d}'
                    image=Image.fromarray(rgba[:,:,:3])
                    image.save(args.output/'images'/split/f'{name}.jpg',quality=95)
                    (args.output/'labels'/split/f'{name}.txt').write_text('\n'.join(labels)+('\n' if labels else ''))
                    records.append({'name':name,'split':split,'seed':seed,'objects':len(labels),'rgba_sha256':digest})
                    if i<4:
                        preview=image.copy();draw=ImageDraw.Draw(preview)
                        for label in labels:
                            xy=np.array(label.split()[1:],float).reshape(-1,2)*[640,480]
                            draw.line([tuple(v) for v in xy]+[tuple(xy[0])],fill='lime',width=3)
                        preview.thumbnail((320,240));previews.append(preview)
                    if (i+1)%25==0 or i+1==count:print(f'{split}: {i+1}/{count}',flush=True)
        data={'path':str(args.output.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'package'}}
        (args.output/'dataset.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
        (args.output/'manifest.json').write_text(json.dumps({'robot':'UR10e','renderer':'TinyRenderer','scenes':records},indent=2))
        montage=Image.new('RGB',(1280,720),'white')
        for i,im in enumerate(previews):montage.paste(im,((i%4)*320,(i//4)*240))
        montage.save(args.output/'preview.png')
        print('Dataset:',args.output/'dataset.yaml')
    finally:
        if robot is not None:robot.planner.close()
        p.disconnect()

if __name__=='__main__':main()
