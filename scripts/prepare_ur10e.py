#!/usr/bin/env python3
"""Rebuild vendored UR10e assets from a pinned official description checkout.

Requires xacro, trimesh, pycollada; normal simulation uses prebuilt bundled assets.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

COMMIT = '6662e15f32c23c12ece57d0050aee9a716e4fc41'
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    import xacro
    import trimesh
    actual = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Expected pinned upstream commit {COMMIT}, got {actual}')
    target = ROOT/'assets/ur10e'
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)/'description'
        shutil.copytree(args.source, source, ignore=shutil.ignore_patterns('.git','meshes'))
        for path in source.rglob('*.xacro'):
            path.write_text(path.read_text().replace('$(find ur_description)', str(source)))
        doc = xacro.process_file(str(source/'urdf/ur.urdf.xacro'),
                                 mappings={'name':'ur10e','ur_type':'ur10e'})
        tree = ET.fromstring(doc.toxml())
    for mesh in tree.findall('.//mesh'):
        relative = mesh.attrib['filename'].replace('package://ur_description/','')
        path = args.source/relative
        if path.suffix == '.dae':
            folder = target/'visual'/path.stem
            folder.mkdir(parents=True, exist_ok=True)
            scene = trimesh.load(path, force='scene')
            for number, geometry in enumerate(scene.geometry.values()):
                material = getattr(geometry.visual, 'material', None)
                if material is not None:
                    material.name = f'material_{number}'
            # Keep separate material groups and bake scene transforms into the OBJ.
            from trimesh.exchange.obj import export_obj
            obj, textures = export_obj(scene, return_texture=True)
            (folder/'mesh.obj').write_text(obj)
            for name, data in (textures or {}).items():
                out = folder/name
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(data if isinstance(data,bytes) else data.encode())
            mesh.set('filename', f'visual/{path.stem}/mesh.obj')
        else:
            folder = target/'collision'; folder.mkdir(exist_ok=True)
            shutil.copy2(path, folder/path.name)
            mesh.set('filename', f'collision/{path.name}')
    # Dummy frames have explicitly negligible mass, avoiding Bullet's 1 kg default.
    for link in tree.findall('link'):
        if link.find('inertial') is None:
            i = ET.SubElement(link,'inertial')
            ET.SubElement(i,'mass',value='0.000001')
            ET.SubElement(i,'inertia',ixx='1e-9',iyy='1e-9',izz='1e-9',ixy='0',ixz='0',iyz='0')
    tool = ET.fromstring('''<link name="vacuum_tool"><inertial><origin xyz="0 0 0.09"/><mass value="0.35"/><inertia ixx="0.001" iyy="0.001" izz="0.0002" ixy="0" ixz="0" iyz="0"/></inertial><visual><origin xyz="0 0 0.09"/><geometry><cylinder radius="0.025" length="0.18"/></geometry><material name="tool"><color rgba="0.15 0.15 0.16 1"/></material></visual><collision><origin xyz="0 0 0.09"/><geometry><cylinder radius="0.025" length="0.18"/></geometry></collision></link>''')
    tree.append(tool)
    tree.append(ET.fromstring('<joint name="vacuum_fixed" type="fixed"><parent link="tool0"/><child link="vacuum_tool"/></joint>'))
    tree.append(ET.fromstring('<link name="wrist_camera"><inertial><mass value="0.10"/><inertia ixx="0.00004" iyy="0.00003" izz="0.00004" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.035 0.055 0.035"/></geometry><material name="camera"><color rgba="0.08 0.10 0.13 1"/></material></visual><collision><geometry><box size="0.035 0.055 0.035"/></geometry></collision></link>'))
    tree.append(ET.fromstring('<joint name="camera_fixed" type="fixed"><parent link="tool0"/><child link="wrist_camera"/><origin xyz="0.1 0 0.0175"/></joint>'))
    ET.indent(tree)
    ET.ElementTree(tree).write(target/'robot.urdf',encoding='utf-8',xml_declaration=True)
    shutil.copy2(args.source/'LICENSE',target/'LICENSE')
    (target/'provenance.json').write_text(json.dumps({'source':'https://github.com/UniversalRobots/Universal_Robots_ROS2_Description','commit':actual,'model':'ur10e','calibration':'upstream nominal, not hardware-specific','changes':['flatten xacro','convert visual DAE to OBJ','portable mesh paths','dummy inertials','0.18m ideal vacuum tool']},indent=2))
    print(target/'robot.urdf')

if __name__ == '__main__':
    main()
