import numpy as np 
import torch
SRC2TARGETS = {
    'head':['head','neck'],
    'neck':['head','neck'],
    'chest': ['chest','rinshoulder','linshoulder','rshoulder','lshoulder','neck','spine','belly','root'],
    'spine': ['spine','chest','rinshoulder','linshoulder','rshoulder','lshoulder','belly','root'],
    'belly': ['belly','spine','chest','root'],
    'root': ['root','belly', 'spine', 'chest', 'rhip','lhip','rknee','lknee'],
    'rhip': ['rhip','rknee','rankle','root','belly','spine'],
    'lhip': ['lhip','lknee','lankle','root','belly','spine'],
    'rknee': ['rknee','rhip','rankle','root'],
    'lknee': ['lknee','lhip','lankle','root'],
    'rankle': ['rankle','rknee','rtoes'],
    'lankle': ['lankle','lknee','ltoes'],
    'rtoes': ['rankle','rknee','rtoes'],
    'ltoes': ['lankle','lknee','ltoes'],
    'lhip': ['lhip','lknee','lankle','root','belly','spine'],
    'rhand': ['rhand','rwrist','relbow'],
    'rwrist': ['rhand','rwrist', 'relbow', 'rshoulder'],
    'relbow': ['rhand','rwrist','relbow','rshoulder','rinshoulder','chest','spine','belly'],
    'rshoulder':['rwrist','relbow','rshoulder','rinshoulder','chest','spine','belly','root'],
    'rinshoulder':['rwrist','relbow','rshoulder','rinshoulder','chest','spine','belly','root'],
    'lhand': ['lhand','lwrist','lelbow'],
    'lwrist': ['lhand','lwrist', 'lelbow', 'lshoulder'],
    'lelbow': ['lhand','lwrist','lelbow','lshoulder','linshoulder','chest','spine','belly','root'],
    'lshoulder':['lwrist','lelbow','lshoulder','linshoulder','chest','spine','belly','root'],
    'linshoulder':['lwrist','lelbow','lshoulder','linshoulder','chest','spine','belly','root'],
}

JOINT_NAMES = [
    "root",
    "lhip",
    "rhip",
    "belly",
    "lknee",
    "rknee",
    "spine",
    "lankle",
    "rankle",
    "chest",
    "ltoes",
    "rtoes",
    "neck",
    "linshoulder",
    "rinshoulder",
    "head",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhand",
    "rhand",
]
BONE_NAMES = [
    ("root", "lhip"),
    ("root", "rhip"),
    ("root", "belly"),
    ("lhip", "lknee"),
    ("rhip", "rknee"),
    ("belly", "spine"),
    ("lknee", "lankle"),
    ("rknee", "rankle"),
    ("spine", "chest"),
    ("lankle", "ltoes"),
    ("rankle", "rtoes"),
    ("chest", "neck"),
    ("chest", "linshoulder"),
    ("chest", "rinshoulder"),
    ("neck", "head"),
    ("linshoulder", "lshoulder"),
    ("rinshoulder", "rshoulder"),
    ("lshoulder", "lelbow"),
    ("rshoulder", "relbow"),
    ("lelbow", "lwrist"),
    ("relbow", "rwrist"),
    ("lwrist", "lhand"),
    ("rwrist", "rhand"),
]
SRC2TARGETS_MAT = torch.zeros([24,24], dtype=torch.float32)
PART2JOINTS = torch.zeros([24,24], dtype=torch.float32)
for s,ts in SRC2TARGETS.items():
    assert s in ts
    sid = JOINT_NAMES.index(s)
    tids = [JOINT_NAMES.index(t) for t in ts]
    SRC2TARGETS_MAT[sid, tids] = 1
    PART2JOINTS[tids, sid] = 1 
    #the non-rigid-deformation of t-joint is affected by the rotation of the s-joint.