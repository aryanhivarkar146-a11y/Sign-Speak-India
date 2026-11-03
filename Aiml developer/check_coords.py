import os
import glob
import pandas as pd
import numpy as np

sample_files = glob.glob(r"C:\Users\aryan\Downloads\isl_data\**\*.parquet", recursive=True)[:20]

stats = {'x': [], 'y': [], 'z': []}
types_seen = set()
pose_ranges = {'x': [], 'y': [], 'z': []}
lh_ranges = {'x': [], 'y': [], 'z': []}
rh_ranges = {'x': [], 'y': [], 'z': []}

for f in sample_files:
    df = pd.read_parquet(f)
    for t in df['type'].unique():
        types_seen.add(t)
    
    pose = df[df['type'] == 'pose']
    lh = df[df['type'] == 'left_hand']
    rh = df[df['type'] == 'right_hand']
    
    if len(pose) > 0:
        pose_ranges['x'].extend([pose['x'].min(), pose['x'].max()])
        pose_ranges['y'].extend([pose['y'].min(), pose['y'].max()])
        pose_ranges['z'].extend([pose['z'].min(), pose['z'].max()])
    if len(lh) > 0:
        lh_ranges['x'].extend([lh['x'].min(), lh['x'].max()])
        lh_ranges['y'].extend([lh['y'].min(), lh['y'].max()])
        lh_ranges['z'].extend([lh['z'].min(), lh['z'].max()])
    if len(rh) > 0:
        rh_ranges['x'].extend([rh['x'].min(), rh['x'].max()])
        rh_ranges['y'].extend([rh['y'].min(), rh['y'].max()])
        rh_ranges['z'].extend([rh['z'].min(), rh['z'].max()])

print("All landmark types in parquets:", types_seen)
print(f"Pose X min: {min(pose_ranges['x']):.3f}, max: {max(pose_ranges['x']):.3f}")
print(f"Pose Y min: {min(pose_ranges['y']):.3f}, max: {max(pose_ranges['y']):.3f}")
print(f"Pose Z min: {min(pose_ranges['z']):.3f}, max: {max(pose_ranges['z']):.3f}")

if lh_ranges['x']:
    print(f"LH X min: {min(lh_ranges['x']):.3f}, max: {max(lh_ranges['x']):.3f}")
    print(f"LH Y min: {min(lh_ranges['y']):.3f}, max: {max(lh_ranges['y']):.3f}")
    print(f"LH Z min: {min(lh_ranges['z']):.3f}, max: {max(lh_ranges['z']):.3f}")
else:
    print("LH: no landmarks present in sample files")

if rh_ranges['x']:
    print(f"RH X min: {min(rh_ranges['x']):.3f}, max: {max(rh_ranges['x']):.3f}")
    print(f"RH Y min: {min(rh_ranges['y']):.3f}, max: {max(rh_ranges['y']):.3f}")
    print(f"RH Z min: {min(rh_ranges['z']):.3f}, max: {max(rh_ranges['z']):.3f}")
else:
    print("RH: no landmarks present in sample files")
