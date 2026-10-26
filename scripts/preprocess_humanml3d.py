#!/usr/bin/env python3
"""Terminal preprocessing for the cloned HumanML3D dataset.

Adapted from raw_pose_processing.ipynb and cal_mean_variance.ipynb.
Copyright (c) 2022 Chuan Guo. See LICENSE_HumanML3D.txt.
"""
import argparse
import csv
from pathlib import Path
import sys
import zipfile

ALIASES = {
    'MPI_HDM05': 'HDM05', 'TCD_handMocap': 'TCDHands', 'MPI_mosh': 'MoSh',
    'Eyes_Japan_Dataset': 'EyesJapanDataset', 'Transitions_mocap': 'Transitions',
    'MPI_Limits': 'PosePrior', 'SSM_synced': 'SSM', 'DFaust_67': 'DFaust',
    'BioMotionLab_NTroje': 'BMLrub',
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--humanml-root', type=Path, default=Path('../HumanML3D'),
                   help='Cloned HumanML3D repository; relative to the working directory.')
    p.add_argument('--stage', choices=['all', 'poses', 'segment', 'features', 'stats', 'verify'],
                   default='all')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, etc.')
    p.add_argument('--chunk-size', type=int, default=512,
                   help='Frames per body-model call; limits GPU memory.')
    p.add_argument('--overwrite', action='store_true', help='Recompute existing outputs.')
    args = p.parse_args()
    if args.chunk_size < 1:
        p.error('--chunk-size must be positive')
    return args


def read_index(root):
    with (root / 'index.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        source = Path(row['source_path'])
        if source.is_absolute() or '..' in source.parts or source.parts[0] != 'pose_data':
            raise ValueError('Unexpected index path: ' + str(source))
        if Path(row['new_name']).name != row['new_name']:
            raise ValueError('Unexpected output name: ' + row['new_name'])
    return rows


def atomic_save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        np.save(f, array)
    temp.replace(path)


def extract_zip(archive, destination):
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(str(archive)) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError('Unsafe archive path: ' + info.filename)
            if not target.exists():
                z.extract(info, str(destination))


def amass_source(root, pose_path):
    rel = Path(pose_path).relative_to('pose_data').with_suffix('.npz')
    path = root / 'amass_data' / rel
    if not path.exists() and rel.parts[0] in ALIASES:
        path = root / 'amass_data' / ALIASES[rel.parts[0]] / Path(*rel.parts[1:])
    return path


def pose_stage(root, rows, args):
    sources = sorted({r['source_path'] for r in rows if 'humanact12' not in r['source_path']})
    pending = [s for s in sources if args.overwrite or not (root / s).exists()]
    if not pending:
        print('poses: all requested outputs already exist', flush=True)
        return
    missing = [str(amass_source(root, s)) for s in pending if not amass_source(root, s).is_file()]
    model_paths = [root / 'body_models' / kind / gender / 'model.npz'
                   for kind in ['smplh', 'dmpls'] for gender in ['male', 'female']]
    missing += [str(p) for p in model_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing AMASS/body model files (first 20):\n' + '\n'.join(missing[:20]))
    from human_body_prior.body_model.body_model import BodyModel
    device = args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = {g: BodyModel(bm_fname=str(root/'body_models/smplh'/g/'model.npz'),
                          num_betas=10, num_dmpls=8,
                          dmpl_fname=str(root/'body_models/dmpls'/g/'model.npz')).to(device).eval()
              for g in ['male', 'female']}
    transform = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    for source in tqdm(pending, desc='AMASS -> pose_data'):
        with np.load(amass_source(root, source), allow_pickle=True) as data:
            step = int(float(data['mocap_framerate']) / 20)
            if step < 1:
                raise ValueError('Source frame rate below 20 fps: ' + source)
            # Preserve the notebook's integer-stride downsampling and gender choice.
            gender = data['gender'].item()
            if isinstance(gender, bytes):
                gender = gender.decode()
            model = models['male' if gender == 'male' else 'female']
            poses, trans = data['poses'][::step], data['trans'][::step]
            if not len(poses):
                raise ValueError('Empty AMASS clip: ' + source)
            chunks = []
            with torch.no_grad():
                for start in range(0, len(poses), args.chunk_size):
                    pose = torch.as_tensor(poses[start:start+args.chunk_size], dtype=torch.float32, device=device)
                    tr = torch.as_tensor(trans[start:start+args.chunk_size], dtype=torch.float32, device=device)
                    betas = torch.as_tensor(np.repeat(data['betas'][:10][None], len(pose), axis=0),
                                            dtype=torch.float32, device=device)
                    body = model(root_orient=pose[:, :3], pose_body=pose[:, 3:66],
                                 pose_hand=pose[:, 66:], trans=tr, betas=betas)
                    chunks.append(body.Jtr.detach().cpu().numpy())
            joints = np.dot(np.concatenate(chunks), transform)
            if not np.isfinite(joints).all():
                raise ValueError('Nonfinite poses: ' + source)
            atomic_save(root / source, joints)


def swap_left_right(data):
    data = data.copy()
    data[..., 0] *= -1
    right = [2, 5, 8, 11, 14, 17, 19, 21]
    left = [1, 4, 7, 10, 13, 16, 18, 20]
    data[:, right], data[:, left] = data[:, left].copy(), data[:, right].copy()
    if data.shape[1] > 24:
        left = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
        data[:, right], data[:, left] = data[:, left].copy(), data[:, right].copy()
    return data


def segment_clip(data, source, start, end):
    data = data.copy()
    if 'humanact12' not in source:
        offsets = {'Eyes_Japan_Dataset': 60, 'MPI_HDM05': 60, 'TotalCapture': 20,
                   'MPI_Limits': 20, 'Transitions_mocap': 10}
        for name, frames in offsets.items():
            if name in source:
                data = data[frames:]
        data = data[start:end]
        data[..., 0] *= -1
    return data, swap_left_right(data)


def segment_stage(root, rows, args):
    humanact = [root / r['source_path'] for r in rows if 'humanact12' in r['source_path']]
    if any(not p.is_file() for p in humanact):
        extract_zip(root/'pose_data/humanact12.zip', root/'pose_data')
    missing = [r['source_path'] for r in rows if not (root / r['source_path']).is_file()]
    if missing:
        raise FileNotFoundError('Run poses first; missing paths:\n' + '\n'.join(missing[:20]))
    for row in tqdm(rows, desc='Segment + mirror'):
        outputs = [root/'joints'/row['new_name'], root/'joints'/('M'+row['new_name'])]
        if not args.overwrite and all(p.is_file() for p in outputs):
            continue
        data = np.load(root / row['source_path'])
        pair = segment_clip(data, row['source_path'], int(row['start_frame']), int(row['end_frame']))
        for path, motion in zip(outputs, pair):
            if len(motion) < 2 or motion.ndim != 3 or not np.isfinite(motion).all():
                raise ValueError('Invalid segmented clip: ' + str(path))
            atomic_save(path, motion)


def feature_stage(root, rows, args):
    import _humanml_features as features
    features.configure(np.load(root/'joints/000021.npy')[:, :22])
    output = root/'HumanML3D'
    reference_dir = root/'.preprocess_reference'
    # Preserve the clone's reference example before overwriting it, including on resume.
    for kind in ['new_joints', 'new_joint_vecs']:
        source, ref = output/kind/'012314.npy', reference_dir/kind/'012314.npy'
        if source.is_file() and not ref.exists():
            atomic_save(ref, np.load(source))
    for row in tqdm(rows, desc='263-D motion features'):
        for name in [row['new_name'], 'M'+row['new_name']]:
            vec, joint = output/'new_joint_vecs'/name, output/'new_joints'/name
            # The original clone already contains 012314; never mistake it for processed output.
            marker = root/'.preprocess_done'/name
            if not args.overwrite and marker.is_file() and vec.is_file() and joint.is_file():
                continue
            source = np.load(root/'joints'/name)[:, :22]
            data, _, _, _ = features.process_file(source, 0.002)
            recovered = features.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), 22)[0].numpy()
            if data.shape != (len(source)-1, 263) or not np.isfinite(data).all() or not np.isfinite(recovered).all():
                raise ValueError('Invalid motion features: ' + name)
            atomic_save(vec, data)
            atomic_save(joint, recovered)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    for kind in ['new_joints', 'new_joint_vecs']:
        ref = reference_dir/kind/'012314.npy'
        if ref.exists():
            expected, actual = np.load(ref), np.load(output/kind/'012314.npy')
            error = np.max(np.abs(expected-actual)) if expected.shape == actual.shape else float('inf')
            print('{} reference max abs error: {}'.format(kind, error), flush=True)
            if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-4, rtol=1e-4):
                raise ValueError('Reference mismatch; inspect preprocessing/environment: ' + kind)


def streaming_stats(paths):
    count = 0
    mean, m2 = np.zeros(263), np.zeros(263)
    for path in tqdm(paths, desc='Mean/std'):
        x = np.load(path).astype(np.float64)
        if x.ndim != 2 or x.shape[1] != 263 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid features: ' + str(path))
        n, batch_mean = len(x), x.mean(axis=0)
        delta = batch_mean-mean
        m2 += ((x-batch_mean)**2).sum(axis=0) + delta**2 * count*n/(count+n)
        mean += delta*n/(count+n)
        count += n
    if not count:
        raise ValueError('No motion frames')
    std = np.sqrt(np.maximum(m2/count, 0))
    for lo, hi in [(0, 1), (1, 3), (3, 4), (4, 67), (67, 193), (193, 259), (259, 263)]:
        std[lo:hi] = std[lo:hi].mean()
    return mean, std


def stats_stage(root, rows):
    output = root/'HumanML3D'
    paths = [output/'new_joint_vecs'/name for r in rows for name in [r['new_name'], 'M'+r['new_name']]]
    mean, std = streaming_stats(paths)
    # Keep training normalization untouched; select 67-D statistics separately in README.
    atomic_save(output/'Mean_263.npy', mean)
    atomic_save(output/'Std_263.npy', std)
    print('Saved Mean_263.npy and Std_263.npy; training Mean.npy/Std.npy unchanged.', flush=True)


def verify_stage(root):
    output = root/'HumanML3D'
    if not (output/'texts').is_dir():
        extract_zip(output/'texts.zip', output)
    for split in ['train', 'val', 'test']:
        names = (output/(split+'.txt')).read_text().split()
        missing = [name for name in names if not (output/'texts'/(name+'.txt')).is_file()
                   or not (output/'new_joint_vecs'/(name+'.npy')).is_file()]
        if missing:
            raise FileNotFoundError('{} missing: {}'.format(split, missing[:20]))
        for name in tqdm(names, desc='Verify '+split):
            x = np.load(output/'new_joint_vecs'/(name+'.npy'), mmap_mode='r')
            if x.ndim != 2 or x.shape[1] != 263 or len(x) < 1 or not np.isfinite(x).all():
                raise ValueError('Invalid features: ' + name)
        print('{}: {} clips OK'.format(split, len(names)), flush=True)


def main():
    args = arguments()
    root = args.humanml_root.resolve()
    if not (root/'index.csv').is_file():
        raise FileNotFoundError('Set --humanml-root to the cloned HumanML3D repository.')
    global np, torch, tqdm
    import numpy as np
    import torch
    from tqdm import tqdm
    if np.lib.NumpyVersion(np.__version__) >= '1.24.0':
        raise RuntimeError('HumanML3D common/quaternion.py requires NumPy < 1.24; use torch_render.')
    sys.path.insert(0, str(root))
    rows = read_index(root)
    stages = ['poses', 'segment', 'features', 'stats', 'verify'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print('=== {} ==='.format(stage), flush=True)
        if stage == 'stats':
            stats_stage(root, rows)
        elif stage == 'verify':
            verify_stage(root)
        else:
            {'poses': pose_stage, 'segment': segment_stage, 'features': feature_stage}[stage](root, rows, args)
    print('Preprocessing complete.', flush=True)


if __name__ == '__main__':
    main()
