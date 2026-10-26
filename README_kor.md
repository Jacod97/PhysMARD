# PhysMARD 학습 실행 가이드

## 1. 환경 설정

각 명령 블록은 `PhysMARD/` 저장소 루트에서 시작한다. 데이터 전처리 저장소는 바로 옆에 둔다.

```text
workspace/
├── PhysMARD/
│   └── checkpoints/t2m/AE/model/latest.tar
└── HumanML3D/
```

```bash
conda env create -n PhysMARD -f environment.yml
conda activate PhysMARD
python -m pip install requests beautifulsoup4 tqdm
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

저장소 환경 기준은 Python 3.10.13 / PyTorch 2.2.0 / CUDA 12.1이다. 환경 설치에 실패하면 해당 오류를 해결한 뒤 진행한다. `clip.load`가 없으면 OpenAI CLIP을 설치한다.

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

## 2. 서버에서 데이터셋 일괄 다운로드·준비

### 2.1. AMASS 18개를 서버에 직접 다운로드

[AMASS](https://amass.is.tue.mpg.de/) 계정을 만든 뒤 아래 명령을 **서버 터미널**에서 실행한다. 이메일과 비밀번호는 실행 중 입력하며 파일에 저장하지 않는다. 로컬 컴퓨터에 데이터를 받을 필요는 없다.

처음 다운로드할 때는 아래 방법을 사용한다. [GBC 다운로드 도구](https://github.com/SJTU-MVASL-Robotics/GBC/blob/master/GBC/utils/data_preparation/download_amass.py)로 로그인하고 HumanML3D에 필요한 18개 `SMPL+H G`만 선택한다. 서버 터미널 연결을 유지한 상태에서 실행한다.

```bash
cd ..
# HumanML3D가 이미 있으면 clone 명령은 생략한다.
git clone https://github.com/EricGuo5513/HumanML3D.git
cd HumanML3D
mkdir -p amass_data
curl -fL 'https://raw.githubusercontent.com/SJTU-MVASL-Robotics/GBC/master/GBC/utils/data_preparation/download_amass.py' \
  -o amass_data/download_amass.py
python - <<'PY'
import getpass
import runpy
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import requests

api = runpy.run_path('amass_data/download_amass.py')
names = set('ACCAD HDM05 TCDHands SFU BMLmovi CMU MoSh EKUT KIT EyesJapanDataset BMLhandball Transitions PosePrior HumanEva SSM DFaust TotalCapture BMLrub'.split())
with requests.Session() as session:
    email = getpass.getpass('AMASS email (hidden): ')
    password = getpass.getpass('AMASS password: ')
    authenticated = api['login'](session, email, password)
    del password
    if authenticated is None:
        raise SystemExit('AMASS 로그인 실패')
    selected = {}
    for url in api['get_download_urls'](session, smplh_only=True):
        if '/gender_specific/' not in url:
            continue
        name = Path(parse_qs(urlparse(url).query)['sfile'][0]).name.replace('.tar.bz2', '')
        if name in names:
            selected.setdefault(name, url)
    assert set(selected) == names, f'누락: {names - set(selected)}'
    for name, url in selected.items():
        # 기존 부분 파일에 잘못 이어붙이지 않도록 해당 파일을 처음부터 받는다.
        result = api['download_file'](session, url, 'amass_data', force=True)
        if result is None:
            raise SystemExit(f'{name} 다운로드 실패')
print('18개 다운로드 완료')
PY
```

압축 파일은 `../HumanML3D/amass_data/`에 저장된다.

### 2.2. 압축 해제와 전처리

```bash
cd ../HumanML3D
mkdir -p amass_data joints
(
  set -e
  for archive in amass_data/*.tar.bz2; do
    bzip2 -t "$archive"
    tar -xjf "$archive" -C amass_data
  done
)
unzip -n pose_data/humanact12.zip -d pose_data
unzip -n HumanML3D/texts.zip -d HumanML3D
```

`scripts/preprocess_humanml3d.py`는 필요한 `.npz`만 읽으므로 `amass_data/`에 압축 파일이나 로그가 남아 있어도 된다. 아래 폴더명은 자동으로 대응하므로 수동으로 변경할 필요 없다.

```text
HDM05            → MPI_HDM05
TCDHands         → TCD_handMocap
MoSh             → MPI_mosh
EyesJapanDataset → Eyes_Japan_Dataset
Transitions      → Transitions_mocap
PosePrior        → MPI_Limits
SSM              → SSM_synced
DFaust           → DFaust_67
BMLrub           → BioMotionLab_NTroje
```

나머지 폴더명은 `ACCAD`, `SFU`, `BMLmovi`, `CMU`, `EKUT`, `KIT`, `BMLhandball`, `HumanEva`, `TotalCapture`다.

추가로 다음 인체 모델을 받아 배치한다.

- [SMPL+H](https://mano.is.tue.mpg.de/download.php): `Extended SMPL+H model used in AMASS project`
- [DMPL](https://smpl.is.tue.mpg.de/download.php): `DMPLs compatible with SMPL`

```text
HumanML3D/body_models/
├── smplh/male/model.npz
├── smplh/female/model.npz
├── dmpls/male/model.npz
└── dmpls/female/model.npz
```

전처리는 설치된 `torch_render` 환경에서 실행한다. 학습용 환경은 `PhysMARD`다. 아래 명령은 **PhysMARD 저장소 루트에서** 실행한다.

```bash
conda activate torch_render
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --device cuda:2 --chunk-size 128
```

AMASS 관절 복원 → 구간 자르기·미러링 → 263차원 특징 생성 → 통계 계산 → split 파일 검사를 순서대로 실행한다. HumanAct12 압축은 필요한 경우 자동 해제한다. 원본 노트북의 `common/`, `paramUtil.py`, body model 모듈을 사용하므로 HumanML3D 클론은 유지한다. 해당 모듈 호환성 때문에 전처리는 NumPy 1.24 미만 환경을 사용한다.

완료한 파일은 재실행 시 건너뛴다. 특정 단계만 실행하려면:

```bash
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --stage poses --device cuda:2 --chunk-size 128
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --stage segment
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --stage features
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --stage stats
python scripts/preprocess_humanml3d.py --humanml-root ../HumanML3D --stage verify
```

GPU 메모리가 부족하면 `--chunk-size 128`, CPU를 쓰려면 `--device cpu`를 지정한다. 입력이나 설정을 바꿔 기존 결과까지 다시 계산할 때는 `--overwrite`를 붙인다.

결과는 `../HumanML3D/HumanML3D/new_joint_vecs/`, `new_joints/`에 저장된다. 263차원 통계는 `Mean_263.npy`, `Std_263.npy`에 저장하고, 학습용 `Mean.npy`, `Std.npy`는 다음 단계에서 선택한다. 참조 샘플과 결과가 맞지 않거나 필요한 파일이 누락되면 오류로 중단한다.

### 2.3. 학습 데이터 연결

```bash
conda activate PhysMARD
mkdir -p datasets
# 링크가 없을 때 한 번만 실행한다.
ln -s ../../HumanML3D/HumanML3D datasets/HumanML3D

# 전처리에서 만든 263차원 통계를 보관한다.
cp -n datasets/HumanML3D/Mean.npy datasets/HumanML3D/Mean_263_original.npy
cp -n datasets/HumanML3D/Std.npy datasets/HumanML3D/Std_263_original.npy

# 공개 AE 사용 경로: 저장소에서 제공하는 67차원 통계 사용.
cp utils/eval_mean_std/t2m/eval_mean.npy datasets/HumanML3D/Mean.npy
cp utils/eval_mean_std/t2m/eval_std.npy datasets/HumanML3D/Std.npy
```

개별 모션은 263차원 파일을 유지하고, 학습 통계만 67차원으로 맞춘다. AE부터 생성 모델 학습까지 같은 통계를 사용한다.

## 3. 오토인코더(AE) 모델 불러오기

현재 AE는 `checkpoints/t2m/AE/model/latest.tar`에 준비되어 있다. 아래 다운로드는 파일이 없을 때만 실행한다.

```bash
mkdir -p checkpoints/t2m
wget -c 'https://huggingface.co/cr8br0ze/AE_humanml3d/resolve/main/AE_humanml3d.zip' \
  -O checkpoints/t2m/AE_humanml3d.zip
unzip -n checkpoints/t2m/AE_humanml3d.zip -d checkpoints/t2m
```

최종 경로를 다음과 맞춘다.

```text
checkpoints/t2m/AE/model/latest.tar
```

학습 명령에 `--ae_name AE`를 지정하면 이 checkpoint의 `ae` 가중치를 자동으로 불러오고 고정한다. 공개 생성 모델 checkpoint는 새 학습에 필요 없다.

## 4. 학습 실행

### 4.1. 첫 실행 확인

```bash
conda activate PhysMARD
python train_MARDM.py --name DDPM_smoke --model MARDM-DDPM-XL \
  --dataset_name t2m --ae_name AE --batch_size 4 --num_workers 0 \
  --epoch 1 --log_every 1
```

`--epoch 1`은 전체 데이터를 한 번 학습한다는 뜻이다. 시작 시 CLIP 다운로드와 초기 생성 관통 평가 때문에 첫 학습 로그까지 시간이 걸릴 수 있다.

### 4.2. 기본 DDPM 학습

```bash
python train_MARDM.py --name DDPM_MSE --model MARDM-DDPM-XL \
  --dataset_name t2m --ae_name AE --batch_size 64 --epoch 500 --seed 3407
```

KL objective로 학습하려면 실험 이름을 `DDPM_KL`로 바꾸고 `--use_kl`을 추가한다.

### 4.3. Physics 학습

```bash
python train_MARDM.py --name PhysMARD_KL --model MARDM-DDPM-XL \
  --dataset_name t2m --ae_name AE --batch_size 64 --epoch 500 \
  --seed 3407 --use_kl --physics --physics_ddim_steps 0
```

학습 batch의 GPU 메모리가 부족하면 `--batch_size`를 줄인다. 초기 calibration과 생성 평가의 batch는 별도로 고정되어 있다.

결과 checkpoint는 `checkpoints/t2m/<실험 이름>/model/latest.tar`에 저장된다. 중단 후에는 기존 명령에 `--is_continue`를 추가한다. 모델·AE·loss 옵션은 기존 실행과 같게 유지한다.

### 4.4. 선택: AE부터 직접 학습

공개 AE를 사용한다면 이 단계는 생략한다. 직접 학습하려면 먼저 통계를 계산한다.

```bash
python - <<'PY'
from utils.cal_mean_std import mean_variance
mean_variance('datasets/HumanML3D/new_joint_vecs', 'datasets/HumanML3D', 22)
PY
```

`train_AE.py`는 평가를 항상 실행하므로 평가 모델과 GloVe도 필요하다.

```bash
mkdir -p checkpoints/t2m
wget -c 'https://huggingface.co/cr8br0ze/evaluators_humanml3d/resolve/main/evaluators_humanml3d.zip' \
  -O checkpoints/t2m/evaluators_humanml3d.zip
unzip -n checkpoints/t2m/evaluators_humanml3d.zip -d checkpoints/t2m
python -m pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view' -O glove.zip
unzip -n glove.zip

python train_AE.py --name AE_scratch --dataset_name t2m \
  --batch_size 256 --epoch 50 --lr_decay 0.05
```

이후 생성 모델 학습 명령에서는 `--ae_name AE_scratch`를 사용한다. AE 학습에 사용한 `Mean.npy`, `Std.npy`를 그대로 유지한다.
