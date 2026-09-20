# TensorFlow 2.20 + uv Apptainer 이미지

Linux x86_64 서버용입니다. Python 3.12.12, uv 0.12.1,
TensorFlow 2.20.0, Keras 3, Numba, NumPy, PyArrow를 설치합니다.
기반 이미지는 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`입니다.
CUDA 12.6.3, cuDNN 9.5.1, NCCL은 기반 이미지에서 가져오고,
`tensorflow==2.20.0`을 설치합니다. `and-cuda` extra와 NVIDIA Python 패키지는
설치하지 않아 CUDA·cuDNN이 중복되지 않습니다.
`cuda-nvcc-12-6` 패키지를 추가해 nvcc, ptxas와 XLA용 libdevice도 제공합니다.
Python은 uv로 `/opt/python`에 설치해 일반 사용자도 실행할 수 있습니다.
호스트에는 NVIDIA 드라이버와 Apptainer가 필요하며 CUDA Toolkit 설치는 필요 없습니다.

TensorFlow 2.20의 공식 테스트 빌드 조합은 CUDA 12.5 + cuDNN 9.3입니다.
CUDA 12.5.1의 NVIDIA cuDNN 이미지에는 cuDNN 9.2가 들어 있어 이보다 새로운
CUDA 12.6.3 + cuDNN 9.5.1 조합을 선택했습니다. 같은 메이저 버전의 호환성을
전제로 한 구성이며 TensorFlow 공식 테스트 조합과 정확히 일치하지 않습니다.
아래 GPU 검사와 실제 학습으로 서버에서 확인해야 합니다.

기존 프로젝트의 `pyproject.toml`은 TensorFlow 2.17 및 tf-keras 환경을 유지합니다.
이 이미지는 별도 `requirements.lock`으로 설치하므로 기존 환경과 독립적으로 시험합니다.
컨테이너 안에서는 `uv sync`나 일반 `uv run`을 실행하지 마세요.
프로젝트의 2.17 환경으로 다시 동기화될 수 있습니다. 학습 실행은 아래 스크립트를 사용합니다.
이미지에는 `tf-keras`를 설치하지 않고 `TF_USE_LEGACY_KERAS=0`을 기본 설정합니다.

## 빌드

`train` 프로젝트 전체를 서버로 복사한 뒤 프로젝트 루트에서 실행합니다.
빌드에는 인터넷이 필요하지만 GPU는 필요 없습니다. 이미지와 빌드 임시 공간에
수 GB 이상을 확보하세요. macOS에서 직접 빌드하는 구성이 아닙니다.

```bash
cd /path/to/train
apptainer build --fakeroot tensorflow-2.20-uv.sif apptainer/tensorflow.def
```

서버에서 fakeroot가 허용되지 않으면 관리자가 다음 명령으로 빌드하거나,
다른 Linux x86_64 빌드 머신에서 생성한 SIF를 복사합니다.

```bash
sudo apptainer build tensorflow-2.20-uv.sif apptainer/tensorflow.def
```

빌드의 `%test`는 버전, Keras 3 사용, tf-keras 및 NVIDIA Python 패키지 부재,
cuDNN 라이브러리 로딩·버전, CPU 연산을 검사합니다.
Python 패키지는 버전과 해시를 고정했습니다. 기반 OCI 이미지는 버전 태그이므로
비트 단위 재현성이 필요하면 해당 태그의 digest와 추가 apt 패키지 버전까지 고정해야 합니다.

## 서버 GPU 확인

GPU가 할당된 노드에서 실행합니다. 드라이버는 이미지에 포함되지 않으며,
서버 GPU 모델과 드라이버 호환성은 이 검사로 확인해야 합니다.
아래 검사는 GPU가 없거나 GPU 연산이 실패하면 오류로 종료됩니다.

```bash
nvidia-smi
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 \
  apptainer exec --nv tensorflow-2.20-uv.sif \
  /opt/venv/bin/python /opt/chess/check_runtime.py --gpu
```

TensorFlow·Keras·CUDA 빌드 정보와 GPU 목록을 출력한 뒤 cuBLAS 행렬곱,
cuDNN 합성곱 및 역전파를 검사합니다. 멀티 GPU 통신과 실제 학습 성능은 별도로
짧은 학습으로 확인하세요. 스케줄러를 사용하면 배정받은 GPU만 선택하세요.
기존 `APPTAINERENV_TF_USE_LEGACY_KERAS=1` 또는
`SINGULARITYENV_TF_USE_LEGACY_KERAS=1` 설정은 제거해야 합니다.

## 학습

```bash
APPTAINER_PYTHON=/opt/venv/bin/python \
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 \
  bash run_train_apptainer.sh ./tensorflow-2.20-uv.sif /data/lichess_monthly \
  --epochs 10 --batch-size 128 \
  --checkpoint-dir outputs/tf220/checkpoints \
  --output-dir outputs/tf220/run
```

학습 코드는 기존 실행 스크립트가 마운트하므로 코드 변경만으로는 이미지 재빌드가
필요하지 않습니다. `/opt/venv`는 SIF에 포함된 읽기 전용 환경입니다.
기존 legacy Keras 체크포인트의 재시작 호환성은 보장되지 않으며 프로젝트는
런타임 정보를 검사합니다. 새 출력 경로에서 학습을 시작하세요.

프로젝트 회귀 테스트도 새 이미지에서 실행할 수 있습니다.

```bash
apptainer exec --nv --bind "$PWD:/workspace/train" --pwd /workspace/train \
  --env PYTHONPATH=/workspace/train/src tensorflow-2.20-uv.sif \
  /opt/venv/bin/python -m unittest discover -s tests
```

## 의존성 갱신

`requirements.in`을 수정한 뒤 uv로 Linux 대상 lock을 재생성하고 이미지를 다시 빌드합니다.

```bash
uv pip compile apptainer/requirements.in --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --only-binary :all: \
  --generate-hashes -o apptainer/requirements.lock
```

TensorFlow 버전을 바꾸면 `.def`의 라벨과 `check_runtime.py`의 버전 검사도 함께 수정합니다.

참고: [NVIDIA CUDA 이미지](https://hub.docker.com/r/nvidia/cuda),
[기반 이미지의 cuDNN 버전](https://gitlab.com/nvidia/container-images/cuda/-/raw/master/dist/12.6.3/ubuntu2404/runtime/cudnn/Dockerfile),
[TensorFlow 공식 빌드 조합](https://www.tensorflow.org/install/source#gpu),
[TensorFlow GPU 설치](https://www.tensorflow.org/install/pip),
[uv 컨테이너 사용](https://docs.astral.sh/uv/guides/integration/docker/),
[Apptainer 정의 파일](https://apptainer.org/docs/user/latest/definition_files.html),
[Apptainer GPU 전달](https://apptainer.org/docs/user/latest/gpu.html).
