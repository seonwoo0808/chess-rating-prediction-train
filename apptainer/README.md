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
실제 GPU 학습으로 서버에서 확인해야 합니다.

Mac과 Linux는 동일한 `pyproject.toml`을 사용합니다. 직접 의존성 버전은
`==`로 고정하고, `uv.lock`은 로컬에서만 생성하며 커밋하지 않습니다.
이미지 빌드에는 `pyproject.toml`만 복사하고 `uv sync --no-install-project --no-dev`로
`/opt/venv`에 설치합니다. 학습 소스는 기존 실행 스크립트가 마운트합니다.
Mac에서는 프로젝트 루트에서 `uv sync`로 `.venv`에 설치합니다.
uv가 플랫폼에 맞는 배포 파일을 선택하며, Mac에서는 NVIDIA CUDA를 사용하지 않습니다.
컨테이너의 `/opt/venv`는 읽기 전용이므로 학습은 아래 스크립트로 실행합니다.
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

직접 Python 의존성은 버전을 고정했지만 간접 의존성은 빌드 시점에 해결하므로
빌드마다 달라질 수 있습니다. 기반 OCI 이미지도 버전 태그를 사용하며,
추가 apt 패키지 버전은 고정하지 않습니다.

## 학습

GPU가 할당된 노드에서 실행하며, 스케줄러가 배정한 GPU만 선택하세요.
기존 `APPTAINERENV_TF_USE_LEGACY_KERAS=1` 또는
`SINGULARITYENV_TF_USE_LEGACY_KERAS=1` 설정은 제거해야 합니다.

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

`pyproject.toml`의 정확한 버전을 수정한 뒤 Mac에서는 `uv sync`,
서버 이미지에는 위의 Apptainer 빌드 명령을 다시 실행합니다.
로컬 `uv.lock`은 자동 갱신되며 `.gitignore`로 제외됩니다.
이미지 빌드는 호스트의 lock 파일을 사용하지 않고 새로 의존성을 해결합니다.
TensorFlow 버전을 바꾸면 `.def`의 라벨도 함께 수정합니다.

참고: [NVIDIA CUDA 이미지](https://hub.docker.com/r/nvidia/cuda),
[기반 이미지의 cuDNN 버전](https://gitlab.com/nvidia/container-images/cuda/-/raw/master/dist/12.6.3/ubuntu2404/runtime/cudnn/Dockerfile),
[TensorFlow 공식 빌드 조합](https://www.tensorflow.org/install/source#gpu),
[TensorFlow GPU 설치](https://www.tensorflow.org/install/pip),
[uv 컨테이너 사용](https://docs.astral.sh/uv/guides/integration/docker/),
[Apptainer 정의 파일](https://apptainer.org/docs/user/latest/definition_files.html),
[Apptainer GPU 전달](https://apptainer.org/docs/user/latest/gpu.html).
