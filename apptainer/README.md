# TensorFlow 2.20 + uv Apptainer 이미지

Linux x86_64 서버용입니다. Python 3.12.12, uv 0.12.1,
TensorFlow 2.20.0, Keras 3, Numba, NumPy, PyArrow를 설치합니다.
대상 GPU는 NVIDIA RTX PRO 6000 Blackwell (`sm_120`)입니다.
최종 기반 이미지는 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`이며,
CUDA 12.8.1, cuDNN 9.8.0, NCCL을 제공합니다. ptxas와 libdevice용으로
`cuda-nvcc-12-8`도 설치합니다. 호스트의 NVIDIA 드라이버를 `--nv`로 사용합니다.
현재 서버의 드라이버는 580.173.02입니다.

## TensorFlow 소스 빌드

정의 파일은 uv 바이너리, TensorFlow 빌드, 최종 실행 이미지의 세 단계로 구성됩니다.
빌드 단계에서 공식 TensorFlow `v2.20.0` 소스를 가져와 Bazel 7.4.1과 Clang 20으로
컴파일합니다. TensorFlow의 hermetic 빌드 설정에 CUDA 12.8.1, cuDNN 9.8.0,
Python ABI 3.12, `compute_120`을 명시합니다. `compute_120` 설정은 해당 GPU용
SASS와 PTX를 포함하도록 하는 옵션입니다. 다른 GPU 세대를 대상으로 한 이미지가 아닙니다.
Bazel은 빌드용 CUDA·cuDNN을 별도로 다운로드하며 최종 이미지에는 빌드 도구와
소스 트리 대신 생성된 wheel만 복사합니다.

Mac과 서버의 의존성 기준은 같은 `pyproject.toml`입니다. 컨테이너에 복사한
파일에만 `[tool.uv.sources]`를 추가하여 TensorFlow를 직접 만든 wheel 경로로 지정합니다.
따라서 `uv sync`가 PyPI의 일반 TensorFlow wheel로 대체하지 않습니다.
Mac의 파일은 바꾸지 않으므로 기존처럼 `uv sync`로 PyPI 패키지를 설치합니다.
`uv.lock`은 커밋하지 않으며 빌드할 때 새로 해결합니다.
직접 의존성 버전은 고정하지만 간접 의존성·apt 패키지·이미지 태그는
빌드 시점에 따라 달라질 수 있습니다.

소스 wheel과 빌드에 사용한 커밋은 이미지의 `/opt/chess/wheels/`에 보관합니다.
학습 코드는 실행 스크립트가 마운트하며, Python 환경은 `/opt/venv`입니다.
`tf-keras`를 설치하지 않고 `TF_USE_LEGACY_KERAS=0`을 사용합니다.
별도 런타임 검사 스크립트나 `%test` 단계는 추가하지 않습니다.

## 빌드

실제 Linux x86_64 머신에서 실행하세요. GPU는 필요 없지만 인터넷과 상당한
CPU·메모리·디스크 공간이 필요합니다. Mac의 x86 에뮬레이션 VM은 권장하지 않습니다.
계획 기준으로 RAM 64GB 이상, 여유 디스크 150GB 이상을 권장하며 실제 사용량과
소요 시간은 서버에 따라 달라집니다. 수 시간이 걸릴 수 있습니다.
기본 병렬 작업은 8개, Bazel 메모리 스케줄링 예산은 32768MB입니다.
이 예산은 프로세스 메모리의 강제 상한이 아닙니다.

프로젝트 루트에서 다음처럼 실행합니다. 임시 디렉터리는 RAM 기반 `/tmp` 대신
용량이 충분한 디스크 경로로 지정하세요.

```bash
mkdir -p "$PWD/.apptainer-tmp"
APPTAINER_TMPDIR="$PWD/.apptainer-tmp" \
  apptainer build --fakeroot \
  --build-arg BUILD_JOBS=8 --build-arg BUILD_RAM_MB=32768 \
  tensorflow-2.20-blackwell.sif apptainer/tensorflow.def
```

메모리 부족 시 `BUILD_JOBS=4`처럼 병렬 작업 수를 줄이세요.
fakeroot를 지원하지 않는 서버에서는 관리자가 `sudo apptainer build`로 빌드하거나,
다른 Linux x86_64 머신에서 만든 SIF를 복사합니다. 실패 후 재빌드하면
Bazel 컴파일 캐시를 재사용하지 않으므로 전체 빌드가 다시 진행될 수 있습니다.

이 구성은 공식 소스와 빌드 옵션을 대조하고 문법을 검증했지만,
현재 Mac 작업 환경에서 전체 Linux 소스 빌드 또는 Blackwell 실행을 검증하지는 못했습니다.
빌드 완료 후 GPU 서버에서 먼저 작은 배치로 학습을 확인하세요.

## 학습

GPU가 할당된 노드에서 실행하며, 스케줄러가 배정한 GPU만 선택하세요.
기존 `APPTAINERENV_TF_USE_LEGACY_KERAS=1` 또는
`SINGULARITYENV_TF_USE_LEGACY_KERAS=1` 설정은 제거해야 합니다.

```bash
APPTAINER_PYTHON=/opt/venv/bin/python \
APPTAINERENV_CUDA_VISIBLE_DEVICES=1,2 \
  bash run_train_apptainer.sh ./tensorflow-2.20-blackwell.sif /data/lichess_monthly \
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
  --env PYTHONPATH=/workspace/train/src tensorflow-2.20-blackwell.sif \
  /opt/venv/bin/python -m unittest discover -s tests
```

## 의존성 갱신

`pyproject.toml`의 정확한 버전을 수정한 뒤 Mac에서는 `uv sync`,
서버 이미지에는 위의 Apptainer 빌드 명령을 다시 실행합니다.
로컬 `uv.lock`은 자동 갱신되며 `.gitignore`로 제외됩니다.
이미지 빌드는 호스트의 lock 파일을 사용하지 않고 새로 의존성을 해결합니다.
TensorFlow 버전을 바꾸면 `.def`의 소스 태그, wheel 파일명, 버전 확인, 라벨과
해당 버전의 컴파일러·CUDA 빌드 설정도 함께 수정합니다.

참고: [TensorFlow 2.20 빌드 옵션](https://github.com/tensorflow/tensorflow/blob/v2.20.0/.bazelrc),
[CUDA·cuDNN·Clang 지원 설정](https://github.com/tensorflow/tensorflow/blob/v2.20.0/third_party/xla/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl),
[NVIDIA CUDA 이미지](https://hub.docker.com/r/nvidia/cuda),
[Apptainer 정의 파일](https://apptainer.org/docs/user/latest/definition_files.html).
