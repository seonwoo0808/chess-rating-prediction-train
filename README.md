# Chess rating training

모델은 `src/models`, 데이터 전처리는 `src/data`에 있습니다.
현재 `src`를 Python 경로에 추가해 사용합니다.

```bash
uv sync
uv run python -m unittest discover -s tests
```

본 학습은 `train` 명령으로 실행합니다. 파일 인자에 디렉터리를 지정하면
그 디렉터리 바로 아래의 모든 `.parquet` 파일을 파일명 순서로 자동 수집합니다.
파일을 직접 여러 개 지정하면 지정한 순서를 유지합니다.

```bash
uv run train /data/lichess_monthly \
  --epochs 10 --batch-size 128 \
  --checkpoint-dir outputs/checkpoints \
  --checkpoint-every 10000 \
  --output-dir outputs/run
```

재시작은 같은 데이터·설정으로 최신 체크포인트를 지정합니다.

```bash
uv run train /data/lichess_db_standard_rated_2024-09.parquet \
  /data/lichess_db_standard_rated_2024-10.parquet \
  --epochs 10 --resume-from outputs/checkpoints \
  --checkpoint-dir outputs/checkpoints \
  --output-dir outputs/run-resumed
```

Apptainer 서버에서는 프로젝트의 스크립트가 이미지와 데이터 디렉터리를
받아 같은 학습 진입점을 실행합니다. 데이터는 읽기 전용으로 마운트하고,
체크포인트와 결과는 프로젝트 디렉터리에 저장합니다.

```bash
chmod +x run_train_apptainer.sh
CUDA_VISIBLE_DEVICES=2 \
  ./run_train_apptainer.sh /opt/tensorflow.sif /data/lichess_monthly \
  --epochs 10 --batch-size 1024 \
  --checkpoint-dir outputs/checkpoints \
  --checkpoint-every 100000
```

스크립트 형식은 `IMAGE DATA_DIR [TRAIN_OPTIONS...]`입니다.
`--nv`로 GPU를 전달하고, 컨테이너 안에서 `/workspace/train/src`를
Python 경로로 설정합니다. TensorFlow 자동 탐색 임계값은
`TF_AUTOTUNE_THRESHOLD` 환경변수로 바꿀 수 있으며 기본값은 `1`입니다.

```python
import logging
from pathlib import Path
from data import build_datasets, warmup_decoder
from models import build_model

logging.basicConfig(level=logging.INFO)
warmup_decoder("numba")
files = sorted(Path("/data").glob("lichess_db_standard_rated_*.parquet"))
train_ds, val_ds = build_datasets(
    files, batch_size=128, validation_size=0.05,
    max_games=None,  # 작은 실행 확인은 2000, None 또는 0이면 전체
)
model = build_model()
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

## 정확한 중간 재시작

스텝 단위 체크포인트를 사용할 때는 일반 `model.fit` 대신
`fit_resumable`을 사용합니다. Keras의 Python generator 데이터셋 iterator는
그 자체를 저장할 수 없으므로, 체크포인트에는 모델·옵티마이저·지표·난수 상태와
데이터 매니페스트를 저장하고 재시작 시 같은 epoch의 앞 배치를 결정적으로 다시
재생한 뒤 저장된 다음 스텝부터 이어갑니다.

```python
from callbacks import StepCheckpoint, fit_resumable
from data import ResumableData

data = ResumableData(
    files, batch_size=128, validation_size=0.05,
    generator_batch_size=512, decoder="numba", seed=42,
)
checkpoint = StepCheckpoint("outputs/checkpoints", every_n_steps=10_000)

model, history = fit_resumable(
    build_model(), data, epochs=10, checkpoint=checkpoint,
)
```

재시작할 때는 새 모델을 만들지 않고 `model=None`과 체크포인트 디렉터리를
전달합니다.

```python
checkpoint = StepCheckpoint("outputs/checkpoints", every_n_steps=10_000)
model, history = fit_resumable(
    None, data, epochs=10, checkpoint=checkpoint,
    resume_from="outputs/checkpoints",
)
```

체크포인트 디렉터리에는 `latest.json` 포인터와 원자적으로 완성된
`step-.../` 디렉터리가 생깁니다. 각 스냅샷은 다음을 포함합니다.

- 저장 시점의 0부터 시작하는 `epoch`, 해당 epoch에서 완료한 `step_in_epoch`, 전체 `global_step`
- 모델 가중치와 Keras 옵티마이저 슬롯·반복 횟수
- 학습 지표, Python/NumPy/TensorFlow 난수 상태, 런타임 버전·정밀도
- 파일 경로·크기·수정 시각, 전처리 코드 해시, 분할·배치 설정

`every_n_steps`는 완료된 학습 배치 기준입니다. epoch 마지막에는 검증이 끝난
뒤 항상 체크포인트를 하나 더 저장하므로, epoch 완료 지점에서 재시작하면
검증을 중복하지 않습니다. 중간 스텝 체크포인트에서 재시작하면 해당 epoch의
검증은 정상적으로 마지막 학습 배치 뒤에 한 번 수행됩니다.

재시작 시 데이터 파일 순서·내용 메타데이터·전처리 코드·배치 설정·TensorFlow/
Keras 버전·정밀도·분산 전략이 달라지면 즉시 거부합니다. 파일이 변경된 경우
자동으로 다른 데이터로 계속 학습하지 않습니다. 정확한 재현을 위해
`steps_per_execution=1` 조건을 사용합니다.

## 전처리 모듈

- `parquet.py`: 이동·레이팅만 읽습니다. 실제 Parquet 중첩 경로를 사용해 잔여 시간은 제외합니다.
- `prefetch.py`: 현재 파일을 소비하는 동안 다음 파일을 백그라운드 스레드에서 Arrow로 적재합니다.
- `decoder.py`: 2바이트 이동을 재생합니다. 캐슬링·앙파상·승격을 처리하며 Python/Numba 경로를 제공합니다.
- `batches.py`: 보드를 제한된 크기의 블록으로 생성합니다. 파일 경계에서도 배치를 이어 채웁니다.
- `split.py`: 파일들을 순서대로 이어 붙인 경기 범위에서 앞부분은 학습, 뒷부분은 검증으로 고정 분할합니다.
- `dataset.py`: 게임 단위 shuffle 후 학습 배치와 prefetch를 구성합니다. 검증은 섞지 않습니다.
- `profiling.py`: 기존 스트리밍 입력 경로의 CPU 단계별 시간을 측정합니다. 비동기 로더 전체 성능 측정은 아닙니다.

출력은 `((boards, valid_steps), ratings)`입니다.
보드는 `int8 [B, 128, 8, 8]`, 마스크는 `bool [B, 128]`,
레이팅은 백·흑 순서의 `float32 [B, 2]`입니다.
각 보드는 수를 둔 직후 상태이며 초기 보드는 포함하지 않습니다.
128 ply보다 긴 수순은 자르고 짧은 수순은 0/False로 패딩합니다.

기존 처리와 동일하게 `ply_list`가 null/빈 목록이면 전부 패딩합니다.
빈 출발 칸을 만난 수순은 그 직전까지만 사용합니다.
완전한 합법수 검증기는 아니므로 표준 초기 배치에서 시작하는 정상 기보를 전제로 합니다.
`game_type`, `result`, 잔여 시간은 입력에 사용하지 않습니다.
분할은 무작위·플레이어별 분할이 아니며 별도 최종 테스트 세트는 없습니다.

## 비동기 파일 전환

`build_datasets`는 단일 파일 경로나 파일 경로 목록을 받습니다.
현재 파일 A + 다음 파일 B만 유지하며, A의 Arrow 참조를 해제한 뒤 B를 소비하고 C를 읽습니다.
파일 개수는 로더 iterator당 최대 2개로 고정되며 메모리 용량 상한은 없습니다.
별도 iterator를 동시에 여러 개 실행하면 각각의 파일 버퍼가 생깁니다.

각 파일의 이동·레이팅을 Arrow 테이블로 적재하고, 보드는 배치 단위로만 복원합니다.
처음에는 첫 파일 적재가 끝날 때까지 기다립니다. 다음 파일이 늦으면 전환 때 기다립니다.
배경 읽기는 서버에서 측정한 방식대로 row group 순서로 진행합니다.
백그라운드 파일 로딩 작업자는 1개입니다.
Arrow 청크를 합칠 때 전체 배열 복사를 하지 않습니다.
`read_batch_size`는 적재 후 Arrow에서 보드 변환으로 넘기는 경기 수이며, 파일 적재 용량 제한이 아닙니다.

파일 경계에서도 생성 배치를 이어 채우고, 마지막 부분 배치를 버리지 않습니다.
`max_games`는 파일별이 아니라 전체 파일 목록 앞부분의 합계에 적용합니다.
학습·검증 경계가 파일 내부이면 해당 범위와 겹치는 row group만 읽고 정확한 행 범위를 사용합니다.
경계 row group이 양쪽에서 읽힐 수 있지만 같은 경기가 양쪽에 포함되지는 않습니다.
파일 순서는 전달한 순서로 고정되고 학습 경기만 shuffle 버퍼에서 매 epoch 다시 섞습니다.
월별 순서가 필요하면 예제처럼 경로를 정렬하세요. 매 epoch 파일을 다시 읽습니다.

학습 데이터셋은 `model.fit(train_ds, validation_data=val_ds)`처럼 사용합니다.
저수준 `async_row_batches`를 직접 쓰다가 중단한다면 `contextlib.closing`으로 감싸거나 `close()`를 호출하세요.
종료 시 작업자를 정리하며, 이미 실행 중인 row group 읽기가 끝난 뒤 취소됩니다.
파일 읽기 오류는 소비자 쪽으로 전달됩니다. TensorFlow를 통해 읽으면 TensorFlow 오류로 감싸질 수 있습니다.
참조 해제 후에도 메모리 할당기가 재사용 목적으로 RSS를 유지할 수 있습니다.
전체 학습 데이터를 보드 형태로 캐시하지 않습니다.

`src/train/main.py`에는 같은 옵션을 받는 `run_training()` 함수도 있습니다.
스크립트나 노트북에서는 이 함수를 직접 호출할 수 있습니다.
