# Chess rating training · PyTorch

Parquet 기보의 최대 128 ply를 CNN–Transformer로 읽고 백·흑 레이팅을 예측합니다.
Python 3.12.12, PyTorch 2.10.0을 사용합니다. Linux에서는 공식 CUDA 12.8 wheel,
macOS에서는 CPU용 네이티브 wheel을 설치합니다.

```bash
uv sync --locked
uv run train /data/lichess_monthly \
  --epochs 10 --batch-size 128 \
  --checkpoint-dir outputs/checkpoints --output-dir outputs/run
```

디렉터리 바로 아래의 `.parquet`를 파일명 순으로 읽습니다. 파일을 여러 개 직접
지정하면 인자 순서를 유지합니다. 기본 검증 비율은 5%, 배치 크기는 128이며
Python 함수와 CLI의 기본값이 같습니다. `--max-games 2000 --epochs 1`로 작은 실행을
확인할 수 있습니다. `--max-games 0`은 전체 파일 합계의 모든 게임을 뜻합니다.

## 실행 환경

`--device auto`는 CUDA가 있으면 사용하고, 없으면 CPU를 사용합니다.
`--device cpu` 또는 `--device cuda`로 고정할 수 있습니다.
여러 GPU는 `torchrun`으로 GPU당 프로세스 하나를 실행하여
`DistributedDataParallel`(DDP)을 사용합니다. CUDA 통신은 NCCL을 사용하며
각 프로세스는 `LOCAL_RANK`에 해당하는 GPU만 사용합니다.
GPU 선택은 `CUDA_VISIBLE_DEVICES`로 제한합니다.
일반 `uv run train` / `python -m train` 실행은 보이는 GPU가 많아도 **한 GPU**를 사용합니다.
이 문서의 실행 예제와 지원 범위는 단일 머신이며, CPU 분산 테스트는 Gloo를 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc-per-node=2 \
  -m train /data/lichess_monthly \
  --batch-size 128 --precision bfloat16
```

`--batch-size`는 **전체 프로세스 합산 배치**입니다. 위 설정은 GPU당 64게임이며
프로세스 수의 배수여야 합니다. GPU 4개면 `--nproc-per-node=4`로 바꿉니다.
학습률은 에포크 종료마다 StepLR로 조정합니다. 기본값은 1에포크 `1e-4`,
2에포크 `3e-5`, 3에포크 `9e-6`이며 `--lr-step-size`와 `--lr-gamma`로
간격과 배율을 바꿀 수 있습니다. 데이터는 분할 후 각 rank에 연속 구간을
배정하여 해당 구간만 읽고 복원하고, 구간 안에서 에포크별로 셔플합니다.
모든 rank의 학습 배치 크기와 스텝 수를 맞추기 위해 학습 끝부분의 최대
`프로세스 수 - 1`게임을 제외합니다. 나머지 마지막 부분 배치는 보존하며,
제외 수는 로그와 `run.json`의 `dropped_train_games`에 기록합니다.
학습 게임 수가 프로세스 수보다 작으면 실행을 거부합니다.
검증은 게임을 제외하거나 중복하지 않으며, 검증 데이터가 없는 rank도 처리합니다.
에포크 지표는 모든 rank의 오차 합과 타깃 수를 합산합니다.
진행 막대의 중간 지표는 rank 0 기준이고, 에포크 요약과 저장된 지표는 전체 기준입니다.

기본 정밀도는 `float32`입니다. `float16`은 CUDA에서 GradScaler와 함께 사용하고,
`bfloat16`은 CPU 또는 지원하는 CUDA GPU에서 사용합니다. 타깃은 `(rating - 1660) / 400`으로
표준화하고 출력·학습 MSE는 float32로 계산합니다. `loss`는 표준화된 MSE이며,
`origin_mae`는 float64에서 오차에 400을 곱해 원래 레이팅 단위로 집계합니다.
매 학습 배치에서 역전파 후 전체 gradient L2 norm을 상수 `1.0`으로 제한합니다.
GradScaler 사용 시 unscale 후 clipping하며, norm이 NaN/Inf이면 가중치 갱신 전에 중단합니다.
`--verbose 1`은 배치 진행률, `2`는 에포크 요약,
`0`은 학습 진행 표시를 끕니다. 데이터 적재 로그는 별도로 출력됩니다.

## 서버에서 uv 가상환경으로 실행

uv와 NVIDIA 드라이버가 설치된 서버의 `train` 프로젝트 디렉터리에서 실행합니다.
`pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `src/`를 함께 배치하세요.

```bash
uv python install 3.12.12
uv sync --locked
source .venv/bin/activate

python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
CUDA_VISIBLE_DEVICES=0 python -m train /data/lichess_monthly \
  --device cuda --max-games 2000 --epochs 1 --batch-size 32 \
  --checkpoint-dir outputs/smoke/checkpoints --output-dir outputs/smoke/run
```

`uv sync --locked`가 프로젝트의 `.venv`를 만들고 고정된 의존성을 설치합니다.
작은 실행을 확인한 뒤 본 학습을 시작합니다. `uv run --locked`를 쓰면 가상환경을
직접 활성화하지 않아도 같은 `.venv`에서 실행됩니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc-per-node=2 \
  -m train /data/lichess_monthly \
  --device cuda --epochs 10 --batch-size 128 --precision bfloat16 \
  --checkpoint-dir outputs/checkpoints --output-dir outputs/run
```

한 GPU만 사용할 때는 `CUDA_VISIBLE_DEVICES=0`으로 지정합니다.
서버 예제는 `--device cuda`를 명시하므로 GPU를 사용할 수 없으면 오류로 종료합니다.
의존성을 변경할 때만 `uv lock`으로 lock 파일을 갱신한 뒤 `uv sync --locked`로 적용하세요.

## 코드 구조

```text
src/train/
  main.py          CLI, 실행 설정, 에포크 흐름
  distributed.py   torchrun 초기화·정리, rank/프로세스 수
  engine.py        공통 학습·검증 루프와 MSE/MAE 집계
  checkpoint.py    원자적 저장과 재시작
  models/
    cnn.py         패딩 포함 고정 크기로 CNN 적용 후 패딩 특징 마스킹
    transformer.py 사전 정규화 attention/FFN 블록
    rating.py      위치 임베딩, 블록 4개, 마스킹 평균, 레이팅 출력
  data/
    parquet.py     이동·레이팅 열과 Arrow 배열 처리
    prefetch.py    현재 파일 + 다음 파일 비동기 적재
    batch_prefetch.py  제한된 대기열로 CPU 학습 배치 미리 준비
    decoder.py     이동 복원(Python 참조 구현, Numba 가속)
    batches.py     파일 경계를 연결한 보드 배치
    split.py       고정 학습·검증 범위
    dataset.py     에포크별 셔플과 PyTorch 텐서 배치
    manifest.py    재시작 시 데이터·전처리 동일성 확인
```

모든 모듈을 설치 가능한 `train` 패키지 안으로 모았습니다. `PYTHONPATH`를 직접
추가하지 않아도 됩니다. 설치 후 `uv run train`, `python -m train`, `python main.py`는
같은 학습을 실행합니다.

모델은 Conv2d 3개(32/64/128 채널), 128차원 임베딩, 4-head Transformer 블록 4개,
256차원 FFN, 마스킹 평균과 백·흑 출력으로 구성됩니다. 체스 시계를 위한
2→128차원 투영층이 추가됩니다. 이전 TensorFlow 모델과는 초기화·연산 구현도 다릅니다.

## 데이터와 메모리

입력은 `((boards, valid_steps, clocks), ratings)`입니다.

- 보드: `int8 [B, 128, 8, 8]`, 빈 칸=0, 백=1..6, 흑=-1..-6.
- 마스크: `bool [B, 128]`.
- 시계: `float32 [B, 128, 2]`, 각 수의 `[남은 초, 기록 존재 여부(0/1)]`.
  실제 0초는 `[0, 1]`, 누락/패딩은 `[0, 0]`입니다.
- 레이팅: 백·흑 순서의 `float32 [B, 2]`.

각 보드는 수를 둔 직후의 상태입니다. 초기 보드는 제외하고 128 ply를 넘으면 자릅니다.
캐슬링·앙파상·승격을 복원합니다. 표준 초기 배치에서 시작하는 정상 기보를 전제로 하며,
완전한 합법수 검증기는 아닙니다. 빈/null 기보는 전부 패딩하고 빈 출발 칸을 만나면
직전 수까지만 사용합니다. 빈 기보에서도 모델 출력과 역전파가 유한하도록 처리합니다.
시계는 Parquet의 `ply_list.time`에서 읽으며, 해당 수를 둔 플레이어의 수 직후 잔여
시간(초)입니다. 보드와 동일하게 최대 128 ply로 자르고, 복원이 중단된 이후는 패딩합니다.
null 시계 또는 `time` 필드가 없는 기존 데이터는 기록 없음으로 처리합니다.
음수·NaN·무한대 시계값은 오류입니다. 경기 유형과 결과는 읽거나 입력하지 않습니다.

모델 내부에서 관측된 시계값만 `(log1p(남은 초) - 4.0) / 1.2`로 정규화합니다.
4.0과 1.2는 전체 데이터에서 측정한 통계가 아닌 임시 추정값이며 모델 buffer로 저장되어
학습·검증·추론에 동일하게 적용됩니다. 호출자는 정규화하지 않은 초를 전달해야 합니다.
`[정규화 시간, 기록 존재 여부]`를 `Linear(2, 128, bias=False)`로 투영해 각 수의
CNN 출력·위치 임베딩에 더합니다. 누락/패딩은 투영 전 `[0, 0]`으로 처리하므로
시간 임베딩도 0입니다. 추가 학습 파라미터는 256개이며 회귀층은 그대로입니다.
전부 패딩인 빈 기보는 기존처럼 마스킹 평균 결과가 0 벡터입니다.

실행 명령은 동일하지만 경기 유형 버전 및 이전 보드 전용 모델과 가중치 구조가 다릅니다.
기존 체크포인트를 재개하지 말고 새 체크포인트·출력 디렉터리에서 학습하세요.
시계 버전에서 저장한 체크포인트는 기존 재개 기능을 사용할 수 있습니다.

파일을 순서대로 연결한 앞부분은 학습, 뒷부분은 검증입니다. 분할한 뒤 학습 데이터만
셔플하므로 검증 경계를 넘지 않습니다. 플레이어별 분할이나 별도 테스트 세트는 없습니다.

학습은 rank마다 `max(해당 rank의 batch_size, shuffle_buffer)`개 게임 블록 안에서 무작위 순열을 만들며,
에포크 번호와 시드로 순서를 결정합니다. 기본 `shuffle_buffer=4096`입니다.
기존 입력 파이프라인의 교체식 shuffle과 순서는 다르며 전체 데이터 무작위 셔플은 아닙니다.
마지막 부분 배치를 보존하고 MSE/MAE를 실제 타깃 개수로 가중 집계합니다.
DDP에서는 위 실행 환경 절의 학습 끝부분 제외 규칙을 먼저 적용합니다.

각 rank가 현재 파일과 다음 파일의 담당 구간을 Arrow로 적재합니다. 구간 경계가
row group 내부이면 해당 row group 읽기는 겹칠 수 있습니다. 파일 버퍼·복원 블록·
prefetch 대기열은 프로세스마다 별도로 존재하므로 GPU 수를 늘릴 때 CPU 메모리와
디스크 읽기 부하도 확인해야 합니다. 이 두 파일에는 바이트 단위
메모리 상한이 없습니다. 보드 배열은 블록 단위로 복원하며 전체 게임 보드를 캐시하지 않습니다.
`--read-batch-size`는 Arrow에서 디코더에 넘길 행 수이고 파일 적재량 제한이 아닙니다.
파일 로딩과 별도로 생산자 스레드 하나가 보드 복원·셔플·텐서 배치를 미리 준비합니다.
기본 `--prefetch-batches 2`는 각 rank의 준비된 CPU 배치 대기열을 최대 2개로 제한합니다.
`--prefetch-batches 0`은 기존처럼 학습 루프에서 배치를 준비하는 비교 경로입니다.
셔플 순서와 마지막 부분 배치는 같으며 CUDA 전송은 기존 학습 루프에서 수행합니다.
대기열 외에 생산 중인 배치와 셔플 블록이 존재하고, 텐서가 블록 배열을 공유하므로
추가 메모리 전체가 단순히 배치 2개 크기로 제한되는 것은 아닙니다. 전체 보드 캐시는 없습니다.
`GameDataset`은 이미 배치된 텐서를 반환합니다. 직접 순회하거나
`DataLoader(dataset, batch_size=None, num_workers=0)`로 사용하세요.
추가 DataLoader 작업자는 데이터 중복과 파일 버퍼 복제를 방지하기 위해 거부합니다.
직접 iterator를 중간에 중단하면 `close()` 또는 `contextlib.closing`으로 정리하세요.
중단 시 생산자와 파일 로더에 취소를 전달하고 종료를 기다립니다. 이미 진행 중인
파일 row group 읽기나 decoder 호출은 즉시 중단하지 못하며 완료 후 정리합니다.
이 변경 전 체크포인트는 기존 데이터·코드 동일성 검사로 재개가 거부됩니다.

## 저장과 재시작

### 에포크 체크포인트 성능 비교

학습을 중단한 뒤, 같은 게임에서 두 체크포인트의 MAE를 비교할 수 있습니다.
서버의 `train` 프로젝트 디렉터리에서 실행하며 GPU가 있으면 한 장을 사용합니다.

```bash
python check_checkpoints.py \
  outputs/checkpoints/epoch-000001.pt \
  outputs/checkpoints/epoch-000002.pt \
  --sample-games 4096 --batch-size 128
```

체크포인트에 기록된 원래 Parquet 파일과 학습/검증 분할을 사용합니다. 학습 범위와
검증 범위의 앞·중간·뒤에서 각각 같은 연속 게임을 평가합니다. 작은 범위에서
구간이 겹치면 중복 구간은 한 번만 표시합니다. `delta_last_minus_first`가 양수면
나중 체크포인트의 MAE가 더 나쁩니다. 이 결과는 고정된 일부 게임의 평가이며
전체 학습·검증 MAE를 대신하지 않습니다. 원본 데이터 파일이나 학습 코드가
변경됐다면 비교를 중단합니다. 이 파일은 학습을 진행하거나 체크포인트를
수정하지 않습니다.

검증까지 완료한 에포크마다 `epoch-000001.pt` 등을 저장하고 `latest.json`을 갱신합니다.
임시 파일을 모두 쓴 뒤 교체하므로 저장 실패 시 이전 최신 체크포인트를 유지합니다.
rank 0만 파일을 저장하며 가중치에 `module.` 접두사를 붙이지 않습니다.
가중치, Adam·StepLR·GradScaler 상태, **rank별** PyTorch CPU/CUDA 난수 상태와 전체 지표 이력을 저장합니다.
스텝 중간 재시작은 지원하지 않습니다.

```bash
uv run train /data/lichess_monthly \
  --epochs 10 --batch-size 128 \
  --resume-from outputs/checkpoints \
  --checkpoint-dir outputs/checkpoints --output-dir outputs/run-resumed
```

DDP 재개는 처음과 같은 `torchrun --standalone --nproc-per-node=N -m train ...`
명령에 `--resume-from`을 추가합니다. 프로세스 수와 배치 크기도 동일해야 합니다.
StepLR 도입 직전 코드로 만든 **1에포크 체크포인트만** 같은 데이터·설정에서
재개할 수 있습니다. 재개 시 2에포크 학습률을 `3e-5`로 설정합니다.
기존 2에포크 체크포인트는 이미 높은 학습률로 진행됐으므로 이 전환을 허용하지 않습니다.
다음처럼 명시적으로 1에포크 파일을 선택하고 새 체크포인트 디렉터리를 사용하세요.
기존 `torchrun` 명령의 데이터 경로·GPU 수·배치 크기·정밀도를 그대로 두고
다음 옵션을 추가합니다.

```bash
--epochs 3 \
--resume-from outputs/checkpoints/epoch-000001.pt \
--checkpoint-dir outputs/step-lr-checkpoints \
--output-dir outputs/step-lr-run
```

`--epochs 3`은 전체 목표 에포크 수이며, 첫 실행과 같은 GPU 수가 필요합니다.
체크포인트 형식은 v3입니다. 이전 DataParallel/v2 체크포인트는 재개하지 않으며
새 체크포인트·출력 디렉터리에서 학습을 시작하세요. 경기 유형 버전 및 보드 전용 모델은
`clock_projection.weight`와 정규화 buffer가 없어 그대로 불러오거나 재개할 수 없습니다.

`--epochs`는 추가 횟수가 아니라 최종 에포크 번호입니다. 특정 `.pt` 파일도 지정할 수 있습니다.
데이터 경로·크기·수정 시각, 전처리/모델/학습 코드, 배치·시드·정밀도·초기 학습률·StepLR 설정·장치 종류·GPU 수·
PyTorch 버전을 비교하여 달라지면 재시작을 거부합니다. 파일 내용 전체를 해시하지는 않습니다.
GPU 실행의 비결정적 연산까지 수치 재현을 보장하지는 않습니다.

최종 결과는 `model.pt`(가중치), `history.json`(전체 지표), `run.json`(실행 정보)입니다.
예측은 다음처럼 합니다.

```python
import torch
from train.models import build_model

model = build_model()
model.load_state_dict(torch.load("outputs/run/model.pt", map_location="cpu", weights_only=True))
model.eval()
with torch.inference_mode():
    # remaining_seconds, clock_present: [B, T], 실제 수별 기록을 사용합니다.
    # 누락된 시간은 0으로 대체하되 존재 여부는 False로 유지합니다.
    seconds = torch.where(clock_present, remaining_seconds, 0.0)
    clocks = torch.stack((seconds, clock_present.float()), dim=-1)
    normalized_ratings = model(boards, valid_steps, clocks)  # [B, 2]
    ratings = normalized_ratings * 400.0 + 1660.0  # 원래 레이팅 단위
```

## 이전 사항과 검증

TensorFlow/Keras 의존성, Keras 직렬화·콜백·legacy 호환 코드, XLA/JIT 옵션,
선택형 패딩 처리, 미사용 프로파일러, 중복 `src/main.py` 진입점을 제거했습니다.
`--jit-compile`, `--skip-padding`, `--generator-batch-size`, `--prefetch`는 제거했고,
정밀도 이름은 `float32`, `float16`, `bfloat16`으로 통일했습니다.
패딩 보드도 CNN에서 처리해 GPU당 CNN 입력 크기를 `B × 128 + 1`로 유지합니다.
`+1`은 dummy 보드이며 마지막 부분 배치는 B가 달라질 수 있습니다. 패딩 입력은
빈 보드로 바꾸고 CNN 출력의 패딩 위치는 0으로 마스킹합니다. `BoardEncoder` 이름과
가중치 구조는 유지합니다. 이전 가변 크기 코드로 저장한 체크포인트는 기존 코드 해시
검사 때문에 `--resume-from` 재개가 거부됩니다.
기존 `.keras` 파일은 로드/변환하지 않으며
새 PyTorch 학습과 별도 출력 경로를 사용해야 합니다.

```bash
uv run python -m unittest discover -s tests -v
```

테스트는 특수 수 복원, 데이터 분할·파일 경계·부분 배치, 비동기 로더 정리와 오류 전달,
마스킹·빈 기보·역전파·CPU bfloat16, 지표 집계, 실제 모델의 연속 학습과 저장 후 재시작
결과 일치를 확인합니다. 로컬 검증은 macOS CPU 기준이며 Linux 서버의
CUDA·복수 GPU 실행은 대상 서버에서 확인해야 합니다.
DDP는 실제 Gloo 프로세스 2개에서 전역 배치와 gradient 일치, 불균등/빈 검증 구간,
rank별 난수 상태를 포함한 재개 일치, `torchrun`을 통한 실제 CNN–Transformer
학습과 저장을 검사합니다. 기존 `benchmarks/` 도구는 DataParallel 비교용으로
남겨 두었으며 DDP 성능 측정용이 아닙니다.
