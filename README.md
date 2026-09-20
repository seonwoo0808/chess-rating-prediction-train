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
CUDA GPU가 여러 개 보이면 `torch.nn.DataParallel`로 전역 배치를 분배합니다.
입력은 한 프로세스에서 준비하므로 GPU 수에 따라 파일 버퍼가 복제되지 않습니다.
GPU 선택은 `CUDA_VISIBLE_DEVICES`로 제한합니다. DDP/다중 노드 실행은 지원하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run train /data/lichess_monthly \
  --batch-size 128 --precision bfloat16
```

기본 정밀도는 `float32`입니다. `float16`은 CUDA에서 GradScaler와 함께 사용하고,
`bfloat16`은 CPU 또는 지원하는 CUDA GPU에서 사용합니다. 레이팅 출력·MSE·MAE는
float32로 계산합니다. `--verbose 1`은 배치 진행률, `2`는 에포크 요약,
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
CUDA_VISIBLE_DEVICES=0,1 uv run --locked train /data/lichess_monthly \
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
  engine.py        공통 학습·검증 루프와 MSE/MAE 집계
  checkpoint.py    원자적 저장과 재시작
  models/
    cnn.py         유효한 보드만 12채널로 확장하고 CNN 적용
    transformer.py 사전 정규화 attention/FFN 블록
    rating.py      위치 임베딩, 블록 4개, 마스킹 평균, 레이팅 출력
  data/
    parquet.py     이동·레이팅 열과 Arrow 배열 처리
    prefetch.py    현재 파일 + 다음 파일 비동기 적재
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
256차원 FFN, 마스킹 평균과 백·흑 출력으로 구성됩니다. 기존 모델의 크기와 블록
구성은 유지하지만 프레임워크의 초기화·연산 구현이 달라 기존 수치와 일치하지 않습니다.

## 데이터와 메모리

입력은 `((boards, valid_steps), ratings)`입니다.

- 보드: `int8 [B, 128, 8, 8]`, 빈 칸=0, 백=1..6, 흑=-1..-6.
- 마스크: `bool [B, 128]`.
- 레이팅: 백·흑 순서의 `float32 [B, 2]`.

각 보드는 수를 둔 직후의 상태입니다. 초기 보드는 제외하고 128 ply를 넘으면 자릅니다.
캐슬링·앙파상·승격을 복원합니다. 표준 초기 배치에서 시작하는 정상 기보를 전제로 하며,
완전한 합법수 검증기는 아닙니다. 빈/null 기보는 전부 패딩하고 빈 출발 칸을 만나면
직전 수까지만 사용합니다. 빈 기보에서도 모델 출력과 역전파가 유한하도록 처리합니다.
경기 결과·게임 유형·잔여 시간은 입력에 사용하지 않습니다.

파일을 순서대로 연결한 앞부분은 학습, 뒷부분은 검증입니다. 분할한 뒤 학습 데이터만
셔플하므로 검증 경계를 넘지 않습니다. 플레이어별 분할이나 별도 테스트 세트는 없습니다.

학습은 `max(batch_size, shuffle_buffer)`개 게임 블록 안에서 무작위 순열을 만들며,
에포크 번호와 시드로 순서를 결정합니다. 기본 `shuffle_buffer=4096`입니다.
기존 입력 파이프라인의 교체식 shuffle과 순서는 다르며 전체 데이터 무작위 셔플은 아닙니다.
마지막 부분 배치를 보존하고 MSE/MAE를 실제 타깃 개수로 가중 집계합니다.

현재 파일과 다음 파일의 이동·레이팅을 Arrow로 적재합니다. 이 두 파일에는 바이트 단위
메모리 상한이 없습니다. 보드 배열은 블록 단위로 복원하며 전체 게임 보드를 캐시하지 않습니다.
`--read-batch-size`는 Arrow에서 디코더에 넘길 행 수이고 파일 적재량 제한이 아닙니다.
파일 로딩만 백그라운드로 진행하며, 보드 복원과 장치 전송은 학습 루프에서 수행합니다.
`GameDataset`은 이미 배치된 텐서를 반환합니다. 직접 순회하거나
`DataLoader(dataset, batch_size=None, num_workers=0)`로 사용하세요.
추가 DataLoader 작업자는 데이터 중복과 파일 버퍼 복제를 방지하기 위해 거부합니다.
직접 iterator를 중간에 중단하면 `close()` 또는 `contextlib.closing`으로 정리하세요.

## 저장과 재시작

검증까지 완료한 에포크마다 `epoch-000001.pt` 등을 저장하고 `latest.json`을 갱신합니다.
임시 파일을 모두 쓴 뒤 교체하므로 저장 실패 시 이전 최신 체크포인트를 유지합니다.
가중치, Adam 상태, GradScaler 상태, PyTorch CPU/CUDA 난수 상태와 전체 지표 이력을 저장합니다.
스텝 중간 재시작은 지원하지 않습니다.

```bash
uv run train /data/lichess_monthly \
  --epochs 10 --batch-size 128 \
  --resume-from outputs/checkpoints \
  --checkpoint-dir outputs/checkpoints --output-dir outputs/run-resumed
```

`--epochs`는 추가 횟수가 아니라 최종 에포크 번호입니다. 특정 `.pt` 파일도 지정할 수 있습니다.
데이터 경로·크기·수정 시각, 전처리/모델/학습 코드, 배치·시드·정밀도·학습률·장치 종류·GPU 수·
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
    ratings = model(boards, valid_steps)  # torch.Tensor 입력, 출력 [B, 2]
```

## 이전 사항과 검증

TensorFlow/Keras 의존성, Keras 직렬화·콜백·legacy 호환 코드, XLA/JIT 옵션,
선택형 패딩 처리, 미사용 프로파일러, 중복 `src/main.py` 진입점을 제거했습니다.
`--jit-compile`, `--skip-padding`, `--generator-batch-size`, `--prefetch`는 제거했고,
정밀도 이름은 `float32`, `float16`, `bfloat16`으로 통일했습니다.
패딩 보드는 CNN 처리에서 항상 제외합니다. 기존 `.keras` 파일은 로드/변환하지 않으며
새 PyTorch 학습과 별도 출력 경로를 사용해야 합니다.

```bash
uv run python -m unittest discover -s tests -v
```

테스트는 특수 수 복원, 데이터 분할·파일 경계·부분 배치, 비동기 로더 정리와 오류 전달,
마스킹·빈 기보·역전파·CPU bfloat16, 지표 집계, 실제 모델의 연속 학습과 저장 후 재시작
결과 일치를 확인합니다. 로컬 검증은 macOS CPU 기준이며 Linux 서버의
CUDA·복수 GPU 실행은 대상 서버에서 확인해야 합니다.
