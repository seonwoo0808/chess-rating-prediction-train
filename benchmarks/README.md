# 독립 학습 벤치마크

> 이 폴더는 이전 **DataParallel 비교용** 벤치마크입니다. 현재 본 학습의 DDP
> 실행 경로와 다르므로 결과를 DDP 성능으로 해석하지 마세요. `torchrun`으로
> 실행하지 말고 아래의 일반 Python 실행 방식으로 사용하세요.

`fixed_batch.py`는 이 저장소의 `src/train`을 import하는 독립 실행 도구입니다.
모델, 데이터 로더, 학습 루프를 복제하거나 수정하지 않습니다.
필요 없으면 `benchmarks/` 폴더만 삭제하면 됩니다. 추가 의존성은 없습니다.

## 실행

서버의 `train` 프로젝트 폴더에서 실행합니다. 이 `benchmarks/` 폴더를
`src/`, `pyproject.toml`과 같은 위치에 복사하고 실제 데이터 경로를 지정하세요.
다른 학습이 같은 GPU를 사용하지 않는 상태에서 측정해야 비교할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python -B benchmarks/fixed_batch.py \
  /data/lichess_monthly \
  --batch-size 1024 --precision bfloat16 \
  --warmup 20 --steps 100 --repeats 3
```

GPU 번호 `0,1`은 실제 할당 번호로 바꿉니다. 데이터는 디렉터리 또는
하나 이상의 Parquet 파일을 지정할 수 있습니다. CUDA가 없으면 기본 실행은
오류로 종료하며 CPU로 자동 대체하지 않습니다.

## 측정 범위

- 기본 앞 8192게임을 대상으로 기존 데이터 로더의 95:5 분할과 4096게임 셔플을
  적용한 뒤 실제 학습 배치 하나를 선택합니다. 로더를 닫고 배치를 첫 GPU에
  올린 후 측정합니다. 전체 배치가 부족하면 오류로 종료합니다.
- 새 모델로 20회 워밍업한 다음 같은 배치를 100회씩 3라운드 반복 학습합니다.
  워밍업과 각 라운드 사이에 모델/optimizer를 초기화하지 않습니다.
- 기존 `run_epoch`를 그대로 호출하므로 forward, backward, Adam, 유한 loss 검사,
  지표 집계가 포함됩니다. GPU 2개에서는 기존과 동일한 `DataParallel`의
  모델 복제, GPU 간 입력 분배, 출력 수집과 gradient 합산도 포함됩니다.
- 파일 읽기, 보드 복원, 셔플, 최초 CPU→GPU 전송은 측정에서 제외됩니다.
  순수 CUDA 커널 시간만 측정하는 도구는 아닙니다.
- 모든 GPU를 라운드 시작/끝에서 동기화합니다. 타이머 때문에 매 step마다
  동기화를 추가하지 않으며, 기존 학습 루프 안의 동기화는 그대로 유지합니다.
- 기본적으로 진행 표시와 20-step CPU 지표 읽기는 끕니다. 실제 `--verbose 1`
  실행과 이 부분까지 맞추려면 `--progress`를 추가하세요.
- 데이터는 읽기만 합니다. 기존 체크포인트를 읽거나 덮어쓰지 않으며 모델과
  optimizer 상태, 결과 파일을 저장하지 않습니다. 결과는 터미널에 출력됩니다.

## 결과 해석

마지막 JSON의 `fixed_batch_median_ms_per_step`이 세 라운드 평균 step 시간들의
중앙값입니다. `round_ms_per_step`으로 안정화 여부도 확인하세요.

- 실제 학습 시간과 비슷하면 모델 실행·동기화·멀티 GPU 비용부터 확인합니다.
- 실제 학습보다 확실히 빠르면 입력 준비·전송 또는 실제 배치별 크기 변동의
  영향이 의심됩니다. 시간 차이를 순수 데이터 로딩 시간으로 단정하지 마세요.
- 이 테스트는 유효 보드 수가 고정됩니다. JSON의 `valid_fraction`과
  `valid_boards_per_replica`를 함께 보내주세요. 첫 배치 하나가 파일 전체의
  게임 길이를 대표하지 않으며 가변 CNN 크기로 인한 비용도 재현하지 않습니다.
- 기존 학습과 GPU, 배치, 정밀도를 맞추고 파일 로딩·초기 워밍업·검증·저장을
  제외한 안정된 학습 구간과 비교하세요. 수렴이나 예측 정확도 시험은 아닙니다.

로컬 CPU에서는 작은 실제 Parquet으로 실행 흐름을 확인했습니다.
CUDA 및 복수 GPU의 처리 시간은 서버에서 검증해야 합니다.

## 입력·전송·가변 배치 비교: pipeline.py

입력 준비·전송과 모델 실행을 나누어 조사하는 도구입니다.
`pipeline.py`도 기존 모델·데이터 로더·`run_epoch`를 그대로 import합니다.
원본 코드 변경이나 monkey patch 없이 iterator 바깥에서 시간을 기록합니다.
기존 `fixed_batch.py`와 독립적으로 실행할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python -B benchmarks/pipeline.py \
  /data/lichess_monthly \
  --batch-size 1024 --precision bfloat16 \
  --warmup 20 --steps 100 --repeats 3 --bank-batches 16
```

다른 학습이 같은 GPU에서 실행되지 않는 상태에서 실행합니다. 아래 순서대로
측정하며 각 학습 시나리오는 같은 시드의 새 모델·Adam으로 시작합니다.
워밍업부터 라운드 끝까지 가중치는 계속 갱신하며 저장하지 않습니다.

| 시나리오 | 반복 입력 | 포함하는 작업 |
|---|---|---|
| `fixed_gpu` | GPU에 올린 같은 배치 | 기존 학습 루프·DataParallel |
| `fixed_cpu` | 일반 CPU RAM의 같은 배치 | 위 작업 + 기존 CPU→GPU 전송 |
| `varied_gpu` | GPU에 올린 실제 16배치를 순서대로 순환 | 서로 다른 배치를 현재 원본 모델로 처리 |
| `varied_cpu` | 같은 실제 16배치를 CPU RAM에서 순환 | 서로 다른 배치 + 기존 CPU→GPU 전송 |
| `streaming` | 기존 데이터 로더에서 매번 다음 배치 | 실제 입력·전송·학습 전체 |
| `input_only` | 같은 데이터 로더에서 다음 배치 | 학습 없이 입력 준비만 |

기본 설정에서 16배치 보드 저장량은 약 128MiB이며 마스크·타깃이 조금 더해집니다.
GPU 상주 배치는 첫 GPU에 저장합니다. GPU 간 분배·모델 복제 비용은 여전히
포함됩니다. CPU 상주 배치는 원본처럼 pinned memory나 비동기 전송을 사용하지 않습니다.

### 기록되는 결과

터미널에 시나리오별 ms/step과 입력 대기 요약을 표시합니다. 전체 결과는
`benchmarks/results/pipeline-날짜-시간.json`에 저장합니다. 기존 파일을 덮어쓰지 않습니다.
이 JSON 전체를 전달하면 배치별 기록까지 분석할 수 있습니다.

- `scenarios.*.median_ms_per_step`: 라운드별 평균 step 시간들의 중앙값.
- `comparisons_not_additive`: CPU/GPU, 고정/가변, streaming/varied_cpu 비교 차이.
  원인별 독립 시간으로 더하거나 순수 전송·복원 비용으로 단정하지 않습니다.
- `first_batch_prepare_ms`: 첫 파일 적재를 포함한 최초 배치 준비 시간.
  워밍업에 속하며 측정 라운드 평균에서는 제외합니다.
- `next_wait`: 워밍업 이후 `next(dataset)` 호출 시간의 평균·중앙값·p95·최댓값.
  배치 prefetch가 켜져 있으면 생산자의 복원 시간 자체가 아니라 준비된 배치를
  받기까지 소비자가 기다린 시간입니다.
- `next_wait_by_block_phase`: 복원 블록 안에서 배치 위치별 준비 시간.
  기본 4096게임 블록 / 배치 1024는 주기 4이고 `0`이 새 블록 준비 위치입니다.
  prefetch 사용 시 실제 소비자 대기 위치는 대기열의 준비 상태에 따라 달라집니다.
  블록 크기가 배치 크기로 나누어떨어지지 않으면 이 그룹 통계를 생략합니다.
- `input_wait_trace`: 초기 워밍업부터 모든 배치의 번호·라운드·대기 시간·유효 보드 수.
  배치 번호는 0부터 시작하며 워밍업과 라운드 사이에 입력을 다시 시작하지 않습니다.
- `bank_metadata`: 미리 준비한 각 배치의 유효 비율과 GPU별 유효 보드 수.

### 측정 해석과 한계

`fixed_cpu`만 크게 느려지면 전송·동기화를, `varied_gpu`부터 느려지면 유효 보드 수와
입력 크기 변화를, `streaming`에서만 크게 느려지면 실제 배치 준비를 우선 확인합니다.
`next_wait_by_block_phase`의 `0`에 지연이 집중되면 블록 준비에 의한 주기적 대기를
직접 확인할 수 있습니다. 각 배치의 디스크·복원·셔플 내부 시간까지 별도로 분해하지는 않습니다.

매 step에 GPU 동기화를 추가하지 않습니다. 각 측정 라운드 경계에서만 모든 GPU를
동기화하며 기존 학습 루프의 동기화는 유지합니다. 따라서 `next_wait_ms`는 CPU에서
보이는 준비 시간이며 이전 배치의 GPU 작업과 겹칠 수 있습니다. 전체 step 시간에
이 값을 더하면 안 됩니다. 배치 메타데이터를 읽는 작은 CPU 계측 비용은 전체
streaming/input_only 시간에 포함되며 `next_wait_ms`에는 포함되지 않습니다.

현재 원본 CNN은 고정 크기이므로 `varied`는 입력 내용이 다르다는 뜻이며 CNN 크기는
고정입니다. 고정/가변 CNN 비교에는 아래 `cnn_shapes.py`를 사용합니다.
기본 워밍업 20회는 16배치를 모두 거칩니다. `--bank-batches`를
늘리면 전부 워밍업할 수 있도록 `--warmup`도 함께 늘리는 편이 좋습니다.
고정 배치와 가변 배치는 평균 게임 길이도 다를 수 있으므로 메타데이터를 함께 비교합니다.

기본 샘플 범위는 요청한 모든 step을 실행할 수 있는 크기로 자동 제한합니다.
기본 설정은 약 35만 게임입니다. 원본 로더가 파일 전체를 미리 읽는 구조이므로 이
제한은 파일 적재량·메모리 압력도 줄입니다. 테스트 streaming이 실제 4it/s를 재현하지
못하면 동일한 데이터 경로와 `--sample-games 0`으로 전체 파일 범위를 다시 측정하세요.
전체 파일 적재는 실제 학습과 같은 수준의 RAM을 사용할 수 있고 시작이 오래 걸릴 수 있습니다.
짧은 샘플이 파일 경계를 넘지 않으면 파일 전환 지연을 확인할 수 없습니다.

진행 표시까지 기존 `--verbose 1`과 맞추려면 `--progress`를 추가합니다.
시나리오는 순차 실행하므로 OS 파일 캐시·GPU 온도 등의 영향이 남습니다.
필요하면 `--modes`로 시나리오와 순서를 명시할 수 있습니다.

```bash
# 실제 파일 적재량으로 입력과 학습만 확인
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python -B benchmarks/pipeline.py \
  /data/lichess_monthly --sample-games 0 --modes streaming input_only --progress
```

`--device cpu`는 작은 데이터로 실행 흐름을 확인하기 위한 옵션입니다. 이 경우
`*_gpu`라는 이름도 CPU 장치 상주를 의미하므로 GPU 성능 결과로 해석하지 않습니다.
추가 의존성·체크포인트 변경은 없으며 결과까지 포함해 `benchmarks/`만 삭제하면 제거됩니다.

### 배치 prefetch 전후 비교

기본 `--prefetch-batches 2`로 현재 학습과 같은 배치 준비를 사용합니다. 다음 실행과
같은 명령에서 `--prefetch-batches 0`으로 바꾼 실행을 비교하세요. 결과 JSON에 설정이
기록됩니다. 두 실행은 동시에 하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python -B benchmarks/pipeline.py \
  /data/lichess_monthly --batch-size 1024 --precision bfloat16 \
  --modes fixed_gpu fixed_cpu streaming input_only --progress --prefetch-batches 2
```

주요 비교 대상은 `streaming` 시간과 `next_wait`입니다. 입력 전용 순회는 학습이 없어
대기열을 빠르게 소비하므로, `input_only` 속도가 그대로여도 학습 중 대기가 줄면 효과가
있습니다. GPU 전송 방식은 바꾸지 않았으므로 `fixed_cpu`의 전송 비용은 남아 있습니다.

## CNN 입력 크기 고정/가변 비교: cnn_shapes.py

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python -B benchmarks/cnn_shapes.py \
  /data/lichess_monthly \
  --batch-size 1024 --precision bfloat16 \
  --warmup 20 --steps 100 --passes 3
```

원본 `BoardEncoder`가 고정 크기로 변경된 이후에도 비교가 유지되도록 `dynamic`은
벤치마크 전용 `DynamicShapeEncoder`에 이전의 패딩 제외 경로를 보존합니다.
`fixed`의 `FixedShapeEncoder`는 원본 `BoardEncoder.forward`를 상속합니다.
두 방식 모두 **원본 CNN 층과 가중치 구조를 그대로 사용**하며, 벤치마크 실행이
원본 파일이나 전역 클래스를 수정하지 않습니다.

| 항목 | dynamic | fixed |
|---|---|---|
| GPU당 입력 게임 수 | 512 | 512 |
| CNN에 넣는 보드 | 유효한 보드만 | 패딩 포함 모든 보드 |
| CNN 입력 shape | `[valid.sum()+1, 12, 8, 8]` | `[65537, 12, 8, 8]` |
| 패딩 위치의 CNN 특징 | scatter 결과의 0 | CNN 처리 후 0으로 마스킹 |
| Transformer 입력 | `[512, 128, 128]` | `[512, 128, 128]` |

65537은 `512 × 128 + dummy 1개`입니다. 공정한 비교를 위해 원본처럼 dummy를
붙이고, one-hot의 int64→float32 변환과 메모리 배치도 유지합니다. 변경하는 것은
유효 보드 선택/scatter 대신 전체 보드를 처리한 후 출력 마스킹을 하는 부분입니다.
고정 방식은 패딩 위치의 입력을 0으로 바꾼 후 채널을 확장하므로 범위를 벗어난
패딩 값도 기존처럼 무시합니다. 패딩을 실제 보드처럼 attention에 넣는 실험이 아닙니다.

두 방식은 별도 Python 프로세스에서 순서대로 실행하여 PyTorch/cuDNN의 프로세스 내
캐시 영향을 분리합니다. 데이터·시드·초기 모델·optimizer 설정은 동일하며 실제
입력 배열 전체의 SHA-256이 같아야 비교 결과를 출력합니다. 원본 소스 해시도 대조합니다.
측정 순서는 dynamic→fixed이며 OS 캐시·온도·다른 GPU 작업까지 격리되는 것은 아닙니다.

### 최초 순회와 재사용 순회

1. 실제 배치 `20 + 100 = 120개`를 미리 준비하고 로더를 닫습니다.
2. 앞 20개로 워밍업합니다.
3. **워밍업과 다른 다음 100개**를 처음 순회하며 측정합니다.
4. 같은 100개를 같은 순서로 두 번 더 순회합니다. 가중치·Adam은 초기화하지 않습니다.

이전 `pipeline.py`의 16개 배치 순환과 달리, 첫 측정부터 많은 배치별 크기를 만납니다.
서로 다른 배치의 유효 보드 수가 우연히 같을 수 있으므로, GPU별 서로 다른 크기 수와
워밍업에서 없었던 크기 수를 결과에 기록합니다. 최초 순회가 모든 크기에 대해
완전히 차가운 캐시라는 뜻은 아닙니다.

기본 `--residency gpu`는 배치를 첫 GPU에 미리 올립니다. 따라서 데이터 로딩·복원·셔플·
최초 CPU→GPU 전송을 제외하고, 기존 학습 루프 전체와 DataParallel 비용을 측정합니다.
원본 전송까지 포함해 비교하려면 `--residency cpu`를 추가하세요. 이 경우에도 데이터는
CPU RAM에 미리 준비하므로 디코딩 시간은 제외합니다.

기본 배치 은행은 약 **975MiB**입니다. 고정 CNN은 패딩까지 처리하므로 모델 연산용
메모리는 가변 CNN보다 증가할 수 있습니다. `peak_allocated_mib_per_gpu`로 실제 피크를
기록합니다. 메모리가 부족하면 양쪽 모두 같은 더 작은 `--batch-size`로 다시 비교하세요.
`--steps`를 줄이면 저장할 입력 메모리는 줄지만 한 step의 CNN 활성값 메모리는 줄지 않습니다.

### 결과

`benchmarks/results/cnn-shapes-날짜-시간.json`에 전체 결과를 저장합니다.
터미널에는 각 pass의 dynamic/fixed 시간과 첫 순회·재사용 순회 요약을 표시합니다.

- `comparisons`: 같은 pass에서 양쪽 시간과 fixed 처리량 향상 비율.
  `fixed_speedup_ratio > 1`이면 fixed가 빠릅니다.
- `dynamic_first_pass_ms`, `fixed_first_pass_ms`: 처음 만나는 측정 배치들의 평균 시간.
- `dynamic_replay_median_ms`, `fixed_replay_median_ms`: 2번째 이후 pass 평균 시간들의 중앙값.
- 전체 JSON의 `batch_shapes`: 워밍업·측정 배치별 실제 CNN 보드 수.
- `identical_input_verified`: 양쪽에서 읽은 입력·마스크·타깃이 동일한지 확인한 결과.

fixed가 처음부터 빠르고 dynamic은 재사용 후 빨라지면 새로운 입력 크기 처리 비용이
유력해집니다. 단, 이 비교는 gather/scatter 제거와 패딩 계산량 증가도 함께 포함하므로
속도 차이만으로 cuDNN 실행 계획 생성 하나를 원인으로 확정하지 않습니다.

JSON 외에 모델이나 입력 데이터를 저장하지 않습니다. 추가 의존성은 없고 이 파일과
결과도 `benchmarks/` 폴더를 삭제하면 제거됩니다. 두 방식의 출력·기울기·패딩 처리와
작은 Parquet의 전체 비교 실행은 로컬 CPU에서 검증했으며 CUDA 검증은 서버에서 수행합니다.
