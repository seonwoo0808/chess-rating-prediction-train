# Parquet 메모리 측정

`measure_parquet_memory.py`는 TensorFlow 없이 Python과 PyArrow만 사용합니다.
서버에 스크립트 하나만 복사해도 됩니다. PyArrow가 없으면 해당 환경에서
`python -m pip install pyarrow`로 설치합니다.

```bash
# 먼저 파일 앞·중간·뒤에 분산된 8개 row group으로 전체 Arrow 크기 추정
python measure_parquet_memory.py /data/month01.parquet --output sample.json

# 기존 전처리처럼 잔여 시간까지 포함하여 비교
python measure_parquet_memory.py /data/month01.parquet --columns with-time --output sample-with-time.json

# 표본 추정 후, 두 파일 전체를 동시에 적재
python measure_parquet_memory.py /data/month01.parquet /data/month02.parquet \
  --mode full --keep-files 2 --output full-two-files.json
```

세 번째 파일도 전달하면 A+B 적재 → A 해제 → B+C 적재 순서가 됩니다.
기본은 이동과 레이팅만 읽습니다. 초기 테스트는 표본 모드를 권장합니다.
전체 모드에는 자동 메모리 상한이 없으며, 실제로 파일 전체를 적재합니다.
표본 모드의 keep-files는 전체 파일 대신 표본들만 유지합니다.

- `arrow_bytes`: 읽은 Arrow 데이터 크기
- `arrow_buffer_bytes`: 참조된 실제 Arrow 버퍼 크기
- `estimated_full_arrow_bytes`: 경기당 크기로 추정한 전체 파일 크기
- `after.rss_bytes`: 적재 후 실제 프로세스 상주 메모리
- `after.process_peak_rss_bytes`: 프로세스 시작 이후 최고 상주 메모리
- `releases`: 참조 해제·GC·Arrow 미사용 메모리 반환 전후의 메모리
- 모든 JSON 크기는 바이트이고 터미널은 GiB(2^30 바이트)입니다.

RSS에는 압축 해제 작업 공간과 메모리 할당기 등도 포함됩니다.
Arrow 크기 추정은 표본의 평균 수순 길이가 전체와 다르면 오차가 납니다.
GC 후에도 할당기가 메모리를 보유할 수 있으므로 release_unused도 별도 측정합니다.
파일은 순차 로딩하지만 keep-files=2이면 두 파일을 동시에 보관합니다.
학습과 비동기 읽기 중첩, 보드 복원, 모델/GPU 메모리는 이 측정에 포함하지 않습니다.
OS 파일 캐시 상태에 따라 읽기 속도는 달라집니다.
