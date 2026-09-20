#!/usr/bin/env python3
"""Measure projected Arrow columns without importing TensorFlow.

Sample: python measure_parquet_memory.py games.parquet
Full:   python measure_parquet_memory.py a.parquet b.parquet --mode full --keep-files 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

GIB = 1024 ** 3


def rss_bytes():
    """Current resident memory, not the historical high-water mark."""
    if sys.platform.startswith('linux'):
        with open('/proc/self/status') as stream:
            for line in stream:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    elif sys.platform == 'darwin':
        try:
            return int(subprocess.check_output(
                ['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip()) * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    return None


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'rss_bytes': rss_bytes(),
        'process_peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
        'arrow_pool_bytes': pa.total_allocated_bytes(),
    }


def gib(value):
    return 'unavailable' if value is None else f'{value / GIB:.3f} GiB'


def selected_groups(count, sample_count):
    if count <= sample_count:
        return list(range(count))
    if sample_count == 1:
        return [count // 2]
    return [i * (count - 1) // (sample_count - 1) for i in range(sample_count)]


def measure_file(path, args):
    before = memory()
    started = time.perf_counter()
    with pq.ParquetFile(path) as source:
        movement_paths = [source.schema.column(i).path for i in range(len(source.schema))
                          if source.schema.column(i).path.startswith('ply_list.')
                          and source.schema.column(i).path.endswith('.movement')]
        if len(movement_paths) != 1:
            raise ValueError(f'Expected one ply_list movement field: {movement_paths}')
        columns = ['white_elo', 'black_elo',
                   'ply_list' if args.columns == 'with-time' else movement_paths[0]]
        total_rows = source.metadata.num_rows
        groups = (list(range(source.num_row_groups)) if args.mode == 'full'
                  else selected_groups(source.num_row_groups, args.sample_row_groups))
        if not groups or total_rows == 0:
            raise ValueError(f'Empty Parquet file: {path}')
        # Keep Arrow chunks: concatenation below does not copy their data buffers.
        parts = []
        last_log = time.perf_counter()
        for i, group in enumerate(groups):
            parts.append(source.read_row_group(group, columns=columns))
            now = time.perf_counter()
            if now - last_log >= 5 or i == len(groups) - 1:
                print(f'  row groups {i + 1}/{len(groups)}, RSS {gib(rss_bytes())}', flush=True)
                last_log = now
        table = pa.concat_tables(parts)
        del parts
        elapsed = time.perf_counter() - started
        selected_compressed = sum(
            source.metadata.row_group(g).column(c).total_compressed_size
            for g in groups for c in range(source.metadata.row_group(g).num_columns)
            if source.metadata.row_group(g).column(c).path_in_schema.split('.')[0]
               in ('white_elo', 'black_elo', 'ply_list')
            and (args.columns == 'with-time' or
                 source.metadata.row_group(g).column(c).path_in_schema.split('.')[-1]
                 in ('white_elo', 'black_elo', 'movement'))
        )
        total_groups = source.num_row_groups
    # Ensure nested projection really omitted time, instead of silently measuring it.
    plies = table.column('ply_list')
    names = [field.name for field in plies.type.value_type]
    if args.columns == 'moves' and names != ['movement']:
        raise ValueError(f'Unexpected projected ply fields: {names}')
    ply_count = sum(pc.sum(pc.list_value_length(chunk)).as_py() or 0 for chunk in plies.chunks)
    after = memory()
    estimated = table.nbytes / table.num_rows * total_rows
    result = {
        'path': str(path.resolve()), 'mode': args.mode, 'columns': args.columns,
        'file_bytes': path.stat().st_size, 'total_rows': total_rows,
        'total_row_groups': total_groups, 'loaded_row_groups': groups,
        'loaded_rows': table.num_rows, 'loaded_ply_count': ply_count,
        'mean_plies': ply_count / table.num_rows,
        'arrow_bytes': table.nbytes,
        'arrow_buffer_bytes': table.get_total_buffer_size(),
        'arrow_bytes_per_game': table.nbytes / table.num_rows,
        'estimated_full_arrow_bytes': estimated,
        'selected_compressed_column_bytes': selected_compressed,
        'read_seconds': elapsed, 'before': before, 'after': after,
        'rss_delta_bytes': (None if before['rss_bytes'] is None or after['rss_bytes'] is None
                            else after['rss_bytes'] - before['rss_bytes']),
        'schema': str(table.schema),
    }
    print(f'  loaded {table.num_rows:,}/{total_rows:,} games; mean ply={result["mean_plies"]:.2f}', flush=True)
    print(f'  Arrow={gib(table.nbytes)}, buffers={gib(result["arrow_buffer_bytes"])}', flush=True)
    label = 'full Arrow' if args.mode == 'full' else 'estimated full Arrow'
    print(f'  {label}={gib(estimated)}, read={elapsed:.2f}s', flush=True)
    print(f'  RSS={gib(after["rss_bytes"])}, RSS delta={gib(result["rss_delta_bytes"])}, '
          f'process peak={gib(after["process_peak_rss_bytes"])}', flush=True)
    return table, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--mode', choices=['sample', 'full'], default='sample')
    parser.add_argument('--sample-row-groups', type=int, default=8,
                        help='Evenly spaced groups including first/last; default 8')
    parser.add_argument('--columns', choices=['moves', 'with-time'], default='moves')
    parser.add_argument('--keep-files', type=int, choices=[1, 2], default=1,
                        help='Maximum simultaneously retained files, default 1')
    parser.add_argument('--output', type=Path, default=Path('parquet-memory-report.json'))
    args = parser.parse_args()
    if args.sample_row_groups < 1:
        parser.error('--sample-row-groups must be positive')
    for path in args.files:
        if not path.is_file():
            parser.error(f'File not found: {path}')
    report = {
        'python': sys.version, 'pyarrow': pa.__version__, 'platform': sys.platform,
        'pid': os.getpid(), 'settings': {
            'mode': args.mode, 'columns': args.columns,
            'sample_row_groups': args.sample_row_groups, 'keep_files': args.keep_files,
        },
        'baseline': memory(), 'files': [], 'releases': [],
        'notes': [
            'GB=10^9 bytes; displayed GiB=2^30 bytes.',
            'Sample estimates extrapolate selected row groups, not a statistical confidence interval.',
            'RSS includes Python, Arrow buffers, metadata, allocator retention and read workspace.',
            'Peak RSS is cumulative for this process, not a separate peak for each file.',
            'No board decoding, TensorFlow, model or training buffers are included.',
            'Keep-files=2 in sample mode retains two samples, not two full files.',
            'Reads are sequential; this measures residency, not overlap with training.',
        ],
    }
    retained = []

    def release_oldest():
        old_path, old_table = retained.pop(0)
        before = memory()
        del old_table
        gc.collect()
        after_gc = memory()
        # Best effort: allocator/OS may keep some pages resident.
        pa.default_memory_pool().release_unused()
        after_release = memory()
        report['releases'].append({'path': old_path, 'before': before,
                                   'after_gc': after_gc, 'after_release_unused': after_release})
        print(f'Released {old_path}: RSS {gib(before["rss_bytes"])} -> '
              f'{gib(after_release["rss_bytes"])}', flush=True)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    save()
    try:
        for path in args.files:
            if len(retained) >= args.keep_files:
                release_oldest()
            print(f'Loading {path} [{args.mode}, {args.columns}]', flush=True)
            table, result = measure_file(path, args)
            retained.append((str(path), table))
            del table
            result['retained_files'] = [p for p, _ in retained]
            result['retained_arrow_bytes'] = sum(t.nbytes for _, t in retained)
            report['files'].append(result)
            save()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        while retained:
            release_oldest()
        report['final'] = memory()
        save()
    print(f'Report saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
