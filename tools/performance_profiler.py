#!/usr/bin/env python3
"""
Performance Profiler Stub

This is a minimal stub to allow pyfaceau to run without the full S1 performance profiler.
For benchmarking, use benchmark_detailed.py which has its own timing mechanisms.
"""

import contextlib


class NullProfiler:
    """No-op profiler that does nothing"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start(self, *args, **kwargs):
        """No-op start"""
        pass

    def stop(self, *args, **kwargs):
        """No-op stop"""
        pass

    @contextlib.contextmanager
    def time_block(self, *args, **kwargs):
        """No-op context manager for timing blocks"""
        yield


def get_profiler():
    """Return a null profiler that does nothing"""
    return NullProfiler()


def set_pipeline_context(*args, **kwargs):
    """No-op context setter"""
    pass
