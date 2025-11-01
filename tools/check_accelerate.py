#!/usr/bin/env python3
"""
Check NumPy BLAS Configuration for Apple Silicon

Verifies that NumPy is using Apple's Accelerate framework for
optimized matrix operations on macOS.

Usage:
    python check_accelerate.py
"""

import sys
import platform


def check_accelerate():
    """Check if NumPy is using Apple Accelerate BLAS"""
    print("=" * 80)
    print("NUMPY BLAS CONFIGURATION CHECK - APPLE SILICON")
    print("=" * 80)
    print("")

    # Check platform
    print("Platform Information:")
    print(f"  System: {platform.system()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")
    print("")

    # Check if on macOS ARM
    is_mac_arm = (platform.system() == 'Darwin' and platform.machine() == 'arm64')
    if is_mac_arm:
        print("✅ Running on Apple Silicon (ARM64)")
    else:
        print(f"⚠️  Not on Apple Silicon (detected: {platform.system()} {platform.machine()})")
    print("")

    # Import NumPy
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
        print("")
    except ImportError:
        print("❌ NumPy not installed!")
        print("   Install with: pip install numpy")
        return False

    # Get BLAS info
    print("NumPy Configuration:")
    print("-" * 80)
    try:
        config = np.__config__.show()
    except Exception as e:
        print(f"❌ Could not get NumPy configuration: {e}")
        return False

    print("")

    # Parse configuration to check for Accelerate
    config_str = str(config) if config is not None else ""

    using_accelerate = False

    try:
        # Try to get build info
        if hasattr(np.__config__, '_built_with_meson'):
            print("Build system: Meson")
        else:
            print("Build system: Legacy")

        # Check for Accelerate in library info
        if hasattr(np.__config__, 'blas_info'):
            blas_info = np.__config__.blas_info
            print(f"\nBLAS Info: {blas_info}")

        # Check show() output for Accelerate
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            np.__config__.show()
        output = f.getvalue()

        if 'Accelerate' in output or 'vecLib' in output:
            using_accelerate = True

    except Exception as e:
        print(f"Note: Could not parse detailed config: {e}")

    print("")
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    if using_accelerate:
        print("✅ NumPy IS using Apple Accelerate framework!")
        print("")
        print("Benefits:")
        print("  - Optimized matrix operations for Apple Silicon")
        print("  - 1.5-2x faster CalcParams performance")
        print("  - Automatic hardware acceleration")
        print("")
        print("No action needed - you're already optimized!")
        return True
    else:
        print("⚠️  NumPy may NOT be using Accelerate framework")
        print("")
        print("To enable Accelerate (recommended):")
        print("")
        print("  1. Uninstall current NumPy:")
        print("     pip uninstall numpy")
        print("")
        print("  2. Reinstall NumPy (will use Accelerate by default on macOS):")
        print("     pip install numpy")
        print("")
        print("  3. Run this script again to verify:")
        print("     python check_accelerate.py")
        print("")
        print("Expected speedup: 1.5-2x on CalcParams (80ms → 40-53ms)")
        return False


def run_performance_test():
    """Quick performance test of matrix operations"""
    import numpy as np
    import time

    print("")
    print("=" * 80)
    print("PERFORMANCE TEST")
    print("=" * 80)
    print("")
    print("Running matrix multiplication benchmark...")
    print("")

    sizes = [100, 500, 1000, 2000]
    for size in sizes:
        # Generate random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Warmup
        for _ in range(3):
            C = A @ B

        # Benchmark
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            C = A @ B
        elapsed = time.perf_counter() - start

        time_per_op = (elapsed / iterations) * 1000
        gflops = (2 * size**3 * iterations / elapsed) / 1e9

        print(f"  {size}x{size}: {time_per_op:.2f}ms per matmul ({gflops:.2f} GFLOPS)")

    print("")
    print("If using Accelerate, you should see:")
    print("  - Fast performance (>50 GFLOPS for large matrices)")
    print("  - Performance scales well with matrix size")
    print("")


if __name__ == '__main__':
    using_accelerate = check_accelerate()

    # Run performance test
    run_performance_test()

    print("=" * 80)
    sys.exit(0 if using_accelerate else 1)
