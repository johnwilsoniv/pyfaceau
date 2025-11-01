#!/usr/bin/env python3
"""
Multi-Video AU Accuracy Validation

Validates PyFaceAU against C++ OpenFace 2.2 across multiple videos
to determine if AU performance is consistent across diverse facial expressions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from validate_accuracy import generate_python_outputs, validate_aus

# Video list
VIDEOS = [
    ('IMG_0441', '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0441.MOV'),
    ('IMG_0443', '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0443.MOV'),
    ('IMG_0452', '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0452.MOV'),
    ('IMG_0453', '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0453.MOV'),
    ('IMG_0579', '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0579.MOV'),
]

CPP_REF_DIR = Path('cpp_reference_multi_video')

def validate_video(video_name, video_path, cpp_ref_csv):
    """
    Validate one video

    Returns:
        dict with video_name, au_correlations, mean_correlation, passing_aus
    """
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"{'='*80}")

    # Load C++ reference
    print(f"  Loading C++ reference: {cpp_ref_csv}")
    cpp_df = pd.read_csv(cpp_ref_csv)

    # Generate Python outputs using C++ landmarks
    print(f"  Generating Python outputs...")
    start_time = time.time()
    python_results = generate_python_outputs(video_path, cpp_df, max_frames=None)
    elapsed = time.time() - start_time
    print(f"  ✓ Processed {len(python_results)} frames in {elapsed:.1f}s ({len(python_results)/elapsed:.1f} fps)")

    # Validate AUs
    print(f"  Validating AUs...")
    au_results = validate_aus(cpp_df, python_results)

    # Extract AU correlations
    au_correlations = {}
    passing_aus = 0

    print(f"\n  AU Correlations:")
    for au_name, corr in sorted(au_results.items()):
        au_correlations[au_name] = corr

        status = "✓" if corr > 0.83 else "✗"
        print(f"    {status} {au_name}: {corr:.4f}")

        if corr > 0.83:
            passing_aus += 1

    mean_corr = np.mean(list(au_results.values()))
    print(f"\n  Mean AU correlation: {mean_corr:.4f}")
    print(f"  Passing AUs: {passing_aus}/17")

    return {
        'video_name': video_name,
        'au_correlations': au_correlations,
        'mean_correlation': mean_corr,
        'passing_aus': passing_aus,
        'frames': len(python_results),
        'fps': len(python_results)/elapsed
    }

def main():
    print("="*80)
    print("Multi-Video AU Accuracy Validation")
    print("="*80)
    print(f"\nValidating {len(VIDEOS)} videos against C++ OpenFace 2.2")
    print(f"Using C++ CLNF landmarks to isolate AU prediction accuracy\n")

    # Validate each video
    results = []
    for video_name, video_path in VIDEOS:
        cpp_ref_csv = CPP_REF_DIR / f"{video_name}.csv"

        if not cpp_ref_csv.exists():
            print(f"✗ Skipping {video_name}: C++ reference not found")
            continue

        try:
            result = validate_video(video_name, video_path, cpp_ref_csv)
            results.append(result)
        except Exception as e:
            print(f"✗ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    print("\n" + "="*80)
    print("Multi-Video Summary")
    print("="*80)

    # Per-video summary
    print(f"\nPer-Video Results:")
    print(f"{'Video':<12} {'Mean Corr':<12} {'Passing':<10} {'Frames':<8} {'FPS':<8}")
    print(f"{'-'*60}")

    for result in results:
        print(f"{result['video_name']:<12} "
              f"{result['mean_correlation']:.4f}       "
              f"{result['passing_aus']}/17      "
              f"{result['frames']:<8} "
              f"{result['fps']:.1f}")

    # Overall statistics
    overall_mean = np.mean([r['mean_correlation'] for r in results])
    overall_passing = np.mean([r['passing_aus'] for r in results])

    print(f"\n{'Overall':<12} {overall_mean:.4f}       {overall_passing:.1f}/17")

    # Per-AU statistics across all videos
    print(f"\n\nPer-AU Results Across All Videos:")
    print(f"{'AU':<8} ", end='')
    for result in results:
        print(f"{result['video_name']:<10} ", end='')
    print(f"{'Mean':<10} {'Passing':<10}")
    print(f"{'-'*80}")

    # Get all AU names
    au_names = sorted(results[0]['au_correlations'].keys())

    au_summary = {}
    for au_name in au_names:
        print(f"{au_name:<8} ", end='')

        au_corrs = []
        for result in results:
            corr = result['au_correlations'][au_name]
            au_corrs.append(corr)
            print(f"{corr:.4f}     ", end='')

        au_mean = np.mean(au_corrs)
        au_passing = sum(1 for c in au_corrs if c > 0.83)

        print(f"{au_mean:.4f}     {au_passing}/{len(results)}")

        au_summary[au_name] = {
            'mean': au_mean,
            'min': np.min(au_corrs),
            'max': np.max(au_corrs),
            'std': np.std(au_corrs),
            'passing_videos': au_passing
        }

    # Identify consistently failing AUs
    print(f"\n\nAU Performance Categories:")
    print(f"\nConsistently Excellent (mean > 0.95):")
    excellent_aus = [au for au, stats in au_summary.items() if stats['mean'] > 0.95]
    if excellent_aus:
        for au in excellent_aus:
            print(f"  ✓ {au}: {au_summary[au]['mean']:.4f} (passing {au_summary[au]['passing_videos']}/{len(results)} videos)")
    else:
        print("  None")

    print(f"\nConsistently Good (0.90 < mean <= 0.95):")
    good_aus = [au for au, stats in au_summary.items() if 0.90 < stats['mean'] <= 0.95]
    if good_aus:
        for au in good_aus:
            print(f"  ✓ {au}: {au_summary[au]['mean']:.4f} (passing {au_summary[au]['passing_videos']}/{len(results)} videos)")
    else:
        print("  None")

    print(f"\nAcceptable (0.83 < mean <= 0.90):")
    acceptable_aus = [au for au, stats in au_summary.items() if 0.83 < stats['mean'] <= 0.90]
    if acceptable_aus:
        for au in acceptable_aus:
            print(f"  ✓ {au}: {au_summary[au]['mean']:.4f} (passing {au_summary[au]['passing_videos']}/{len(results)} videos)")
    else:
        print("  None")

    print(f"\nConsistently Failing (mean <= 0.83):")
    failing_aus = [au for au, stats in au_summary.items() if stats['mean'] <= 0.83]
    if failing_aus:
        for au in failing_aus:
            print(f"  ✗ {au}: {au_summary[au]['mean']:.4f} (passing {au_summary[au]['passing_videos']}/{len(results)} videos)")
    else:
        print("  None - All AUs passing!")

    # Save detailed results to CSV
    print(f"\n\nSaving detailed results...")

    # Create per-video AU correlation matrix
    au_matrix = []
    for result in results:
        row = {'video': result['video_name']}
        row.update(result['au_correlations'])
        row['mean_correlation'] = result['mean_correlation']
        row['passing_aus'] = result['passing_aus']
        au_matrix.append(row)

    df_matrix = pd.DataFrame(au_matrix)
    df_matrix.to_csv('multi_video_au_correlations.csv', index=False)
    print(f"  ✓ Saved: multi_video_au_correlations.csv")

    # Create AU summary statistics
    au_summary_rows = []
    for au_name, stats in au_summary.items():
        au_summary_rows.append({
            'AU': au_name,
            'Mean': stats['mean'],
            'Min': stats['min'],
            'Max': stats['max'],
            'StdDev': stats['std'],
            'Passing_Videos': f"{stats['passing_videos']}/{len(results)}"
        })

    df_summary = pd.DataFrame(au_summary_rows)
    df_summary = df_summary.sort_values('Mean', ascending=False)
    df_summary.to_csv('multi_video_au_summary.csv', index=False)
    print(f"  ✓ Saved: multi_video_au_summary.csv")

    print(f"\n{'='*80}")
    print(f"Multi-Video Validation Complete!")
    print(f"{'='*80}")
    print(f"\nOverall Performance:")
    print(f"  Mean AU correlation: {overall_mean:.4f} ({overall_mean*100:.2f}%)")
    print(f"  Average passing AUs: {overall_passing:.1f}/17")
    print(f"  Excellent AUs (>0.95): {len(excellent_aus)}")
    print(f"  Good AUs (0.90-0.95): {len(good_aus)}")
    print(f"  Acceptable AUs (0.83-0.90): {len(acceptable_aus)}")
    print(f"  Failing AUs (<0.83): {len(failing_aus)}")

if __name__ == '__main__':
    main()
