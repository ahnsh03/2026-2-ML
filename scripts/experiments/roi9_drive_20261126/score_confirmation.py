"""Score a fresh GT-screened confirmation without refitting either calibration.

The previous supported-road followup has been retired into calibration v2.
This cohort is independent of that fit, conditional on geometry/GT support.
"""
from pathlib import Path
import argparse
from score_supported_roads import main as score_supported


def main():
    # Reuse the same scoring implementation through explicit frozen inputs.
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'extension', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    import sys
    sys.argv = [sys.argv[0], '--study', str(a.study), '--extension', str(a.extension),
                '--output', str(a.output), '--calibration', 'calibration_v2_final',
                '--policy', 'calibration_v2_final/policy_frozen.json', '--fresh-confirmation']
    score_supported()


if __name__ == '__main__':
    main()
