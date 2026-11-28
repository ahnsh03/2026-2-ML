"""Inspect within-pose RGB settling; SDK hour is not a visibility truth label."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.bridge import sha256_file
from quality_estimation.spatial_features import SpatialProfile


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['archive', 'events', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    frames, events = records(a.archive / 'rgb.jsonl'), records(a.events)
    roi = SpatialProfile.load().native
    masks = {'near': np.isin(roi, [6, 7]), 'hood': roi == 8,
             'upper_top': np.isin(roi, [0, 1])}
    table = []
    for row in frames:
        time = datetime.fromisoformat(row['received_utc'])
        eligible = [i for i, e in enumerate(events) if datetime.fromisoformat(e['requested_utc']) <= time]
        if not eligible:
            continue
        index = eligible[-1]
        event = events[index]
        elapsed = (time - datetime.fromisoformat(event['requested_utc'])).total_seconds()
        if elapsed > 18.5:
            continue
        image = a.archive / row['image']
        assert sha256_file(image) == row['expected_sha256']['rgb']
        gray = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2GRAY) / 255.
        table.append(dict(event=index, sdk_hour=event['hour'], seconds_after_request=elapsed,
                          sample_id=row['sample_id'],
                          **{name: float(np.median(gray[mask])) for name, mask in masks.items()}))
    summaries = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
    fig.subplots_adjust(bottom=.24, wspace=.3)
    for index, event in enumerate(events):
        data = [r for r in table if r['event'] == index]
        if not data:
            raise ValueError('time event received no RGB frames')
        summary = dict(event=index, sdk_hour=event['hour'], frames=len(data))
        for name, ax in zip(masks, axes):
            x = [r['seconds_after_request'] for r in data]
            y = [r[name] for r in data]
            ax.plot(x, y, marker='.', label=f'{event["hour"]:02d}h ({index + 1})')
            ax.set_title(name.replace('_', ' ').title())
            ax.set_xlabel('Seconds after SDK request (receiver clock)')
            ax.set_ylabel('Median luma [0,1]')
            ax.grid(alpha=.15)
            early = [r[name] for r in data if 1.5 <= r['seconds_after_request'] <= 4]
            late = [r[name] for r in data if 12 <= r['seconds_after_request'] <= 17]
            summary[name] = dict(early_median=float(np.median(early)) if early else None,
                                 late_median=float(np.median(late)) if late else None)
        summaries.append(summary)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('One-pose photometric check: hour and physical visibility are different')
    fig.text(.015, .015, 'Sunny, same parked pose, ROS RGB only. Receiver UTC is not calibrated exposure UTC. No GT, accuracy/training admission or visibility exemption.', fontsize=8.5)
    artifacts = []
    for extension in ['png', 'svg']:
        path = a.output / ('06_photometric_settling.' + extension)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        artifacts.append(path)
    plt.close(fig)
    report = dict(events=summaries, rows=table,
                  input_sha256={p.name: sha256_file(p) for p in [a.events, a.archive / 'rgb.jsonl']},
                  source_sha256=sha256_file(Path(__file__)),
                  artifacts={p.name: sha256_file(p) for p in artifacts},
                  scope='One parked pose, no GT or physical visibility labels. Temporal diagnostic; no policy tuning or confirmation accuracy claim.')
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
