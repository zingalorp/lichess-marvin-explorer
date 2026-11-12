"""Build compact population means from clock-annotated Marvin Parquet rows.

Run with the existing Marvin data environment; pyarrow is development-only.
No outcomes or WDL predictions are used to select these inputs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('dataset', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
controls = {'15+0','60+0','120+0','180+0','180+2','300+0','300+3','600+0','600+5','900+10','1800+0'}
stats = defaultdict(lambda: [0, 0.0, 0, 0.0])
rows = 0
for file in sorted(args.dataset.glob('*.parquet')):
    pf = pq.ParquetFile(file)
    names = pf.schema_arrow.names
    base = 'tc_base_s' if 'tc_base_s' in names else 'base_time'
    cols = ['move_ply', 'active_elo', 'opp_elo', base, 'active_inc',
            'active_clock', 'opp_clock', 'time_target', 'policy_is_resign', 'policy_is_flag']
    for batch in pf.iter_batches(columns=cols):
        arrays = {name: np.asarray(batch.column(name)) for name in cols}
        for i in range(len(batch)):
            if arrays['policy_is_resign'][i] or arrays['policy_is_flag'][i]: continue
            tc = f"{int(arrays[base][i])}+{int(arrays['active_inc'][i])}"
            if tc not in controls: continue
            ply = int(arrays['move_ply'][i])
            if not 0 <= ply < 160: continue
            rating = int(arrays['active_elo'][i])
            # Pool both colours by their own move number, with 200-point Elo cells.
            bucket = max(1200, min(2400, rating // 200 * 200))
            n = ply // 2
            target = float(arrays['time_target'][i])
            clock = float(arrays['active_clock'][i])
            if not (np.isfinite(target) and target >= 0 and np.isfinite(clock) and clock >= 0): continue
            for r in (bucket, 0):
                a = stats[(tc,r,n)]
                a[0] += 1; a[1] += clock
                a[2] += 1; a[3] += target
            # Opponent has ceil(ply/2) completed own moves.
            opp_rating = int(arrays['opp_elo'][i])
            if 1200 <= opp_rating <= 2500:
                opp_bucket = max(1200, min(2400, opp_rating // 200 * 200))
                opp_clock = float(arrays['opp_clock'][i])
                if np.isfinite(opp_clock) and opp_clock >= 0:
                    for r in (opp_bucket,0):
                        a = stats[(tc,r,(ply+1)//2)]
                        a[0] += 1; a[1] += opp_clock
            rows += 1
    print(file.name, rows, flush=True)
profiles = {}
for (tc,r,n), (count,clock,count_time,timing) in stats.items():
    profiles.setdefault(tc,{}).setdefault(str(r),{})[str(n)] = {
        'count':count, 'clock':round(clock/count,4),
        'timeCount':count_time, 'time':round(timing/count_time,4) if count_time else None}
output = {'version':1,'source':'Marvin data/testset_2026-05; clock-annotated Lichess May 2026 prefix',
          'rows':rows,'ratingBucket':200,'minimumSamples':30,
          'method':'Arithmetic means by exact clock control, 200-point Elo cell, completed own moves; both colours pooled; no outcome conditioning.',
          'profiles':profiles}
args.output.write_text(json.dumps(output,separators=(',',':')),encoding='utf-8')
print('Wrote',args.output,'rows',rows,'controls',list(profiles),flush=True)
