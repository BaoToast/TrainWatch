"""
自我測試：產生一段「假列車」影片（灰色背景＋一個方塊由左往右、一個由右往左通過），用正式的判讀流程跑一次。
用途：① 單元測試 ② GitHub 打包完成後，在 Windows 上實際執行 exe 驗證（影片讀寫、OpenCV、判讀都能動）。
執行：TrainWatch.exe --selftest 結果.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

from . import core, export

W, H, FPS = 640, 360, 15.0


def make_video(path, seconds=40.0, trains=((5.0, +1), (22.0, -1)), speed=120.0, length=260, seed=1):
    """trains：(車頭碰到參考線 x=350 的秒數, 方向 +1 往右 / -1 往左)"""
    import cv2
    rng = np.random.default_rng(seed)
    bg = np.full((H, W, 3), 110, np.uint8)
    bg[:, :, 1] = 130
    for _ in range(400):                       # 一些固定的紋理（樹、電線桿）
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(2, 8)), tuple(int(v) for v in rng.integers(40, 200, 3)), -1)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    if not wr.isOpened():
        raise RuntimeError('無法建立測試影片')
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        fr = bg.copy()
        for t0, d in trains:
            head = 350 + d * (t - t0) * speed          # 車頭位置
            tail = head - d * length
            x0, x1 = sorted((int(head), int(tail)))
            if x1 >= 0 and x0 < W:
                cv2.rectangle(fr, (max(0, x0), 168), (min(W - 1, x1), 214), (230, 230, 235), -1)
                for wx in range(max(0, x0) + 10, min(W - 1, x1) - 10, 30):   # 車窗
                    cv2.rectangle(fr, (wx, 176), (wx + 14, 190), (40, 40, 60), -1)
        noise = rng.integers(-4, 5, fr.shape, dtype=np.int16)
        fr = np.clip(fr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        wr.write(fr)
    wr.release()
    return [(t0, t0 + length / speed, d) for t0, d in trains]   # 預期 (車頭到達, 車尾離開, 方向)


def run(out_json=None):
    tmp = tempfile.mkdtemp(prefix='trainwatch_selftest_')
    vid = os.path.join(tmp, 'selftest.avi')
    expect = make_video(vid)
    prof = core.Profile(name='自我測試', make_clips=True)
    base = dt.datetime(2026, 1, 1, 8, 0, 0).timestamp()
    evs = core.number_events(core.process([vid], [base], prof))
    out_dir = os.path.join(tmp, 'out')
    os.makedirs(out_dir, exist_ok=True)
    core.save_frames_and_clips(evs, prof, out_dir)
    timing = [dict(offset=base, source='測試', support=1.0, msg='')]
    export.write_excel(os.path.join(out_dir, '列車通過紀錄.xlsx'), evs, prof, [vid], timing)
    export.write_csv(os.path.join(out_dir, '列車通過紀錄.csv'), evs)
    valid = [e for e in evs if e['valid']]
    problems = []
    if len(valid) != len(expect):
        problems.append('筆數 %d，預期 %d' % (len(valid), len(expect)))
    for e, (a, b, d) in zip(valid, expect):
        if abs((e['start'] - base) - a) > 0.4:
            problems.append('車頭 %.2f 預期 %.2f' % (e['start'] - base, a))
        if abs((e['end'] - base) - b) > 0.6:
            problems.append('車尾 %.2f 預期 %.2f' % (e['end'] - base, b))
        if e['direction'] != ('往右' if d > 0 else '往左'):
            problems.append('方向 %s 預期 %s' % (e['direction'], '往右' if d > 0 else '往左'))
        if not e.get('clip') or not os.path.exists(os.path.join(out_dir, '短片', e['clip'])):
            problems.append('短片沒有產生')
        if len(e.get('shots') or []) != 3:
            problems.append('截圖不是 3 張')
    rep = dict(ok=not problems, problems=problems,
               events=[dict(start=round(e['start'] - base, 2), end=round(e['end'] - base, 2), direction=e['direction'],
                            valid=e['valid'], reason=e['reason']) for e in evs],
               expect=expect, workdir=tmp)
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
    return rep


if __name__ == '__main__':
    r = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if r['ok'] else 1)
