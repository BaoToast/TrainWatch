r"""
自我測試：產生一段「假列車」影片（灰色背景＋一個方塊由左往右、一個由右往左通過），用正式的判讀流程跑一次。
用途：① 單元測試 ② GitHub 打包完成後，用免安裝資料夾裡的 Python 在 Windows 上實際驗證（影片讀寫、OpenCV、判讀都能動）。
執行：runtime\python.exe app\main.py --selftest 結果.json
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


def make_scene(path, seconds, trains=(), exposure=(), blobs=(), local_light=(), seed=2, vertical=False, ir_at=None, camera_shift=None, texture=400):
    """較完整的假影片（驗收測試用）。
    trains：dict(t0=車頭碰到參考線 x=350 的秒數, d=+1 往右／-1 往左, speed=像素／秒, length=車長像素,
                 y=(上, 下), stops=[(開始停的秒數, 停多久), …], car=每節車廂長, gap=車廂間空隙像素)
    exposure：[(開始秒, 結束秒, 亮度倍率)]（整個畫面一起變亮／變暗，模擬攝影機自動調亮度）
    blobs：[(開始秒, 結束秒, x, y, 半徑)]（亮光，模擬車燈照射）
    local_light：[(開始秒, 結束秒, 亮度倍率)]（只有軌道一帶 y=140～240 變亮／變暗，模擬陽光、雲影；過渡 2 秒）
    vertical：True＝整個畫面轉 90 度（360×640，軌道上下走向，列車往下／往上；設定用 VERTICAL_PROFILE）
    ir_at：從這一秒起畫面變成紅外線模式（黑白、對比和亮度都不同），模擬傍晚攝影機切換
    camera_shift：(秒, dx, dy, 轉幾度)，從這一秒起整個畫面移動／轉動，模擬攝影機被碰歪
    texture：背景上的圓點數（越多紋理越密，像真實的樹叢、電線桿）
    回傳每列車預期的 (車頭到達, 車尾離開)"""
    import cv2
    rng = np.random.default_rng(seed)
    bg = np.full((H, W, 3), 110, np.uint8)
    bg[:, :, 1] = 130
    for _ in range(texture):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(2, 8)), tuple(int(v) for v in rng.integers(40, 200, 3)), -1)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (H, W) if vertical else (W, H))
    if not wr.isOpened():
        raise RuntimeError('無法建立測試影片')

    def moved(tr, t):
        """t 秒時車頭已經走了多少秒（扣掉停車時間）；以車頭碰到參考線為 0"""
        m = t - tr['t0']
        for a, dur in tr.get('stops', ()):
            if t > a:
                m -= min(dur, t - a)
        return m
    expect = []
    for tr in trains:
        stop_total = sum(d for _a, d in tr.get('stops', ()))
        expect.append((tr['t0'], tr['t0'] + tr.get('length', 260) / tr.get('speed', 120.0) + stop_total))
    for i in range(int(seconds * FPS)):
        t = i / FPS
        fr = bg.copy()
        for tr in trains:
            d, sp, ln = tr.get('d', 1), tr.get('speed', 120.0), tr.get('length', 260)
            y0, y1 = tr.get('y', (168, 214))
            head = 350 + d * moved(tr, t) * sp
            car, gap = tr.get('car', ln), tr.get('gap', 0)
            k = 0.0
            while k < ln:                                # 一節一節畫（車廂之間留空隙）
                a_, b_ = head - d * k, head - d * min(ln, k + car)
                x0, x1 = sorted((int(a_), int(b_)))
                if x1 >= 0 and x0 < W:
                    cv2.rectangle(fr, (max(0, x0), y0), (min(W - 1, x1), y1), (230, 230, 235), -1)
                    for wx in range(x0 + 10, x1 - 24, 30):       # 車窗跟著車身移動（座標從車廂算，不從畫面邊緣算）
                        if -14 < wx < W:
                            cv2.rectangle(fr, (wx, y0 + 8), (wx + 14, y0 + 20), (40, 40, 60), -1)
                k += car + gap
        for a_, b_, x, y, r in blobs:
            if a_ <= t <= b_:
                cv2.circle(fr, (x, y), r, (255, 255, 255), -1)
        for a_, b_, fac in local_light:
            if a_ <= t <= b_ + 2:
                k = min(1.0, (t - a_) / 2.0) if t <= b_ else max(0.0, 1 - (t - b_) / 2.0)
                band = fr[140:240].astype(np.float32) * (1 + (fac - 1) * k)
                fr[140:240] = np.clip(band, 0, 255).astype(np.uint8)
        f = 1.0
        for a_, b_, fac in exposure:
            if a_ <= t <= b_:
                f = fac
        out = fr.astype(np.float32) * f + rng.normal(0, 2.5, fr.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if ir_at is not None and t >= ir_at:
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = np.clip((g - 110) * 1.6 + 150, 0, 255).astype(np.uint8)     # 紅外線：黑白、比較亮、對比不同
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if camera_shift is not None and t >= camera_shift[0]:
            _t, dx, dy, ang = camera_shift
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
            M[0, 2] += dx; M[1, 2] += dy
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        if vertical:
            out = np.ascontiguousarray(out.transpose(1, 0, 2))
        wr.write(out)
    wr.release()
    return expect


def vertical_profile():
    """make_scene(vertical=True) 用的監測站設定（原本的參考線、軌道範圍 x、y 對調）"""
    p = core.Profile()
    p.ref_line = [[y, x] for x, y in p.ref_line]
    x0, y0, x1, y1 = p.track_rect
    p.track_rect = [y0, x0, y1, x1]
    return p


def run_scene(path, seconds, profile=None, **kw):
    """產生場景並用正式流程判讀；回傳 (有效事件[(開始, 結束, 方向, 備註, 需確認)], 預期)"""
    expect = make_scene(path, seconds, **kw)
    prof = profile or (vertical_profile() if kw.get('vertical') else core.Profile())
    base = dt.datetime(2026, 1, 1, 8, 0, 0).timestamp()
    evs = core.number_events(core.process([path], [base], prof))
    got = [(round(e['start'] - base, 2), round(e['end'] - base, 2), e['direction'], e['reason'], e.get('need_check'))
           for e in evs if e['valid']]
    return got, expect


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
