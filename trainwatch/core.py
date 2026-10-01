"""
列車通過判讀：核心演算法（不含畫面）。

判讀方式（傳統影像處理，不用 AI）：
  1. 參考線：使用者在畫面上畫一條橫過軌道的線（寬度可調）。線上的像素和「沒有列車時的背景」差很多 → 參考線被佔用。
     佔用開始＝車頭到達參考線，佔用結束＝車尾離開參考線。雙軌同時有車時，重疊的佔用自然合併成一筆
     （開始取最早、結束取最晚）。
  2. 軌道範圍：使用者沿著軌道粗略框一個長方形。事件期間統計「軌道範圍內有變化的欄數比例」（覆蓋率），
     列車會佔滿很長一段；路上經過的汽車、行人、閃爍只佔一小段 → 用來排除非列車。
  3. 方向：軌道範圍內每一欄「第一次被擋到」與「最後一次被擋到」的時間。往右走 → 參考線右側的欄位比左側晚被擋到、
     也晚離開。兩者一致＝有把握；不一致時以車頭到達順序為準，並標記「需人工確認」。
  4. 時間：讀畫面上的時間字幕（osd.py）算出每支影片的時間對照，再加上每一格在影片裡的實際時間
     （晚上攝影機可能每秒只有 10 格，檔案標示的格數不準）。不使用檔名上的時間。

背景：沒有事件時跟著光線慢慢更新；事件進行中幾乎不更新，避免把列車學成背景。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import re
from typing import Callable, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # 測試環境
    cv2 = None


# ---------------------------------------------------------------- 設定
@dataclasses.dataclass
class Profile:
    """監測站設定（存成 JSON，下次同一攝影點直接套用）"""
    name: str = '未命名監測站'
    ref_line: list = dataclasses.field(default_factory=lambda: [[350, 165], [350, 215]])  # [[x1,y1],[x2,y2]]（影片原始像素）
    ref_width: int = 7                    # 參考線判讀寬度（像素）
    track_rect: list = dataclasses.field(default_factory=lambda: [200, 160, 470, 220])   # [x0,y0,x1,y1]
    on_level: float = 15.0                # 參考線變化量超過 → 佔用
    off_level: float = 10.0               # 低於 → 視為沒有佔用（遲滯）
    pixel_diff: float = 25.0              # 單一像素和背景差多少算「有變化」
    merge_gap: float = 1.0                # 佔用中斷少於幾秒視為同一筆（車廂之間的空隙）
    min_duration: float = 0.6             # 少於幾秒視為閃爍（排除，列在「已排除」）
    min_coverage: float = 0.25            # 軌道範圍覆蓋率最大值低於此值 → 非列車（排除）
    min_both: float = 0.15                # 參考線左右兩側「同時」都被擋住的比例最大值低於此值 → 非列車（車燈照射、路上的車）
    long_event: float = 600.0             # 超過幾秒標記「需人工確認」（可能停車或畫面變化）
    static_motion: float = 2.5            # 參考線上前後兩格的變化小於此值 → 視為畫面靜止
    static_sec: float = 3.0               # 佔用中畫面靜止超過幾秒 → 結束這一筆並重新學背景
                                          # （夜間列車經過後攝影機亮度會跳一下，背景對不上，會一直以為還有車）
    clip_before: float = 3.0              # 短片：事件前幾秒
    clip_after: float = 3.0               # 短片：事件後幾秒
    make_clips: bool = True
    osd_rect: list = dataclasses.field(default_factory=lambda: [14, 8, 204, 24])   # 時間字幕外框 [x0,y0,x1,y1]
    osd_format: str = 'YYYY-MM-DD HH:MM:SS'
    osd_templates: list = None            # 「教程式認字」學到的字樣；None＝內建

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d):
        p = Profile()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def horizontal(self):
        """軌道是左右走向（寬大於高）→ 方向寫往左／往右；否則寫往上／往下"""
        x0, y0, x1, y1 = self.track_rect
        return abs(x1 - x0) >= abs(y1 - y0)

    def ref_pos(self):
        """參考線在軌道方向上的位置（左右走向用 x，上下走向用 y）"""
        (xa, ya), (xb, yb) = self.ref_line
        return (xa + xb) / 2 if self.horizontal() else (ya + yb) / 2


# ---------------------------------------------------------------- 時間
def fmt_time(ts: float, with_date=False) -> str:
    d = dt.datetime.fromtimestamp(ts)
    s = d.strftime('%H:%M:%S') + '.%d' % int((ts - math.floor(ts)) * 10)
    return (d.strftime('%Y-%m-%d ') + s) if with_date else s


# ---------------------------------------------------------------- 判讀
class Detector:
    """逐格餵影像（灰階、已裁切到工作範圍），產生事件。可跨檔案連續使用（背景延續）。"""
    SIDE_WINDOW = 2.0      # 事件前後看幾秒
    EDGE_WINDOW = 1.5      # 事件開頭／結尾看幾秒
    MAX_REC = 20000        # 方向判斷最多記幾格（很長的事件不會吃光記憶體）

    def __init__(self, profile: Profile, crop):
        self.p = profile
        self.cx0, self.cy0, cx1, cy1 = crop
        h, w = cy1 - self.cy0, cx1 - self.cx0
        m = np.zeros((h, w), np.uint8)
        (xa, ya), (xb, yb) = profile.ref_line
        cv2.line(m, (int(xa - self.cx0), int(ya - self.cy0)), (int(xb - self.cx0), int(yb - self.cy0)), 255,
                 max(1, int(profile.ref_width)))
        self.ref_mask = m > 0
        x0, y0, x1, y1 = profile.track_rect
        self.tr = (int(min(x0, x1) - self.cx0), int(min(y0, y1) - self.cy0),
                   int(max(x0, x1) - self.cx0), int(max(y0, y1) - self.cy0))
        self.horiz = profile.horizontal()
        rp = profile.ref_pos() - (min(x0, x1) if self.horiz else min(y0, y1))
        self.ref_idx = int(round(rp))
        self.bg = None
        self.hist = []          # 最近幾秒的 (t, 覆蓋向量)
        self.cur = None         # 進行中的事件
        self.pending_close = None
        self.events: List[dict] = []
        self.prev = None
        self.first_t = None     # 第一個檔案第一格的時間
        self.small = self.bg_small = self.out_mask = None   # 整個畫面縮小版（估計攝影機自動調亮度）

    def feed(self, t_abs: float, gray: np.ndarray, sat: float, file: str, pos: float, frame_bgr=None):
        g = gray.astype(np.float32)
        if self.first_t is None:
            self.first_t = t_abs
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g.copy()
        gain = self._exposure(frame_bgr)
        if gain != 1.0:
            g = g / gain        # 攝影機自動調亮度（例如亮的列車經過後整個畫面變暗）→ 先還原再比較
        diff = np.abs(g - self.bg)
        motion = float(np.abs(g - self.prev)[self.ref_mask].mean()) if (self.prev is not None and self.prev.shape == g.shape) else 99.0
        self.prev = g
        score = float(diff[self.ref_mask].mean()) if self.ref_mask.any() else 0.0
        tx0, ty0, tx1, ty1 = self.tr
        tr = diff[ty0:ty1, tx0:tx1] > self.p.pixel_diff
        prof = tr.mean(axis=0) if self.horiz else tr.mean(axis=1)   # 沿軌道方向每一欄（列）的變化比例
        active = prof > 0.3
        cov = float(active.mean()) if active.size else 0.0
        k = self.ref_idx
        lft, rgt = active[:max(1, k - 3)], active[k + 3:]
        both = min(float(lft.mean()) if lft.size else 0.0, float(rgt.mean()) if rgt.size else 0.0)
        self.hist.append((t_abs, active))
        while self.hist and t_abs - self.hist[0][0] > self.SIDE_WINDOW + 0.2:
            self.hist.pop(0)

        on = score > self.p.on_level
        occupied = score > (self.p.off_level if (self.cur is not None and self.pending_close is None) else self.p.on_level)
        if self.cur is None:
            if on:
                self.cur = dict(at_file_start=(self.first_t is not None and t_abs - self.first_t < 2.0),start=t_abs, start_file=file, start_pos=pos, end=t_abs, end_file=file, end_pos=pos,
                                peak=score, coverage=cov, sat=[sat], rec=list(self.hist),
                                last_motion=t_abs, scores=[(t_abs, score, file, pos)], tail=[(t_abs, score, file, pos)],
                                static_end=False)
        else:
            c = self.cur
            if len(c['rec']) < self.MAX_REC:
                c['rec'].append((t_abs, active))
            if motion > self.p.static_motion:
                c['last_motion'] = t_abs
            if t_abs - c['start'] < 15 and len(c['scores']) < 400:
                c['scores'].append((t_abs, score, file, pos))
            c['tail'].append((t_abs, score, file, pos))
            while t_abs - c['tail'][0][0] > 15:
                c['tail'].pop(0)
            if occupied and t_abs - c['last_motion'] > self.p.static_sec:
                # 還算「佔用」但參考線上已經好幾秒完全不動：車已經走了、背景（亮度）變了 → 在最後有動靜的時間結束
                if c['last_motion'] < c['end']:
                    c['end'] = c['last_motion']
                    c['end_file'], c['end_pos'] = file, pos - (t_abs - c['last_motion'])
                c['static_end'] = True
                self._close()
                self.bg = g * gain
                if self.small is not None:
                    self.bg_small = self.small.copy()
                return
            if occupied:
                self.pending_close = None
                c['end'], c['end_file'], c['end_pos'] = t_abs, file, pos
                c['peak'] = max(c['peak'], score)
                c['coverage'] = max(c['coverage'], cov)
                c['both'] = max(c.get('both', 0.0), both)
                c['sat'].append(sat)
            else:
                if self.pending_close is None:
                    self.pending_close = t_abs
                if t_abs - c['end'] > max(self.p.merge_gap, self.SIDE_WINDOW):
                    self._close()
        # 背景更新：沒有事件時一律學習（畫面某處永久改變時才不會一直卡在「有點不一樣」）；事件中幾乎不動
        a = 0.02 if self.cur is None else 0.0005
        if self.cur is not None and t_abs - self.cur['start'] > self.p.long_event:
            a = 0.02   # 佔用太久（停車或畫面整個變了）：開始重新學背景
        self.bg = self.bg * (1 - a) + g * gain * a
        if self.small is not None:
            if self.bg_small is None or self.bg_small.shape != self.small.shape:
                self.bg_small = self.small.copy()
            else:
                self.bg_small = self.bg_small * (1 - a) + self.small * a

    EXPO_SCALE = 8

    def _exposure(self, frame_bgr):
        """整個畫面（扣掉軌道範圍附近與上方字幕）目前亮度 ÷ 背景亮度的中位數"""
        self.small = None
        if frame_bgr is None:
            return 1.0
        k = self.EXPO_SCALE
        self._expo_n = getattr(self, '_expo_n', 0) + 1
        sm = frame_bgr[::k, ::k, 1].astype(np.float32)        # 綠色頻道、每 8 點取 1 點（夠估亮度，快很多）
        self.small = sm
        self._gain = self._exposure_calc(sm)
        return self._gain

    def _exposure_calc(self, sm):
        k = self.EXPO_SCALE
        if self.bg_small is None or self.bg_small.shape != sm.shape:
            return 1.0
        if self.out_mask is None or self.out_mask.shape != sm.shape:
            m = np.ones(sm.shape, bool)
            x0, y0, x1, y1 = self.p.track_rect
            m[max(0, (min(y0, y1) - 20) // k):(max(y0, y1) + 20) // k + 1, max(0, (min(x0, x1) - 20) // k):(max(x0, x1) + 20) // k + 1] = False
            m[:40 // k] = False
            self.out_mask = m
        sel = self.out_mask & (self.bg_small > 15) & (self.bg_small < 245)
        if sel.sum() < 50:
            return 1.0
        r = float(np.median(sm[sel] / self.bg_small[sel]))
        return r if 0.5 < r < 2.0 and abs(r - 1) > 0.01 else 1.0

    def finish(self):
        if self.cur is not None:
            self._close()
        return self.events

    def _onset(self, c):
        """回傳 (車頭時間差, 車尾時間差)：參考線後側（右／下）中位數減前側（左／上）中位數，>0 表示往右／往下"""
        rec = [(t, a) for (t, a) in c['rec'] if c['start'] - self.SIDE_WINDOW <= t <= c['end'] + self.SIDE_WINDOW]
        if len(rec) < 3:
            return 0.0, 0.0
        T = np.array([t for t, _ in rec])
        A = np.stack([a for _, a in rec])
        # 每一欄找「連續有變化至少 0.4 秒」的區段：第一段的開頭＝被擋到，最後一段的結尾＝離開（零星閃動不算）
        on, off, cols = [], [], []
        for j in range(A.shape[1]):
            col = A[:, j]
            if not col.any():
                continue
            runs, i, n = [], 0, len(col)
            while i < n:
                if col[i]:
                    k2 = i
                    while k2 + 1 < n and col[k2 + 1]:
                        k2 += 1
                    if T[k2] - T[i] >= 0.4:
                        runs.append((T[i], T[k2]))
                    i = k2 + 1
                else:
                    i += 1
            if runs:
                cols.append(j); on.append(runs[0][0]); off.append(runs[-1][1])
        if len(cols) < 6:
            return 0.0, 0.0
        cols, on, off = np.array(cols), np.array(on), np.array(off)
        k = self.ref_idx
        L, R = cols < k - 5, cols > k + 5
        if not L.any() or not R.any():
            return 0.0, 0.0
        return float(np.median(on[R]) - np.median(on[L])), float(np.median(off[R]) - np.median(off[L]))

    def _level(self, c):
        vals = np.array([v for _, v, _, _ in (c.get('scores') or [])] + [v for _, v, _, _ in (c.get('tail') or [])])
        if len(vals) < 4:
            return None
        return max(self.p.on_level, 0.5 * float(np.percentile(vals, 80)))

    def _refine(self, c):
        """參考線變化量在事件開頭與結尾常有一段「有點不一樣、但不是車身」的尾巴：
        夜間車燈在車頭到達前先照亮參考線；車尾離開後背景（光線）已經和事件開始前不同。
        以事件中變化量的高位數（第 80 百分位）的一半為門檻：第一次達到＝車頭到達，最後一次達到＝車尾離開。
        回傳 (開頭往後移幾秒, 結尾往前移幾秒)"""
        lvl = self._level(c)
        if lvl is None:
            return 0.0, 0.0
        s_shift = e_shift = 0.0
        for t, v, f, pos in c['scores']:
            if v >= lvl:
                if t - c['start'] > 0.3:
                    s_shift = t - c['start']
                    c['start'], c['start_file'], c['start_pos'] = t, f, pos
                break
        for t, v, f, pos in reversed([x for x in c['tail'] if x[0] <= c['end']]):
            if v >= lvl:
                if c['end'] - t > 0.3 and t > c['start']:
                    e_shift = c['end'] - t
                    c['end'], c['end_file'], c['end_pos'] = t, f, pos
                break
        return s_shift, e_shift

    def _close(self):
        c = self.cur
        self.cur = None
        self.pending_close = None
        first_change, last_change = c['start'], c['end']
        d_on, d_off = self._onset(c)          # 方向用完整的變化（含車燈）判斷
        shift, eshift = self._refine(c)
        th = 0.15
        if abs(d_on) >= th and abs(d_off) >= th and (d_on > 0) == (d_off > 0):
            sign, agree = (1 if d_on > 0 else -1), True
        elif abs(d_on) >= th:
            sign, agree = (1 if d_on > 0 else -1), False          # 以車頭到達的順序為準
        elif abs(d_off) >= th:
            sign, agree = (1 if d_off > 0 else -1), False
        else:
            sign, agree = 0, False
        if sign == 0:
            direction = '無法判定'
        elif self.horiz:
            direction = '往右' if sign > 0 else '往左'
        else:
            direction = '往下' if sign > 0 else '往上'
        night = float(np.median(c['sat'])) < 12
        dur = c['end'] - c['start']
        ev = dict(first_change=first_change, start=c['start'], end=c['end'], start_file=c['start_file'], end_file=c['end_file'],
                  start_pos=c['start_pos'], end_pos=c['end_pos'], coverage=round(c['coverage'], 3), both=round(c.get('both', 0.0), 3), peak=round(c['peak'], 1),
                  direction=direction, dir_on=round(d_on, 2), dir_off=round(d_off, 2), dir_confident=bool(agree),
                  night=night, valid=True, reason='', need_check=False)
        reasons = []
        if dur < self.p.min_duration:
            ev['valid'] = False
            reasons.append('時間太短（%.1f 秒，可能是閃爍或畫面亮度跳動）' % dur)
        elif c['coverage'] < self.p.min_coverage:
            ev['valid'] = False
            reasons.append('軌道範圍內變化範圍太小（%.0f%%，可能是汽車、行人或樹葉）' % (c['coverage'] * 100))
        elif c.get('both', 0.0) < self.p.min_both:
            ev['valid'] = False
            reasons.append('只有參考線一側有變化，沒有東西橫跨參考線（可能是車燈照射或路上的車）')
        if ev['valid']:
            if not agree:
                ev['need_check'] = True
                reasons.append('方向不確定')
            if c.get('static_end'):
                ev['need_check'] = True
                reasons.append('結束前畫面靜止（若列車停在參考線上，車尾時間請確認）')
            if dur > self.p.long_event:
                ev['need_check'] = True
                reasons.append('佔用時間很長（可能停車或畫面變化）')
            if shift > 0:
                reasons.append('參考線在 %s 就開始有微小變化（可能是車燈先照到），車頭時間取變化明顯的時刻' % fmt_time(first_change))
            if c.get('at_file_start'):
                ev['need_check'] = True
                reasons.append('影片一開始就有變化（列車可能在影片開始前就已到達，車頭時間是影片開頭）')
            if night:
                reasons.append('夜間')
        ev['reason'] = '；'.join(reasons)
        ev['last_change'] = last_change
        self.events.append(ev)


def work_crop(profile: Profile, w: int, h: int, pad: int = 8):
    """只處理參考線與軌道範圍的外框，速度快很多"""
    xs = [profile.ref_line[0][0], profile.ref_line[1][0], profile.track_rect[0], profile.track_rect[2]]
    ys = [profile.ref_line[0][1], profile.ref_line[1][1], profile.track_rect[1], profile.track_rect[3]]
    x0 = max(0, int(min(xs)) - pad); x1 = min(w, int(max(xs)) + pad + 1)
    y0 = max(0, int(min(ys)) - pad); y1 = min(h, int(max(ys)) + pad + 1)
    return x0, y0, x1, y1


def draw_overlay(frame, profile: Profile, color_ref=(0, 0, 255), color_track=(0, 200, 255)):
    f = frame.copy()
    x0, y0, x1, y1 = [int(v) for v in profile.track_rect]
    cv2.rectangle(f, (x0, y0), (x1, y1), color_track, 1)
    (xa, ya), (xb, yb) = profile.ref_line
    cv2.line(f, (int(xa), int(ya)), (int(xb), int(yb)), color_ref, 2)
    return f


def process(files: List[str], bases: List[float], profile: Profile,
            progress: Callable[[float, str], None] = None, cancel: Callable[[], bool] = None):
    """依序處理多個檔案（同一攝影機）。bases：每個檔案「影片內 0 秒」對應的畫面時間（epoch 秒，由 osd.calibrate 算出）。
    回傳事件 list（dict）"""
    det = None
    durs = []
    for f in files:
        info = video_info(f)
        durs.append((info or {}).get('dur') or 600)
    total = sum(durs) or 1
    done = 0.0
    for fi, (f, base) in enumerate(zip(files, bases)):
        cap = cv2.VideoCapture(f)
        if not cap.isOpened():
            raise RuntimeError('無法開啟影片：' + os.path.basename(f))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop = work_crop(profile, W, H)
        if det is None:
            det = Detector(profile, crop)
            det.bg = initial_background(f, crop)   # 開頭若剛好有列車，不會被當成背景
        k = 0
        last_pos = 0.0
        sat = 50.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos <= last_pos and k > 0:   # 少數格式讀不到時間 → 用上一格推算
                pos = last_pos + 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15)
            last_pos = pos
            x0, y0, x1, y1 = crop
            sub = fr[y0:y1, x0:x1]
            gray = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if k % 15 == 0:
                sat = float(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            det.feed(base + pos, gray, sat, f, pos, fr)
            k += 1
            if progress and k % 150 == 0:
                progress(min(1.0, (done + pos) / total), '%s（%d/%d）' % (os.path.basename(f), fi + 1, len(files)))
            if cancel and k % 30 == 0 and cancel():
                cap.release()
                return det.finish()
        cap.release()
        done += durs[fi]
    return det.finish() if det else []


def initial_background(path, crop, seconds=90.0, n=31):
    """第一支影片開頭 seconds 秒內平均取 n 格，取中位數當初始背景"""
    cap = cv2.VideoCapture(path)
    x0, y0, x1, y1 = crop
    frames, step, nxt = [], seconds / n, 0.0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos > seconds:
                break
            if pos >= nxt:
                frames.append(cv2.GaussianBlur(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0))
                nxt += step
    finally:
        cap.release()
    if len(frames) < 3:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.float32)


# ---------------------------------------------------------------- 截圖與短片
def save_frames_and_clips(events, profile: Profile, out_dir: str, progress=None, cancel=None):
    """每筆事件存開始／中間／結束截圖，並（可選）擷取前後幾秒的短片"""
    shot_dir = os.path.join(out_dir, '截圖'); os.makedirs(shot_dir, exist_ok=True)
    clip_dir = os.path.join(out_dir, '短片')
    if profile.make_clips:
        os.makedirs(clip_dir, exist_ok=True)
    for i, ev in enumerate(events):
        tag = '%03d' % ev['no'] if ev.get('valid') else 'X%03d' % ev['no']
        ev['shots'] = []
        same = ev['start_file'] == ev['end_file']
        spots = (('1車頭', ev['start_file'], ev['start_pos']),
                 ('2中間', ev['start_file'], (ev['start_pos'] + ev['end_pos']) / 2 if same else ev['start_pos'] + 1.0),
                 ('3車尾', ev['end_file'], ev['end_pos']))
        for lab, f, pos in spots:
            fr, _p = read_frame_at(f, pos)
            if fr is None:
                continue
            name = '%s_%s.jpg' % (tag, lab)
            imwrite(os.path.join(shot_dir, name), draw_overlay(fr, profile))
            ev['shots'].append(name)
        ev['clip'] = ''
        if profile.make_clips:
            name = '%s.mp4' % tag
            try:
                write_clip(ev, profile, os.path.join(clip_dir, name))
                ev['clip'] = name
            except Exception as e:  # 短片失敗不影響其他結果
                ev['reason'] = (ev.get('reason', '') + '；短片產生失敗：' + str(e)).strip('；')
        if progress:
            progress((i + 1) / max(1, len(events)), '產生截圖與短片 %d/%d' % (i + 1, len(events)))
        if cancel and cancel():
            break


def imwrite(path, img):
    """cv2.imwrite 不支援中文路徑（Windows）→ 用 imencode"""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        with open(path, 'wb') as fh:
            fh.write(buf.tobytes())


def write_clip(ev, profile: Profile, path):
    segs = []
    if ev['start_file'] == ev['end_file']:
        segs.append((ev['start_file'], max(0.0, ev['start_pos'] - profile.clip_before), ev['end_pos'] + profile.clip_after))
    else:   # 跨兩個檔案
        segs.append((ev['start_file'], max(0.0, ev['start_pos'] - profile.clip_before), 1e9))
        segs.append((ev['end_file'], 0.0, ev['end_pos'] + profile.clip_after))
    # VideoWriter 也不支援中文路徑 → 先寫到暫存英文檔名再搬
    tmp = os.path.join(os.path.dirname(path), '_tmp_clip.mp4')
    OUT_FPS = 15.0
    wr = None
    t_out = 0.0          # 已輸出的長度（秒）。依影片實際時間補格／跳格，播放速度才會正確
    try:
        for f, a, b in segs:
            cap = cv2.VideoCapture(f)
            seek_to(cap, a)
            t0 = None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if pos < a:
                    continue
                if pos > b:
                    break
                if t0 is None:
                    t0 = pos - 1e-6
                    seg_base = t_out
                img = draw_overlay(fr, profile)
                if wr is None:
                    h, w = fr.shape[:2]
                    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), OUT_FPS, (w, h))
                target = seg_base + (pos - t0)
                while t_out <= target:
                    wr.write(img)
                    t_out += 1.0 / OUT_FPS
            cap.release()
    finally:
        if wr is not None:
            wr.release()
    if os.path.exists(tmp):
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)


def seek_to(cap, t):
    """跳到 t 秒之前一點（MKV 跳轉常常跳過頭，所以先往前多退，再逐格讀到 t）"""
    back = 3.0
    for _ in range(5):
        start = max(0.0, t - back)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        ok, _fr = cap.read()
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ok or pos <= t or start <= 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            return
        back *= 2
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)


def read_frame_at(path, t):
    """讀取影片 t 秒的那一格（預覽用）；回傳 (frame, 實際秒數)"""
    cap = cv2.VideoCapture(path)
    try:
        seek_to(cap, t)
        best = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            best = (fr, pos)
            if pos >= t - 0.05:
                break
        return best if best else (None, 0.0)
    finally:
        cap.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return dict(w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=fps, dur=(n / fps if n > 0 else 0.0))
    finally:
        cap.release()


def number_events(events):
    """依時間排序；有效事件編號 1,2,3…，已排除的另外編號（Excel 顯示 X1,X2…）"""
    events.sort(key=lambda e: e['start'])
    a = b = 0
    for e in events:
        if e['valid']:
            a += 1; e['no'] = a
        else:
            b += 1; e['no'] = b
    return events
