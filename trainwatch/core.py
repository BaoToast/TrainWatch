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
    long_event: float = 600.0             # 超過幾秒只標記「需人工確認」（不會因此結束，也不會重學背景）
    static_motion: float = 2.5            # 參考線上前後兩格的變化小於此值 → 視為畫面靜止（列車停住）
    static_sec: float = 3.0               # 佔用中靜止超過幾秒 → 備註「列車曾停止」（只記錄，不結束事件）
    day_saturation: float = 25.0          # 畫面彩度高於此值＝白天（夜間紅外線模式是黑白，彩度接近 0）
    light_ncc: float = 0.45               # 白天：參考線附近和背景的紋理相似度高於此值 → 只是光線變亮／變暗（陽光、雲影），不是列車
    max_static: float = 1800.0            # 佔用中「完全靜止」超過幾秒標需人工確認（v1.0.10 起只標記、不結束事件，#67）
                                          # （防止攝影機被撞歪、畫面永久改變時，整天變成一筆）
    clip_before: float = 3.0              # 短片：事件前幾秒
    clip_after: float = 3.0               # 短片：事件後幾秒
    make_clips: bool = True
    osd_rect: list = dataclasses.field(default_factory=lambda: [14, 8, 204, 24])   # 時間字幕外框 [x0,y0,x1,y1]
    osd_format: str = 'YYYY-MM-DD HH:MM:SS'
    osd_templates: list = None            # 「教程式認字」學到的字樣；None＝內建
    frame_size: list = None               # 畫參考線時的畫面大小 [寬, 高]；影片大小不同時要警告

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
class CameraGuard:
    """攝影機位置基準（#75～#77）。和列車背景（Detector.bg）分開：
    - 基準＝批次開頭前 2 秒內 5 格「軌道以外」畫面的邊緣強度圖中位數（不用 90 秒中位數：攝影機在那段期間移動時，
      移動後的畫面會佔多數，反而把舊位置當成異常）。
    - 比較方法：cv2.phaseCorrelate（相位相關）算目前畫面相對基準「平移了幾像素」與可信度 r。
      用邊緣強度、再除以高位數正規化 → 整體亮度、曝光變化不影響。
      真實樣本實測：同一光線下 5 秒～10 分鐘，平移 < 0.3 像素、r ≥ 0.74；人為平移 4×3、10×0 像素量得到誤差 < 0.5；
      移動 14×9＋轉 1.5 度：量到 13～14 像素平移。
    - 判定「位置改變」：平移 ≥ MOVE_PX（6 像素）且 r ≥ 0.15 開始懷疑；之後量到的平移一直穩定（方向、大小差 < 3 像素），
      持續 HOLD 秒（有列車通過時 HOLD_EVENT 秒）就確認。
      **不用「r 變低」當移動的證據**：白天有風時樹葉晃動，r 在 10 秒內就會從 0.9 掉到 0.5 以下（實測），會大量誤判。
    - 基準何時更新：每 UPDATE 秒，如果目前畫面和基準「沒有平移（< 1 像素）而且 r ≥ 0.3」，就換成目前畫面
      （基準一直保持是幾秒前的畫面，跟上樹葉、光線、陰影）。攝影機被碰是突然的平移，不會被慢慢學進去。
      **疑似移動期間（suspect）完全不更新**；錄影中斷、重建列車背景也不更新（錄影中斷另外用 gap_check 比對）。
    - 超過 STALE 秒都沒辦法更新（r 一直很低，例如彩色↔紅外線切換、光線劇烈變化），但也沒有平移：重新建立基準，
      並記在 notes（品質摘要、Excel 會寫出來，不是默默發生）。
    - 已知限制：只有轉動、幾乎沒有平移的碰撞偵測不到（轉 1.5 度時，畫面中央附近的參考線只移動約 1 像素）。"""
    K = 2               # 畫面縮成 1/2 再比（320×180）
    MOVE_PX = 6.0       # 平移幾像素（原始 640×360）以上算移動。6 像素在參考線上約差 0.02～0.06 秒，使用者可接受誤差是 5 秒
    HOLD = 5.0
    HOLD_EVENT = 5.0     # 列車在軌道範圍裡，比對時已經遮掉，所以和平常一樣
    UPDATE = 2.0
    STALE = 60.0
    STEP = 0.5          # 每幾秒比一次

    def __init__(self, profile: Profile, size=(640, 360)):
        self.p = profile
        self.size = None
        self.ref = None
        self.ref_t = None
        self.build = []          # 建立基準用的前幾格
        self.build_t0 = None
        self.last = -1e9
        self.suspect = None      # (開始時間, 影片, 秒數)
        self.suspect_v = (0.0, 0.0)
        self.mode = None         # 'day'／'night'
        self.mode_t = -1e9
        self.notes = []
        self.last_result = None

    def _mask(self, w, h):
        k = self.K
        m = np.ones((h // k, w // k), bool)
        x0, y0, x1, y1 = self.p.track_rect
        m[max(0, (min(y0, y1) - 20) // k):(max(y0, y1) + 20) // k + 1, max(0, (min(x0, x1) - 20) // k):(max(x0, x1) + 20) // k + 1] = False
        m[:40 // k] = False      # 上方時間字幕
        return m

    def prep(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.size != (w, h):
            self.size = (w, h)
            self.mask = self._mask(w, h)
            self.win = cv2.createHanningWindow((w // self.K, h // self.K), cv2.CV_32F)
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (w // self.K, h // self.K), interpolation=cv2.INTER_AREA)
        e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        hi = float(np.percentile(e[self.mask], 99)) if self.mask.any() else 1.0
        e = np.minimum(e / (hi + 1e-6), 1.0)
        e[~self.mask] = 0
        return e

    def compare(self, a, b, vec=False):
        """回傳 (平移像素（原始大小）, r)；vec=True 另外回傳 (dx, dy)"""
        (dx, dy), r = cv2.phaseCorrelate(a, b, self.win)
        out = (float(np.hypot(dx, dy)) * self.K, float(r))
        return out + ((dx * self.K, dy * self.K),) if vec else out

    def check(self, t, frame_bgr, sat, active, file, pos):
        """每一格呼叫。確認位置改變時回傳 (改變開始的時間, 影片, 秒數)，否則 None"""
        if frame_bgr is None or t - self.last < self.STEP:
            return None
        self.last = t
        mode = 'day' if sat >= self.p.day_saturation else ('night' if sat < 12 else self.mode)
        if mode != self.mode:
            if self.mode is not None:
                self.mode_t = t
            self.mode = mode
        cur = self.prep(frame_bgr)
        if self.ref is None:
            if self.build_t0 is None:
                self.build_t0 = t
            self.build.append(cur)
            if len(self.build) >= 5 or t - self.build_t0 >= 2.0:
                self.ref, self.ref_t, self.build = np.median(np.stack(self.build), axis=0).astype(np.float32), t, []
            return None
        shift, r, v = self.compare(self.ref, cur, vec=True)
        self.last_result = (shift, r)
        # 疑似移動：第一次要 r ≥ 0.15；之後只要量到的平移方向、大小一直差不多（< 3 像素），就算 r 變低也持續
        # （基準停在移動前，樹葉晃動會讓 r 越來越低，但真正的平移量會一直穩定；雜訊造成的假平移每次都不一樣）
        start = shift >= self.MOVE_PX and r >= 0.15
        keep = (self.suspect is not None and shift >= self.MOVE_PX
                and np.hypot(v[0] - self.suspect_v[0], v[1] - self.suspect_v[1]) < 3.0)
        if keep or start:
            if self.suspect is None:
                self.suspect, self.suspect_v = (t, file, pos), v
            if t - self.suspect[0] >= (self.HOLD_EVENT if active else self.HOLD):
                return self.suspect
            return None                       # 疑似移動期間不更新基準
        self.suspect = None
        if shift < 1.0 and r >= 0.3:
            if t - self.ref_t >= self.UPDATE:
                self.ref, self.ref_t = cur, t
        elif t - self.ref_t >= self.STALE and shift < 3.0:
            self.ref, self.ref_t = cur, t
            why = ('切換為%s' % ('紅外線（黑白）' if self.mode == 'night' else '彩色')) if t - self.mode_t < self.STALE + 5 \
                else '畫面變化很大'
            self.notes.append(dict(file=file, t=t, text='%s：%s（%s）' % (fmt_time(t), why, os.path.basename(file or ''))))
        return None

    def gap_check(self, path):
        """錄影中斷後：拿中斷前最後的基準，和下一支影片前 2 秒的畫面比對。回傳 'same'／'moved'／'uncertain'。
        same＝確認位置一致（基準換成新影片的畫面，繼續判讀）；moved／uncertain＝停止判讀（不知道就不默默繼續）"""
        if self.ref is None:
            return 'same'
        frames = []
        cap = cv2.VideoCapture(path)
        try:
            while len(frames) < 5:
                ok, fr = cap.read()
                if not ok:
                    break
                p = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if p >= len(frames) * 0.4:
                    frames.append(self.prep(fr))
        finally:
            cap.release()
        if len(frames) < 3:
            return 'uncertain'
        new = np.median(np.stack(frames), axis=0).astype(np.float32)
        shift, r = self.compare(self.ref, new)
        self.last_result = (shift, r)
        if shift >= self.MOVE_PX and r >= 0.15:
            return 'moved'
        if shift < 3.0 and r >= GAP_SAME_R:
            self.ref, self.suspect, self.last = new, None, -1e9
            return 'same'
        return 'uncertain'


class Detector:
    """逐格餵影像（灰階、已裁切到工作範圍），產生事件。可跨檔案連續使用（背景延續）。"""
    SIDE_WINDOW = 2.0      # 事件前後看幾秒
    EDGE_WINDOW = 1.5      # 事件開頭／結尾看幾秒
    HEAD_SEC = 60.0        # 方向判斷：保留事件開頭幾秒的逐格資料
    TAIL_SEC = 60.0        #           與最後幾秒（列車停很久也不會吃光記憶體，車頭／車尾的證據都在）

    def __init__(self, profile: Profile, crop):
        self.p = profile
        self.cx0, self.cy0, cx1, cy1 = crop
        h, w = cy1 - self.cy0, cx1 - self.cx0
        m = np.zeros((h, w), np.uint8)
        (xa, ya), (xb, yb) = profile.ref_line
        cv2.line(m, (int(xa - self.cx0), int(ya - self.cy0)), (int(xb - self.cx0), int(yb - self.cy0)), 255,
                 max(1, int(profile.ref_width)))
        self.ref_mask = m > 0
        self.light_since = None   # 白天「只是光線變化」從什麼時候開始
        x0, y0, x1, y1 = profile.track_rect
        self.tr = (int(min(x0, x1) - self.cx0), int(min(y0, y1) - self.cy0),
                   int(max(x0, x1) - self.cx0), int(max(y0, y1) - self.cy0))
        self.horiz = profile.horizontal()
        # 紋理比對用：參考線「沿軌道方向」前後各加寬 10 像素的帶狀區域
        # （左右走向往左右加寬；上下走向往上下加寬。v1.0.8 以前一律往左右，上下走向時只把線拉長，修正清單 #51）
        self.band = cv2.dilate(m, np.ones((1, 21) if self.horiz else (21, 1), np.uint8)) > 0
        rp = profile.ref_pos() - (min(x0, x1) if self.horiz else min(y0, y1))
        self.ref_idx = int(round(rp))
        self.bg = None
        self.hist = []          # 最近幾秒的 (t, 覆蓋向量)
        self.cur = None         # 進行中的事件
        self.pending_close = None
        self.events: List[dict] = []
        self.prev = None
        self.first_t = None     # 第一個檔案第一格的時間（錄影中斷後重新計算）
        self.start_kind = 'video'   # 一開始就有變化時的原因：video＝第一支影片開頭、gap＝錄影中斷後
        self.bg_note = ''       # 初始背景的說明（背景不確定時，這支影片的列車都標需確認）
        self.bg_note_file = None
        self.guard = CameraGuard(profile)   # 攝影機位置基準（和列車背景分開，錄影中斷後也不重建，#75、#76）
        self.camera_stop = None  # 偵測到攝影機位置改變：dict(t=畫面時間, file=影片, pos=影片內秒數)，判讀就此停止（#71）
        self.small = self.bg_small = self.out_mask = None   # 整個畫面縮小版（估計攝影機自動調亮度）

    def feed(self, t_abs: float, gray: np.ndarray, sat: float, file: str, pos: float, frame_bgr=None):
        g = gray.astype(np.float32)
        self._frame, self._sat = frame_bgr, sat
        if self.first_t is None:
            self.first_t = t_abs
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g.copy()
        gain = self._exposure(frame_bgr)
        if gain != 1.0:
            g = g / gain        # 攝影機自動調亮度（例如亮的列車經過後整個畫面變暗）→ 先還原再比較
        if self.camera_stop or self._camera_check(t_abs, g, gain, file, pos):
            return
        diff = np.abs(g - self.bg)
        if self.prev is not None and self.prev.shape == g.shape:
            fd = np.abs(g - self.prev)
            tx0, ty0, tx1, ty1 = self.tr
            # 有沒有在動：參考線上的平均變化，或軌道範圍內明顯變化的像素比例（車身顏色均勻時參考線上看不出在動）
            moving = (float(fd[self.ref_mask].mean()) > self.p.static_motion
                      or float((fd[ty0:ty1, tx0:tx1] > 10).mean()) > 0.004)
        else:
            moving = True
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
        # 白天：亮度差很多、但紋理和背景幾乎一樣（地上的東西都還在原位，只是被陽光照亮或雲遮暗）→ 不算有車。
        # 夜間（紅外線黑白畫面）列車被車燈照白時紋理相似度也很高，所以只在白天用。
        light = False
        if sat >= self.p.day_saturation and score > self.p.off_level:
            u = g[self.band]; v = self.bg[self.band]
            u = u - u.mean(); v = v - v.mean()
            ncc = float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-6))
            light = ncc > self.p.light_ncc
        if light:
            on = occupied = False
            if self.light_since is None:
                self.light_since = t_abs
        else:
            self.light_since = None
        if self.cur is None:
            if on:
                at_start = self.first_t is not None and t_abs - self.first_t < 2.0
                self.cur = dict(at_file_start=self.start_kind if at_start else '', start=t_abs, start_file=file, start_pos=pos,
                                end=t_abs, end_file=file, end_pos=pos,
                                peak=score, peak_t=t_abs, peak_file=file, peak_pos=pos, coverage=cov, sat=[sat], rec=list(self.hist), rec_tail=[],
                                last_motion=t_abs, last_motion_file=file, last_motion_pos=pos,
                                scores=[(t_abs, score, file, pos)], tail=[(t_abs, score, file, pos)],
                                static_now=0.0, static_max=0.0, static_alarm=False, incomplete='')
        else:
            c = self.cur
            if t_abs - c['start'] <= self.HEAD_SEC:
                c['rec'].append((t_abs, active))
            else:
                c['rec_tail'].append((t_abs, active))
                while c['rec_tail'] and t_abs - c['rec_tail'][0][0] > self.TAIL_SEC:
                    c['rec_tail'].pop(0)
            if moving:
                c['last_motion'], c['last_motion_file'], c['last_motion_pos'] = t_abs, file, pos
            c['static_now'] = t_abs - c['last_motion']
            if occupied:
                c['static_max'] = max(c['static_max'], c['static_now'])
            if t_abs - c['start'] < 15 and len(c['scores']) < 400:
                c['scores'].append((t_abs, score, file, pos))
            c['tail'].append((t_abs, score, file, pos))
            while t_abs - c['tail'][0][0] > 15:
                c['tail'].pop(0)
            if occupied and t_abs - c['last_motion'] > self.p.max_static:
                # 列車停多久都不結束（使用者 2026-10-02 決定，#67）：超過 max_static 只標需人工確認，事件繼續，
                # 等車真的開走、確認離開才結束。攝影機被碰歪另外由 _camera_check 處理
                c['static_alarm'] = True
            if occupied:
                self.pending_close = None
                c['end'], c['end_file'], c['end_pos'] = t_abs, file, pos
                if score > c['peak']:
                    c['peak'], c['peak_t'], c['peak_file'], c['peak_pos'] = score, t_abs, file, pos
                c['coverage'] = max(c['coverage'], cov)
                c['both'] = max(c.get('both', 0.0), both)
                c['sat'].append(sat)
            else:
                if self.pending_close is None:
                    self.pending_close = t_abs
                if t_abs - c['end'] > max(self.p.merge_gap, self.SIDE_WINDOW):
                    self._close()
        # 背景更新：沒有事件時一律學習（畫面某處永久改變時才不會一直卡在「有點不一樣」）；
        # 事件進行中（含列車停住的時候）完全凍結，停著的列車不會被學成背景。光線變化交給亮度修正（_exposure）處理。
        if self.cur is not None:
            return
        if self.light_since is not None and t_abs - self.light_since >= 1.0:
            # 光線變化持續 1 秒：直接把現在的畫面當新背景（不然要好幾分鐘才學得完，期間來的列車會判不準）
            self.bg = g * gain
            if self.small is not None:
                self.bg_small = self.small.copy()
            self.light_since = None
            return
        # 沒有事件時：和背景差很多的像素（例如列車已經進入軌道範圍、還沒碰到參考線）只用很慢的速度學，
        # 慢速列車才不會在碰到參考線之前就被學進背景；其他像素照常學（光線、天色變化）
        raw = g * gain
        fast = diff < self.p.pixel_diff
        self.bg = np.where(fast, self.bg * 0.98 + raw * 0.02, self.bg * 0.999 + raw * 0.001)
        a = 0.02
        if self.small is not None:
            if self.bg_small is None or self.bg_small.shape != self.small.shape:
                self.bg_small = self.small.copy()
            else:
                self.bg_small = self.bg_small * (1 - a) + self.small * a

    EXPO_SCALE = 8
    def _camera_check(self, t_abs, g, gain, file, pos):
        """攝影機位置改變（被碰歪、轉動）：交給 CameraGuard 判斷（和列車背景分開，#75～#77）。
        確認後：進行中的那一筆在改變的那一刻結束（車尾未確認；改變那一刻才開始的，就不是列車），
        然後**停止這次判讀**（self.camera_stop）：參考線、軌道範圍是固定的像素位置，攝影機動了就不再對準原本的
        真實位置，不可以重建背景後繼續用（#71）。回傳 True＝已偵測到，process() 會停止"""
        hit = self.guard.check(t_abs, self._frame, self._sat, self.cur is not None, file, pos)
        if not hit:
            return False
        t0, f0, p0 = hit
        if self.cur is not None:
            c = self.cur
            if c['end'] > t0:
                c['end'], c['end_file'], c['end_pos'] = t0, f0, p0
            c['camera_start'] = c['start'] >= t0 - 0.3          # 畫面改變那一刻才開始的：不是列車
            c['camera_t'] = t0
            self.pending_close = None
            self.finish('camera')
        self.camera_stop = dict(t=t0, file=f0, pos=p0, kind='moved', result='moved')
        return True

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

    def finish(self, why='video_end', note=''):
        """影片全部處理完（video_end）、使用者取消（cancel）或錄影中斷（gap）時收尾。
        這時參考線上還有車（不是已經離開、正在等確認），車尾時間就不是真的車尾離開，標「車尾未確認」（#45、#49）"""
        if self.cur is not None:
            # 正在等確認（車尾看起來離開了，但還沒等滿 max(merge_gap, SIDE_WINDOW) 秒，可能只是車廂空隙）
            # 也不算已確認（#66）
            self.cur['incomplete'] = why
            self.cur['incomplete_note'] = note
            self.cur['confirming'] = self.pending_close is not None
            self._close()
        return self.events

    def restart(self, bg, note='', kind='gap'):
        """錄影中斷（gap）後重新開始：背景、上一格、最近的歷史都不延續（#49）"""
        self.bg = bg
        self.bg_small = None
        self.prev = None
        self.hist = []
        self.light_since = None
        self.pending_close = None
        self.first_t = None
        self.start_kind = kind
        self.bg_note = note

    def _onset(self, c):
        """回傳 (車頭時間差, 車尾時間差)：參考線後側（右／下）中位數減前側（左／上）中位數，>0 表示往右／往下"""
        rec = [(t, a) for (t, a) in c['rec'] + c.get('rec_tail', []) if c['start'] - self.SIDE_WINDOW <= t <= c['end'] + self.SIDE_WINDOW]
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
        if c.get('incomplete'):
            return s_shift, e_shift          # 車尾未確認（影片結束、取消、錄影中斷）：保留最後看到列車的時間
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
                  night=night, valid=True, reason='', need_check=False, end_known=True,
                  peak_t=c.get('peak_t'), peak_file=c.get('peak_file'), peak_pos=c.get('peak_pos'),
                  start_shift=round(shift, 2), end_shift=round(eshift, 2), check_items=[])
        reasons = []
        items = ev['check_items']             # 品質摘要用的分類（和備註同時寫，用詞一致）
        if dur < self.p.min_duration:
            ev['valid'] = False
            reasons.append('時間太短（%.1f 秒，可能是閃爍或畫面亮度跳動）' % dur)
        elif c['coverage'] < self.p.min_coverage:
            ev['valid'] = False
            reasons.append('軌道範圍內變化範圍太小（%.0f%%，可能是汽車、行人或樹葉）' % (c['coverage'] * 100))
        elif c.get('both', 0.0) < self.p.min_both:
            ev['valid'] = False
            reasons.append('只有參考線一側有變化，沒有東西橫跨參考線（可能是車燈照射或路上的車）')
        if c.get('incomplete'):
            ev['end_known'] = False
        ev['start_known'] = not c.get('at_file_start')
        if c.get('camera_start'):
            ev['valid'] = False
            reasons.insert(0, '攝影機位置在 %s 改變（可能被碰到或轉動），這一筆是畫面改變造成的，不是列車；判讀在這裡停止'
                           % fmt_time(c['camera_t']))
        if ev['valid']:
            if not agree:
                ev['need_check'] = True
                reasons.append('方向不確定'); items.append('方向不確定')
            wait = max(self.p.merge_gap, self.SIDE_WINDOW)
            if c.get('confirming') and c.get('incomplete') in ('video_end', 'cancel', 'gap'):
                ev['need_check'] = True
                when = {'video_end': '影片就結束了', 'cancel': '就按了取消',
                        'gap': '錄影就中斷了%s' % c.get('incomplete_note', '')}[c['incomplete']]
                reasons.append('車尾未確認：車尾看起來在 %s 離開，但還沒等滿確認時間（%g 秒，用來排除車廂之間的空隙）%s'
                               % (fmt_time(c['end']), wait, when))
                items.append('車尾離開後確認時間不足（車尾未確認）')
            elif c.get('incomplete') == 'camera':
                ev['need_check'] = True
                reasons.append('車尾未確認：攝影機位置在 %s 改變（可能被碰到或轉動），車尾時間無法確認；判讀在這裡停止'
                               % fmt_time(c['camera_t']))
                items.append('攝影機位置改變（車尾未確認）')
            elif c.get('incomplete') == 'video_end':
                ev['need_check'] = True
                reasons.append('車尾未確認：影片結束時列車仍在參考線上（車尾時間是影片最後一格 %s）' % fmt_time(c['end']))
                items.append('影片結束時列車仍在參考線上（車尾未確認）')
            elif c.get('incomplete') == 'cancel':
                ev['need_check'] = True
                reasons.append('車尾未確認：判讀被取消時列車仍在參考線上（車尾時間是取消前最後一格 %s）' % fmt_time(c['end']))
                items.append('取消時列車仍在參考線上（車尾未確認）')
            elif c.get('incomplete') == 'gap':
                ev['need_check'] = True
                reasons.append('車尾未確認：錄影在 %s 中斷%s，當時列車仍在參考線上' % (fmt_time(c['end']), c.get('incomplete_note', '')))
                items.append('錄影中斷時列車仍在參考線上（車尾未確認）')
            if c.get('static_alarm'):
                ev['need_check'] = True
                reasons.append('參考線上曾經超過 %s 完全沒有動靜（可能列車長時間停駛，或攝影機／畫面異常），已計入通過時間，請確認'
                               % fmt_span(self.p.max_static))
                items.append('長時間完全沒有動靜')
            elif c.get('static_max', 0) >= self.p.static_sec:
                reasons.append('列車曾在參考線上停止約 %d 秒（已計入通過時間）' % round(c['static_max']))
            if dur > self.p.long_event:
                ev['need_check'] = True
                reasons.append('佔用時間很長（%d 分鐘，可能停車或畫面變化）' % round(dur / 60)); items.append('佔用時間很長')
            if shift > 0:
                reasons.append('參考線在 %s 就開始有微小變化（可能是車燈先照到），車頭時間取變化明顯的時刻（往後 %.1f 秒）'
                               % (fmt_time(first_change), shift))
            if max(shift, eshift) > REFINE_CHECK:
                ev['need_check'] = True
                items.append('車頭／車尾自動修正超過 %d 秒' % REFINE_CHECK)
            if eshift > 0:
                reasons.append('參考線到 %s 還有微小變化（可能是車尾後光線或背景改變），車尾時間取變化明顯的時刻（往前 %.1f 秒）'
                               % (fmt_time(last_change), eshift))
            if c.get('at_file_start') == 'gap':
                ev['need_check'] = True
                reasons.append('錄影中斷後，下一支影片一開始就有變化（列車可能在錄影恢復前就已到達，車頭時間是影片開頭）')
                items.append('錄影中斷後一開始就有列車（車頭未確認）')
            elif c.get('at_file_start'):
                ev['need_check'] = True
                reasons.append('影片一開始就有變化（列車可能在影片開始前就已到達，車頭時間是影片開頭）')
                items.append('影片一開始就有列車（車頭未確認）')

            if self.bg_note and self.bg_note_file in (c['start_file'], c['end_file']):
                ev['need_check'] = True
                reasons.append(self.bg_note); items.append('影片開頭的背景不確定')
            if night:
                reasons.append('夜間')
        ev['reason'] = '；'.join(reasons)
        ev['last_change'] = last_change
        self.events.append(ev)


def _ncc(u, v):
    """兩組像素的正規化相關（紋理相似度，-1～1；和整體亮度、對比的倍率無關）"""
    u = u - u.mean(); v = v - v.mean()
    return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-6))


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
            progress: Callable[[float, str], None] = None, cancel: Callable[[], bool] = None, info: dict = None):
    """依序處理多個檔案（同一攝影機）。bases：每個檔案「影片內 0 秒」對應的畫面時間（epoch 秒，由 osd.calibrate 算出）。
    回傳事件 list（dict）。info（dict）：偵測到攝影機位置改變時填入 info['camera']＝dict(t, file, pos)，
    判讀在那裡停止，已完成的事件照常回傳（#71）"""
    det = None
    durs = []
    for f in files:
        vi = video_info(f)
        durs.append((vi or {}).get('dur') or 600)
    total = sum(durs) or 1
    done = 0.0
    prev_end = None                       # 上一支影片最後一格的畫面時間（實際讀到的，不用檔頭的長度）
    for fi, (f, base) in enumerate(zip(files, bases)):
        cap = cv2.VideoCapture(f)
        if not cap.isOpened():
            raise RuntimeError('無法開啟影片：' + os.path.basename(f))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop = work_crop(profile, W, H)
        if det is None:
            det = Detector(profile, crop)
            batch_t0 = base
            det.bg, det.bg_note = initial_background(f, crop, profile)   # 開頭若剛好有列車經過，不會被當成背景
            det.bg_note_file = f
        elif prev_end is not None and base < prev_end - OVERLAP_SEC:
            # 實際讀完上一支後，下一支的時間比它還早：時間倒退或重疊，不可以把時間往回餵給判讀（#69）
            raise TimeOrderError('依實際影片時間確認，「%s」的開始時間 %s 比上一支「%s」實際的結尾 %s 還早 %.1f 秒'
                                 '（時間重疊或倒退）。請檢查影片是否重複加入、順序或畫面時間是否正確。'
                                 % (os.path.basename(f), fmt_time(base), os.path.basename(files[fi - 1]),
                                    fmt_time(prev_end), prev_end - base))
        elif prev_end is not None and base - prev_end > GAP_SEC:
            # 錄影中斷（缺檔）：前後不是連續畫面。進行中的那一筆在中斷處結束（車尾未確認），背景重新建立（#49）
            gap = base - prev_end
            det.finish('gap', '（下一支影片 %s 才開始，中間缺 %s）' % (fmt_time(base), fmt_span(gap)))
            # 攝影機位置：中斷前後要確認一致才可以繼續用原本的參考線（#76）
            res = det.guard.gap_check(f)
            if res != 'same':
                cap.release()
                if info is not None:
                    shift, r = det.guard.last_result or (0.0, 0.0)
                    info['camera'] = dict(t=base, file=f, pos=0.0, kind='gap', result=res, prev_end=prev_end,
                                          shift=round(shift, 1), r=round(r, 2))
                if info is not None and det.guard.notes:
                    info['notes'] = list(det.guard.notes)
                return det.events
            bg, note = initial_background(f, crop, profile)
            det.restart(bg, note)
            det.bg_note_file = f
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
            if det.camera_stop:                    # 攝影機位置改變：停止判讀（#71）
                cap.release()
                evs = det.finish('camera')
                _mark_initial_window(evs, det, batch_t0)
                if info is not None:
                    info['camera'] = dict(det.camera_stop)
                    if det.guard.notes:
                        info['notes'] = list(det.guard.notes)
                return evs
            if progress and k % 150 == 0:
                progress(min(1.0, (done + pos) / total), '%s（%d/%d）' % (os.path.basename(f), fi + 1, len(files)))
            if cancel and k % 30 == 0 and cancel():
                cap.release()
                return det.finish('cancel')
        prev_end = base + last_pos + 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15) if k else base
        cap.release()
        done += durs[fi]
    if det and info is not None and det.guard.notes:
        info['notes'] = list(det.guard.notes)
    return det.finish('video_end') if det else []


def _mark_initial_window(evs, det, batch_t0, window=90.0):
    """攝影機在批次開頭 window 秒內移動：初始列車背景（開頭 90 秒中位數）混到了移動後的畫面，
    這段時間的判讀不可靠（#75）。從批次一開始就有、一直到攝影機改變才結束的那一筆＝背景錯位造成的，不是列車；
    其他在改變前開始的標需確認"""
    t0 = det.camera_stop['t']
    if t0 - batch_t0 > window:
        return
    for e in evs:
        if e['start'] >= t0 or not e.get('valid'):
            continue
        if e['start'] - batch_t0 < 2.0 and e['end'] >= t0 - 0.5:
            e['valid'] = False
            e['reason'] = ('攝影機在影片開頭 %d 秒內移動，初始背景混到移動後的畫面，這一筆是背景錯位造成的，不是列車；' % window
                           + e.get('reason', '')).rstrip('；')
        else:
            e['need_check'] = True
            e['reason'] = (e.get('reason', '') + '；攝影機在影片開頭 %d 秒內移動，初始背景可能混到移動後的畫面，這一筆的時間可能不準'
                           % window).strip('；')
            e.setdefault('check_items', []).append('攝影機在開頭移動（時間可能不準）')


GAP_SAME_R = 0.3   # 錄影中斷前後比對：r 至少這麼高（而且平移 < 3 像素）才算確認位置一致
GAP_SEC = 5.0       # 前一支影片結束到下一支開始超過幾秒，就當成錄影中斷（和開始判讀前的「缺檔」檢查一致）
OVERLAP_SEC = 2.0   # 下一支影片比上一支實際結尾早超過幾秒，就是時間重疊／倒退（和開始判讀前的「重疊」檢查一致）


class TimeOrderError(RuntimeError):
    """判讀途中發現影片時間倒退或重疊（#69）"""


def fmt_span(sec):
    sec = int(round(sec))
    if sec >= 3600:
        return '%d 小時 %d 分' % (sec // 3600, sec % 3600 // 60)
    if sec >= 60:
        return '%d 分 %d 秒' % (sec // 60, sec % 60)
    return '%d 秒' % sec


def initial_background(path, crop, profile: Profile = None, n=31, seconds=90.0):
    """初始背景。回傳 (背景, 說明)。
    ① 開頭 seconds 秒內平均取 n 格的中位數（影片開頭有列車「經過」也不會被學進去）。
       不直接用整支影片：傍晚、清晨光線變化大，整支的中位數和開頭的畫面差很多，會造成一開始就誤判有車（修正清單 #32）。
    ② 列車在開頭「停著」超過一半時間時，①會把列車學成背景（修正清單 #56）。另外取整支影片的中位數，
       亮度調成和①一樣後比較參考線上的差異：差異超過「佔用開始門檻」＝有東西在參考線上停很久，但分不出是
       「開頭停著、後來開走」還是「後來開進來、一直停到影片結束」（兩種情形畫面是對稱的）。
       所以照樣用①，回傳說明，這支影片的列車都標需人工確認（寧可請人確認，也不要默默給錯的時間）。
    先試跳轉（快）；跳轉不可靠的影片（實際每秒格數和檔頭不同）改成從頭依序讀。"""
    info = video_info(path) or {}
    total = info.get('dur') or seconds
    head = _median_frames(path, crop, [min(seconds, total) * (k + 0.5) / n for k in range(n)])
    if head is None or profile is None or total < seconds * 1.5:
        return head, ''
    whole = _median_frames(path, crop, [total * (k + 0.5) / 41 for k in range(41)])
    if whole is None:
        return head, ''
    det = Detector(profile, crop)
    ref = det.ref_mask
    r = float(np.median(head / np.maximum(whole, 1.0)))
    r = r if 0.5 < r < 2.0 else 1.0
    adj = whole * r
    if float(np.abs(head - adj)[ref].mean()) > profile.on_level:
        return head, BG_UNSURE
    return head, ''


REFINE_CHECK = 5.0   # 車頭／車尾自動修正（_refine）超過幾秒就標需確認。使用者可接受的誤差是 5 秒（#57；實際樣本最大 4.1 秒）
BG_UNSURE = '這支影片開頭的畫面和整支影片不一樣（可能有列車在影片開頭停著、或停了很久），背景不確定，列車時間可能不準'



def _median_frames(path, crop, targets):
    """在 targets（影片內秒數）各取一格，回傳中位數（灰階、模糊、只取工作範圍）"""
    x0, y0, x1, y1 = crop
    prep = lambda fr: cv2.GaussianBlur(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    frames = []
    cap = cv2.VideoCapture(path)
    try:
        ok_seek = path not in _BAD_SEEK
        for t in (targets if ok_seek else []):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            p = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok or abs(p - t) > 2.0:
                ok_seek = False
                break
            frames.append(prep(fr))
        if not ok_seek:
            _BAD_SEEK.add(path)
            cap.release()
            cap = cv2.VideoCapture(path)
            frames, i = [], 0
            while i < len(targets) and cap.grab():
                if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= targets[i]:
                    ok, fr = cap.retrieve()
                    if ok:
                        frames.append(prep(fr))
                    i += 1
    finally:
        cap.release()
    if len(frames) < 3:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.float32)


# ---------------------------------------------------------------- 截圖與短片
# 注意：有些攝影機實際每秒格數和檔頭寫的不同（例：檔頭 15、實際 10）。這種檔案 OpenCV 的「跳到某秒」會跳錯
# （實測跳 100 秒落在 246 秒），跳到後段甚至讀不到畫面。所以截圖與短片一律「每支影片從頭依序讀一次」，
# 用每一格實際的時間決定要不要存，不使用跳轉。

LONG_CLIP = 60.0      # 超過幾秒的事件，短片只做車頭前後、車尾前後兩段（使用者 2026-10-02 決定，#59）
CLIP_PART = 10.0      # 兩段各取車頭之後／車尾之前幾秒


def _timeline(files, bases, events):
    """每支影片的 (畫面時間起點, 長度)。有 bases 就用（判讀時的時間對照）；沒有就從事件反推（舊的呼叫方式）"""
    tl = {}
    if bases:
        for f, b in zip(files, bases):
            if b is not None:
                tl[f] = (b, (video_info(f) or {}).get('dur') or 0.0)
        return tl
    for ev in events:
        for f, t, p in ((ev['start_file'], ev['start'], ev['start_pos']), (ev['end_file'], ev['end'], ev['end_pos'])):
            if f not in tl:
                tl[f] = (t - p, (video_info(f) or {}).get('dur') or 0.0)
    return tl


def _at(tl, order, t):
    """畫面時間 → (影片, 影片內秒數)；落在缺口或範圍外回傳 None"""
    for f in order:
        if f in tl:
            b, d = tl[f]
            if b - 1e-3 <= t <= b + d + 1e-3:
                return f, max(0.0, t - b)
    return None


def save_frames_and_clips(events, profile: Profile, out_dir: str, progress=None, cancel=None, files=None, bases=None):
    """每筆事件存三張截圖，「是列車」的另外擷取短片。
    截圖：列車＝車頭／中間／車尾；已排除＝車頭／變化最大／車尾（變化最大那一刻最看得出為什麼被排除，#60）。
    短片（只有列車，#60）：事件前後各 clip_before／clip_after 秒；事件超過 LONG_CLIP 秒時只做
    「車頭前 clip_before～車頭後 CLIP_PART 秒」＋「車尾前 CLIP_PART 秒～車尾後 clip_after」兩段接在一起（#59）。
    跨幾支影片都照畫面時間切成每支影片的一段，依序接起來（v1.0.8 以前只接開始與結束兩支，中間的會漏掉，#54）。
    files／bases：影片處理順序與每支的時間起點（判讀時的時間對照）。"""
    shot_dir = os.path.join(out_dir, '截圖'); os.makedirs(shot_dir, exist_ok=True)
    clip_dir = os.path.join(out_dir, '短片')
    if profile.make_clips:
        os.makedirs(clip_dir, exist_ok=True)
    order = list(files or [])
    for ev in events:
        for f in (ev['start_file'], ev['end_file']):
            if f not in order:
                order.append(f)
    tl = _timeline(list(files or []), bases, events) if bases else {}
    for f, v in _timeline([], None, events).items():    # 沒給 bases，或事件裡的影片不在 files 裡 → 從事件反推
        tl.setdefault(f, v)
    shots = {f: [] for f in order}       # 檔案 → [(秒數, 事件, 標籤)]
    clips = {f: [] for f in order}       # 檔案 → [(開始秒, 結束秒, 事件, 第幾段)]
    last_file = {}                       # 事件 id → 短片最後一段所在的檔案
    for ev in events:
        # 檔名前綴第一次產生後就固定（media_tag）：之後改成列車、編號重排再重做，也不會蓋到別筆的截圖與短片
        ev['tag'] = ev.get('media_tag') or ('%03d' % ev['no'] if ev.get('valid') else 'X%03d' % ev['no'])
        ev['media_tag'] = ev['tag']
        ev['shots'], ev['clip'] = [], ''
        # 車頭／車尾未確認時，截圖名稱不寫「車頭／車尾」，避免人工覆核時誤以為是確定的時間（#70）
        shots[ev['start_file']].append((ev['start_pos'], ev, '1車頭' if ev.get('start_known', True) else '1第一畫面_車頭未確認'))
        if ev.get('valid') or not ev.get('peak_file'):
            m = _at(tl, order, (ev['start'] + ev['end']) / 2)
            if m is None:                 # 中間落在缺口：用車頭那支影片
                m = (ev['start_file'], ev['start_pos'] + 1.0)
            shots[m[0]].append((m[1], ev, '2中間'))
        else:
            shots[ev['peak_file']].append((ev['peak_pos'], ev, '2變化最大'))
        shots[ev['end_file']].append((ev['end_pos'], ev, '3車尾' if ev.get('end_known', True) else '3最後畫面_車尾未確認'))
        if profile.make_clips and ev.get('valid'):
            a, b = ev['start'] - profile.clip_before, ev['end'] + profile.clip_after
            spans = [(a, b)] if ev['end'] - ev['start'] <= LONG_CLIP else \
                [(a, ev['start'] + CLIP_PART), (ev['end'] - CLIP_PART, b)]
            seg = 0
            for sa, sb in spans:
                for f in order:
                    if f not in tl:
                        continue
                    fb, fd = tl[f]
                    lo, hi = max(sa, fb), min(sb, fb + (fd if fd > 0 else 1e9))
                    if hi > lo:
                        clips.setdefault(f, []).append((lo - fb, hi - fb, ev, seg))
                        last_file[id(ev)] = f
                        seg += 1
    writers = {}                          # 事件 id → 短片寫入狀態
    todo = [f for f in order if shots.get(f) or clips.get(f)]
    for k, f in enumerate(todo):
        if cancel and cancel():
            break
        _one_pass(f, sorted(shots.get(f, []), key=lambda x: x[0]), clips.get(f, []), profile, shot_dir, clip_dir, writers)
        for ev_id in [e for e, lf in last_file.items() if lf == f and e in writers]:
            _close_clip(writers.pop(ev_id), clip_dir)
        if progress:
            progress((k + 1) / max(1, len(todo)), '產生截圖與短片 %d/%d' % (k + 1, len(todo)))
    for w in list(writers.values()):      # 取消時收尾
        _close_clip(w, clip_dir)
    for ev in events:
        ev['shots'].sort()
        ev.pop('tag', None)


def _one_pass(path, shot_reqs, clip_reqs, profile, shot_dir, clip_dir, writers):
    OUT_FPS = 15.0
    need_until = max([p for p, _e, _l in shot_reqs] + [b for _a, b, _e, _s in clip_reqs] + [0.0])
    cap = cv2.VideoCapture(path)
    si = 0
    last = None                           # 上一格（截圖的位置剛好在影片最後時用）
    try:
        while True:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            want_shot = si < len(shot_reqs) and pos >= shot_reqs[si][0] - 1e-3
            active = [c for c in clip_reqs if c[0] - 1e-3 <= pos <= c[1]]
            if not want_shot and not active:
                if pos > need_until:
                    break
                continue
            ok, fr = cap.retrieve()
            if not ok:
                continue
            img = draw_overlay(fr, profile)
            last = img
            while si < len(shot_reqs) and pos >= shot_reqs[si][0] - 1e-3:
                _p, ev, lab = shot_reqs[si]
                name = '%s_%s.jpg' % (ev['tag'], lab)
                imwrite(os.path.join(shot_dir, name), img)
                ev['shots'].append(name)
                si += 1
            for a, b, ev, seg in active:
                w = writers.get(id(ev))
                if w is None:
                    h, wd = img.shape[:2]
                    tmp = os.path.join(clip_dir, '_tmp_%s.mp4' % ev['tag'])
                    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), OUT_FPS, (wd, h))
                    w = writers[id(ev)] = dict(vw=vw, tmp=tmp, ev=ev, t_out=0.0, seg=None, t0=0.0, base=0.0)
                if w['seg'] != (path, seg):          # 新的一段（跨檔案時接在後面）
                    w['seg'], w['t0'], w['base'] = (path, seg), pos, w['t_out']
                target = w['base'] + (pos - w['t0'])
                while w['t_out'] <= target + 1e-6:  # 依實際時間補格／跳格，播放速度才正確
                    w['vw'].write(img)
                    w['t_out'] += 1.0 / OUT_FPS
    finally:
        cap.release()
    while si < len(shot_reqs) and last is not None:  # 要的位置超過影片最後一格：用最後一格
        _p, ev, lab = shot_reqs[si]
        name = '%s_%s.jpg' % (ev['tag'], lab)
        imwrite(os.path.join(shot_dir, name), last)
        ev['shots'].append(name)
        si += 1


def _close_clip(w, clip_dir):
    ev = w['ev']
    try:
        w['vw'].release()
        name = '%s.mp4' % ev['tag']
        dst = os.path.join(clip_dir, name)
        if os.path.exists(w['tmp']) and os.path.getsize(w['tmp']) > 0:
            if os.path.exists(dst):
                os.remove(dst)
            os.replace(w['tmp'], dst)        # VideoWriter 不支援中文路徑 → 先寫英文暫存檔名再改名
            ev['clip'] = name
    except Exception as e:  # 短片失敗不影響其他結果
        ev['reason'] = (ev.get('reason', '') + '；短片產生失敗：' + str(e)).strip('；')


def imwrite(path, img):
    """cv2.imwrite 不支援中文路徑（Windows）→ 用 imencode"""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        with open(path, 'wb') as fh:
            fh.write(buf.tobytes())


_BAD_SEEK = set()


def open_at(path, t, near=3.0):
    """開啟影片並停在 t 秒之前不遠（下一次 read() 讀到的畫面 ≤ t）。
    先試一次跳轉，讀一格檢查實際時間；跳錯（每秒格數和檔頭不符的影片）就從頭依序讀過去。"""
    cap = cv2.VideoCapture(path)
    if t > near:
        if path not in _BAD_SEEK:
            cap.set(cv2.CAP_PROP_POS_MSEC, (t - near / 2) * 1000)
            ok = cap.grab()
            p = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ok and t - near <= p <= t:
                return cap
            _BAD_SEEK.add(path)           # 這支跳轉不可靠，之後直接依序讀
            cap.release()
            cap = cv2.VideoCapture(path)
        while True:                       # 依序讀（只解碼不轉圖，比較快）
            if not cap.grab():
                break
            if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= t - near:
                break
    return cap


def jump_near(path, t, cancel=None, near=4.0, tries=6):
    """跳轉不可靠的影片（檔頭每秒格數和實際不同）：每次用新開的影片試跳，
    依「要求的位置 → 實際落點」的比例修正下一次的要求，通常 2～3 次就落在 t 之前 near 秒內。
    回傳停在 t 之前的影片（接著依序讀到 t）；都不成功時回傳落點在 t 之前最接近的一次（最差是從頭）。"""
    best_cap, best_p = cv2.VideoCapture(path), 0.0
    x, ratio = t - near / 2, 1.0
    for _ in range(tries):
        if cancel and cancel():
            break
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, x) * 1000)
        ok = cap.grab()
        p = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ok and p <= t and p > best_p:
            best_cap.release()
            best_cap, best_p = cap, p
            if t - p <= near:
                return best_cap
        else:
            cap.release()
        if ok and p > 0 and x > 0:
            ratio = p / x
        else:
            ratio *= 1.5                 # 讀不到（跳太後面）：要求往前一點
        x = (t - near / 2) / ratio
    return best_cap


class PreviewReader:
    """預覽用：記住目前開著的影片與位置。往後看（+1 秒、往右拖）時接著讀，不用每次從頭；
    往回看或換影片才重新開。cancel() 讓讀到一半的工作停下來（使用者又拖了別的位置）。
    回傳 (frame, 實際秒數) 或 (None, 0.0)。progress(已讀秒數, 目標秒數) 給畫面顯示「讀取中」。"""

    def __init__(self):
        self.cap = None
        self.path = None
        self.pos = -1.0          # 下一次 grab 之前，上一格的時間

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.path, self.pos = None, None, -1.0

    def read(self, path, t, cancel=None, progress=None):
        if self.cap is None or path != self.path or t < self.pos - 0.05 or (path not in _BAD_SEEK and t - self.pos > 6):
            self.close()
            if t <= 3.0:
                self.cap = cv2.VideoCapture(path)
            elif path in _BAD_SEEK:
                self.cap = jump_near(path, t, cancel)
            else:
                self.cap = open_at(path, t)   # 跳轉可靠的影片：直接跳（若跳錯會被記成 _BAD_SEEK 並從頭讀）
            self.path, self.pos = path, -1.0
        cap = self.cap
        n = 0
        while True:
            if not cap.grab():
                self.close()
                return None, 0.0
            p = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.pos = p
            if p >= t - 0.05:
                ok, fr = cap.retrieve()
                return (fr, p) if ok else (None, 0.0)
            n += 1
            if n % 30 == 0:
                if cancel and cancel():
                    return None, 0.0
                if progress:
                    progress(p, t)


def read_frame_at(path, t):
    """讀取影片 t 秒的那一格（預覽用）；回傳 (frame, 實際秒數)"""
    cap = open_at(path, t)
    try:
        best = None
        while True:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if pos >= t - 0.05:
                ok, fr = cap.retrieve()
                if ok:
                    best = (fr, pos)
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
