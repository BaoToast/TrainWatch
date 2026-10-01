"""
讀取畫面上的時間字幕（OSD），例如「2026-01-01 08:00:00」。

做法：字幕是固定位置、等寬字型。依「時間字幕位置」框把字幕切成一格一格，每一格和 0～9 的字樣比對（相關係數）。
  - 內建字樣是從這台攝影機的畫面學來的（白字黑邊）。換了不同牌子的攝影機，可在程式裡用「教程式認字」重新學一次：
    輸入預覽畫面上看到的時間，程式往後讀 13 秒，自動學到 0～9。
  - 一支影片讀很多格（開頭 20 秒每一格＋整支平均取 30 個點）投票，偶爾一格被樹葉、車燈干擾也不會讀錯。
  - 秒數跳號的那一格可以算出小於 1 秒的對齊，時間精度大約是 1 格（0.07～0.1 秒）。
  - 不使用檔名上的時間（檔名可能和畫面差 1 秒以上）。
"""
from __future__ import annotations

import base64
import collections
import datetime as dt

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

CELL_W, CELL_H = 10, 16
DEFAULT_FORMAT = 'YYYY-MM-DD HH:MM:SS'
DEFAULT_RECT = [14, 8, 204, 24]       # 時間字幕外框 [x0,y0,x1,y1]（這台攝影機 640×360）
MIN_MARGIN = 0.03                     # 每一格最佳與次佳字樣的相關係數差，低於此值這一格不採用

_B64 = (
    'AAAAAAAAAAAAAAAAAP///////wAAAP////////8AAAD///8AAP///wD///8AAAD///8A////AAAAAP//AP///wAAAAD//2z/'
    '//8AAAAA//9s////AAAAAP//AP///wAAAAD//wD+//8AAAAA//8AAP///wAA////AAD///8A/////wAA/////////wAAAAAA'
    '/////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//AAAAAAAAAP///wAAAAAAAP////8AAAAAAP//////AAAAAAD//wD/'
    '/wAAAAAAAAAA//8AAAC6AAAAAP//AAAAugAAAAD//wAAAAAAAAAA//8AAAAAAAAAAP//AAAAAAAAAAD//wAAAAAAAAAA//8A'
    'AAAAAAAAAP//AAAAAAAAAAD//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////AAAA//////////8A/////wAA////'
    'AP///wAAAP///wAAAAAAAAD///8AAAAAAAAA////AAEAAAAA////AAAAAAAA/////wAAAAAAAP///wAAAAAAAP///wAAAAAA'
    'AP///wAAAAAAAP////8AAAAAAAD//////////wD///////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//////wAAAP//'
    '////////AAD///8AAP///wD///8AAAD///8AAAAAAAAA////AAAAAAD//////wAAAAD//////wAAAAAAAP//////AAAAAAAA'
    'AP///wAAAP8AAAAA//8A////AAAA////AAD///8AAP///wAA//////////8AAAD+/////wAAAAAAAAAAAAAAAAAAAAAAAAAA'
    'AAAAAAAAAAAA//8AAAAAAAAA////AAAAAAAA/////wAAAAAA//////8AAAAAAP//////AAAAAP///wD//wAAAP////8A//8A'
    'AAD///8AAP//AAD///8AAAD//wAA////////////AP///////////wAAAAAAAAD//wAAAAAAAAAA//8AAAAAAAAAAP//AAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAA/////////wAA//////////8AAP///wAAAAAAAAD///8AAAAAAAAA////////AAAA////'
    '//////8AAP///////////wD///8AAAD///8AAAAAAAAAAP//AAD/AAAAAAD//wD///8AAAD///8A////AAD/////AAD/////'
    '////AAAAAP//////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/////wAAAAAAAP///wAAAAAAAP///wAAAAAAAAD///8A'
    'AAAAAAD+//8AAAAAAAD//////////wAA//////////8A/////wAAAP//AP///wAAAAD//wD///8AAAAA//8A////AAAAAP//'
    'AAD///8AAP///wAA//////////8AAAD///////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////AP///////////wAA'
    'AAAAAAD///8AAAAAAAAA////AAAAAAAA+P//AAAAAAAAAP///wAAAAAAAP///wAAAAAAAAD///8AAAAAAAAA//8AAAAAAAAA'
    '////AAAAAAAAAP///wAAAAAAAAD//wAAAAAAAAD///8AAAAAAAAA////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////'
    '////AAAA//////////8AAP///wAA////AP///wAAAP///wAA//8AAAD///8AAP//////////AAD/////////AAAA////////'
    '//8A////AAAA////Hv///wAAAAD//x7///8AAAAA//8A/////wAA////AAD//////////wAAAP///////wAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAP7//////QAAAP//////////AP////8AAP///wD///8AAAD///8A////AAAA////AP///wAAAP///wD/'
    '////AP////8AAP////////0AAAAA/v/////9AAcAAAAA////AAAHAAAAAP///wAAAAAAAP///wAAAAAAAAD///8AAAAAAAD/'
    '//8AAAAAAAAAAAAAAAAAAA=='
)


def default_templates():
    q = np.frombuffer(base64.b64decode(_B64), np.uint8).reshape(10, CELL_H, CELL_W)
    return q.astype(np.float32) / 255.0


def templates_to_list(T):
    return np.round(np.asarray(T) * 255).astype(int).tolist()


def templates_from_list(lst):
    if not lst:
        return default_templates()
    return np.asarray(lst, np.float32).reshape(10, CELL_H, CELL_W) / 255.0


# ---------------------------------------------------------------- 格式
def tokens(fmt):
    """回傳 {'Y':(起,長), 'mo':…, 'D':…, 'H':…, 'mi':…, 'S':…}；HH 之前的 MM 是月、之後的是分"""
    hpos = fmt.find('HH')
    if hpos < 0 or 'SS' not in fmt or 'DD' not in fmt:
        raise ValueError('格式要有 DD、HH、MM、SS，例如 YYYY-MM-DD HH:MM:SS')
    t = {}
    if 'YYYY' in fmt:
        t['Y'] = (fmt.find('YYYY'), 4)
    elif 'YY' in fmt:
        t['Y'] = (fmt.find('YY'), 2)
    else:
        raise ValueError('格式要有 YYYY 或 YY')
    mo = fmt.find('MM')
    if not (0 <= mo < hpos):
        raise ValueError('月份 MM 要在 HH 前面')
    mi = fmt.find('MM', hpos)
    if mi < 0:
        raise ValueError('分鐘 MM 要在 HH 後面')
    t.update(mo=(mo, 2), D=(fmt.find('DD'), 2), H=(hpos, 2), mi=(mi, 2), S=(fmt.find('SS'), 2))
    return t


def digit_positions(fmt):
    return [i for i, ch in enumerate(fmt) if ch in 'YMDHS']


def parse(text, fmt):
    """依格式取出日期時間；不合理回傳 None"""
    try:
        t = tokens(fmt)
        g = {k: int(text[a:a + n]) for k, (a, n) in t.items()}
        y = g['Y'] + (2000 if t['Y'][1] == 2 else 0)
        return dt.datetime(y, g['mo'], g['D'], g['H'], g['mi'], g['S'])
    except (ValueError, TypeError):
        return None


def render(d, fmt):
    """datetime → 依格式排好的字串（非數字位置保留格式字元）"""
    t = tokens(fmt)
    out = list(fmt)
    vals = dict(Y='%04d' % d.year, mo='%02d' % d.month, D='%02d' % d.day, H='%02d' % d.hour,
                mi='%02d' % d.minute, S='%02d' % d.second)
    for k, (a, n) in t.items():
        out[a:a + n] = list(vals[k][-n:])
    return ''.join(out)


# ---------------------------------------------------------------- 讀字
def _feat(c):
    return np.clip((c - 150.0) / 80.0, 0, 1)


def cells(frame, rect, fmt):
    """把字幕切成 len(fmt) 格，每格縮放成 10×16"""
    x0, y0, x1, y1 = rect
    n = len(fmt)
    pitch = (x1 - x0) / float(n)
    h = float(y1 - y0)
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    sx, sy = pitch / CELL_W, h / CELL_H
    out = []
    for i in range(n):
        M = np.float32([[sx, 0, x0 + i * pitch], [0, sy, y0]])
        out.append(_feat(cv2.warpAffine(g, M, (CELL_W, CELL_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)))
    return out


def _norm(a):
    a = a - a.mean()
    n = np.sqrt((a * a).sum())
    return a / n if n > 1e-6 else a


def read_text(frame, rect, fmt, T):
    """回傳 (字串, 最小信心差)"""
    cs = cells(frame, rect, fmt)
    Tn = np.stack([_norm(t).ravel() for t in T])
    out, worst = [], 1.0
    for i, ch in enumerate(fmt):
        if ch not in 'YMDHS':
            out.append(ch)
            continue
        sc = Tn @ _norm(cs[i]).ravel()
        o = np.argsort(-sc)
        out.append(str(int(o[0])))
        worst = min(worst, float(sc[o[0]] - sc[o[1]]))
    return ''.join(out), worst


def read_datetime(frame, rect, fmt, T):
    txt, w = read_text(frame, rect, fmt, T)
    d = parse(txt, fmt)
    return (d if w >= MIN_MARGIN else None), txt, w


# ---------------------------------------------------------------- 一支影片的時間對照
def calibrate(path, rect, fmt=DEFAULT_FORMAT, T=None, dense_seconds=20.0, n_spread=30, cancel=None):
    """
    算出這支影片「影片內秒數 → 畫面時間」：畫面時間（epoch 秒）＝ 影片內秒數 ＋ offset。
    回傳 dict(ok, offset, start, support, samples, jumps, msg)
    """
    if T is None:
        T = default_templates()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return dict(ok=False, msg='無法開啟影片', samples=0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n / fps if n > 0 else 600
    samples = []          # (影片內秒數, datetime)
    tries = 0

    def take(fr, pos):
        d, _txt, _w = read_datetime(fr, rect, fmt, T)
        if d is not None:
            samples.append((pos, d))
    try:
        while True:                          # 1) 開頭每一格
            ok, fr = cap.read()
            if not ok:
                break
            tries += 1
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            take(fr, pos)
            if pos > dense_seconds:
                break
        for k in range(1, n_spread + 1):     # 2) 整支平均取樣
            if cancel and cancel():
                break
            t = dur * k / (n_spread + 1)
            if t <= dense_seconds:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            for _ in range(3):
                ok, fr = cap.read()
                if not ok:
                    break
                tries += 1
                take(fr, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    if len(samples) < 5:
        return dict(ok=False, samples=len(samples),
                    msg='讀不到畫面上的時間字幕（請確認「時間字幕位置」框，或用「教程式認字」）')
    raw = np.array([d.timestamp() - p for p, d in samples])
    fl = np.floor(raw).astype(np.int64)
    cnt = collections.Counter(fl.tolist())
    best = max(cnt, key=lambda k: cnt[k] + cnt.get(k - 1, 0))     # 相位會讓樣本分在相鄰兩個整數
    good = raw[(fl >= best - 1) & (fl <= best)]
    support = len(good) / len(raw)
    # 畫面時間 = floor(影片秒數 + 相位) + 基準 → (畫面秒 − 影片秒) 的上緣就是精確的 offset
    off = float(np.percentile(good, 97))
    res = raw - off
    jumps = int(np.sum((res < -1.25) | (res > 0.25)))
    ok = support >= 0.8
    msg = ''
    if not ok:
        msg = '畫面時間讀取不穩定（一致率 %d%%），請確認' % round(support * 100)
    elif jumps > max(3, 0.05 * len(samples)):
        msg = '影片中的畫面時間有跳動（%d 個取樣對不上），請確認' % jumps
    return dict(ok=ok, offset=off, start=dt.datetime.fromtimestamp(off), support=round(support, 3),
                samples=len(samples), tries=tries, jumps=jumps, msg=msg)


def learn(path, t0, shown, rect, fmt=DEFAULT_FORMAT, seconds=13.0):
    """
    「教程式認字」：t0（影片內秒數）那一格畫面上顯示的時間是 shown（datetime）。
    程式往後讀，最後一位秒數每變一次就加一秒當答案，收集 0～9 的字樣。回傳 (templates, 缺少的數字)
    """
    from .core import open_at
    cap = open_at(path, t0)
    last_s = tokens(fmt)['S'][0] + 1
    acc = collections.defaultdict(list)
    prev, sec = None, 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos < t0 - 0.02:
                continue
            if pos > t0 + seconds:
                break
            cs = cells(fr, rect, fmt)
            f = cs[last_s]
            if prev is not None and np.abs(f - prev).mean() > 0.08:
                sec += 1
            prev = f
            txt = render(shown + dt.timedelta(seconds=sec), fmt)
            for i, ch in enumerate(fmt):
                if ch in 'YMDHS':
                    acc[txt[i]].append(cs[i])
    finally:
        cap.release()
    base = default_templates()
    T = np.stack([np.mean(acc[str(d)], axis=0) if acc.get(str(d)) else base[d] for d in range(10)])
    missing = [d for d in range(10) if not acc.get(str(d))]
    return T, missing
