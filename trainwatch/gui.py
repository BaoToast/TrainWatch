"""列車通過判讀：視窗介面（tkinter）"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

from PIL import Image, ImageTk

from . import core, export, osd
from .core import Profile

VERSION = '1.0.11'
APP = '列車通過判讀'
VIDEO_TYPES = [('影片', '*.mkv *.mp4 *.avi *.mov *.ts *.h264 *.264 *.dav'), ('所有檔案', '*.*')]

CANVAS_W, CANVAS_H = 900, 506      # 第 1 頁影片畫面的「建議」大小；實際大小跟著視窗縮放
# 每一頁內容的最小尺寸：視窗比這個小時出現捲軸，內容不會被切掉（v1.0.7）
TAB_MIN = {1: (0, 0), 2: (0, 0)}       # 0＝依內容實際需要的大小
MAX_FILES = 300            # 一次最多加入幾支影片
MAX_HOURS = 48.0           # 一次最多判讀幾小時的影片（總時數；影片長短不一時以這個為準）
SEC_PER_VIDEO_MIN = (1.0, 2.0)   # 預估：每 1 分鐘影片判讀約需幾秒（快的電腦～慢的電腦）

# 進階設定的合理範圍（超出範圍不給存）
PARAM_RANGE = dict(on_level=(3, 100), off_level=(1, 99), pixel_diff=(3, 120), merge_gap=(0.1, 10),
                   min_duration=(0.1, 10), min_coverage=(0.01, 1.0), min_both=(0.0, 1.0), static_sec=(1, 600),
                   max_static=(60, 86400), long_event=(10, 86400), clip_before=(0, 30), clip_after=(0, 30))


def app_dir():
    """程式的最上層資料夾（「啟動列車通過判讀.bat」所在處）。v1.0.7 以前監測站設定存在這裡"""
    home = os.environ.get('TRAINWATCH_HOME')
    if home and os.path.isdir(home):
        return os.path.abspath(home)
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RESULT_FOLDER = '判讀結果'
STATUS_EMPTY = '請先加入影片，再在右邊畫參考線與軌道範圍。'
STATUS_READY = '影片已加入。畫好框線後，按「開始判讀」。'


DATA_FOLDER = '列車通過判讀設定'
PROFILE_FOLDER = '監測站設定'
SETTINGS_FILE = '程式設定.json'
LAST_FILE = '_上次使用.txt'
ASKED_FILE = '_已詢問匯入舊版設定.txt'


def documents_dir():
    """使用者的「文件」資料夾（Windows 會照實際位置，例如被移到 D 槽或 OneDrive）"""
    if sys.platform.startswith('win'):
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and os.path.isdir(buf.value):
                return buf.value
        except Exception:
            pass
    d = os.path.join(os.path.expanduser('~'), 'Documents')
    return d if os.path.isdir(d) else os.path.expanduser('~')


def data_dir():
    """監測站設定與程式設定的固定位置（v1.0.8 起）：文件\\列車通過判讀設定。
    放在程式資料夾外面，換新版不用再複製。TRAINWATCH_DATA 可指定別處（測試用）"""
    d = os.environ.get('TRAINWATCH_DATA')
    return os.path.abspath(d) if d else os.path.join(documents_dir(), DATA_FOLDER)


def settings_path():
    """程式設定（目前只有結果存放的預設位置）"""
    return os.path.join(data_dir(), SETTINGS_FILE)


def profile_names_in(folder):
    try:
        return sorted(f[:-5] for f in os.listdir(folder) if f.lower().endswith('.json') and not f.startswith('_'))
    except OSError:
        return []


def find_old_settings(folder):
    """使用者選的舊版資料夾裡，監測站設定在哪裡。可以選程式資料夾，也可以直接選「監測站設定」資料夾。
    回傳 (監測站設定資料夾或 None, 程式設定.json 或 None)"""
    folder = os.path.abspath(folder)
    prof = None
    for d in (os.path.join(folder, PROFILE_FOLDER), folder, os.path.join(folder, DATA_FOLDER, PROFILE_FOLDER)):
        if profile_names_in(d):
            prof = d
            break
    cands = [os.path.join(folder, SETTINGS_FILE), os.path.join(folder, DATA_FOLDER, SETTINGS_FILE)]
    if prof == folder:                      # 直接選了「監測站設定」資料夾 → 程式設定在上一層
        cands.append(os.path.join(os.path.dirname(folder), SETTINGS_FILE))
    st = next((f for f in cands if os.path.isfile(f)), None)
    return prof, st


def import_settings(src_prof, src_settings, dst_root):
    """把舊版的監測站設定（與程式設定）複製到 dst_root。已經有同名的監測站不覆蓋。
    回傳 (複製了的監測站名稱, 因為同名沒有複製的名稱, 程式設定有沒有複製)"""
    dst_prof = os.path.join(dst_root, PROFILE_FOLDER)
    os.makedirs(dst_prof, exist_ok=True)
    copied, skipped = [], []
    had = set(profile_names_in(dst_prof))
    if src_prof and os.path.abspath(src_prof) != os.path.abspath(dst_prof):
        for name in profile_names_in(src_prof):
            if name in had:
                skipped.append(name)
                continue
            shutil.copy2(os.path.join(src_prof, name + '.json'), os.path.join(dst_prof, name + '.json'))
            copied.append(name)
        last = os.path.join(src_prof, LAST_FILE)
        if os.path.isfile(last) and not had:
            shutil.copy2(last, os.path.join(dst_prof, LAST_FILE))
    st_done = False
    dst_st = os.path.join(dst_root, SETTINGS_FILE)
    if src_settings and os.path.isfile(src_settings) and not os.path.exists(dst_st):
        shutil.copy2(src_settings, dst_st)
        st_done = True
    return copied, skipped, st_done


def load_settings():
    for p in (settings_path(), os.path.join(os.path.expanduser('~'), '列車通過判讀_程式設定.json')):
        try:
            with open(p, encoding='utf-8') as fh:
                d = json.load(fh)
            if isinstance(d, dict):
                return d
        except (OSError, ValueError):
            pass
    return {}


def save_settings(d):
    txt = json.dumps(d, ensure_ascii=False, indent=2)
    try:
        os.makedirs(data_dir(), exist_ok=True)
    except OSError:
        pass
    for p in (settings_path(), os.path.join(os.path.expanduser('~'), '列車通過判讀_程式設定.json')):
        try:
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write(txt)
            return True
        except OSError:
            continue
    return False


def result_root(location, first_video):
    """「判讀結果」資料夾的位置。location＝None 表示放在（第一支）影片旁邊；
    使用者選的資料夾本身就叫「判讀結果」時直接用，不再多包一層"""
    base = location or os.path.dirname(os.path.abspath(first_video))
    if os.path.basename(os.path.normpath(base)) == RESULT_FOLDER:
        return base
    return os.path.join(base, RESULT_FOLDER)


class ScrollArea(ttk.Frame):
    """可捲動的分頁容器：視窗夠大時內容跟著視窗伸縮；比 min_w×min_h 小時，內容維持最小尺寸並出現捲軸"""
    def __init__(self, master, min_w, min_h):
        super().__init__(master)
        self.min_w, self.min_h = min_w, min_h
        bg = ttk.Style(self).lookup('TFrame', 'background') or '#dcdad5'
        self.cv = tk.Canvas(self, highlightthickness=0, bd=0, bg=bg)
        self.vs = ttk.Scrollbar(self, orient='vertical', command=self.cv.yview)
        self.hs = ttk.Scrollbar(self, orient='horizontal', command=self.cv.xview)
        self.cv.configure(yscrollcommand=self.vs.set, xscrollcommand=self.hs.set)
        self.cv.grid(row=0, column=0, sticky='nsew')
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)
        self.inner = ttk.Frame(self.cv)
        self._win = self.cv.create_window(0, 0, window=self.inner, anchor='nw')
        self.cv.bind('<Configure>', lambda e: self._fit())
        self._last = None
        self.after(400, self._watch)

    def _watch(self):
        try:
            self._fit()
            self.after(400, self._watch)
        except tk.TclError:
            pass

    def _fit(self):
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        if cw <= 1:
            return
        mw = max(self.min_w, self.inner.winfo_reqwidth())
        mh = max(self.min_h, self.inner.winfo_reqheight())
        key = (cw, ch, mw, mh)
        if key == self._last:
            return
        self._last = key
        w, h = max(cw, mw), max(ch, mh)
        self.cv.itemconfigure(self._win, width=w, height=h)
        self.cv.configure(scrollregion=(0, 0, w, h))
        for sb, need, kw in ((self.vs, ch < mh, dict(row=0, column=1, sticky='ns')),
                             (self.hs, cw < mw, dict(row=1, column=0, sticky='ew'))):
            if need:
                sb.grid(**kw)
            else:
                sb.grid_remove()
        if ch >= mh:
            self.cv.yview_moveto(0)
        if cw >= mw:
            self.cv.xview_moveto(0)

    def wheel(self, units, horizontal=False):
        if horizontal:
            if self.hs.winfo_ismapped():
                self.cv.xview_scroll(units, 'units')
        elif self.vs.winfo_ismapped():
            self.cv.yview_scroll(units, 'units')


def final_order(files, timing):
    """正式讀完畫面時間後：依時間排序（判讀一定要照時間順序）、找出重疊的影片。
    回傳 (files, timing, 排序說明, 重疊清單)"""
    idx = list(range(len(files)))
    srt = sorted(idx, key=lambda i: (timing[i].get('offset') is None, timing[i].get('offset') or 0))
    note = ''
    if srt != idx:
        note = '正式讀取畫面時間後，影片順序和清單不同，\n已依畫面時間排序後判讀。'
    files = [files[i] for i in srt]
    timing = [timing[i] for i in srt]
    overlap = []
    for k in range(1, len(files)):
        a, b = timing[k - 1].get('offset'), timing[k].get('offset')
        if a is None or b is None:
            continue
        dur = (core.video_info(files[k - 1]) or {}).get('dur') or 0.0
        if b < a + dur - 2.0:
            overlap.append('%s 和 %s 重疊約 %d 秒' % (os.path.basename(files[k - 1]), os.path.basename(files[k]), a + dur - b))
    return files, timing, note, overlap


def camera_stop_text(cam):
    """攝影機位置改變而停止判讀的說明（品質摘要、Excel「設定與影片」、results.json 都用這一段）"""
    return ('偵測到攝影機位置／角度改變。\n影片：%s（影片內約%s）\n畫面時間：約%s\n'
            '判讀在這裡停止，之後的影片沒有判讀。\n'
            '原本參考線與軌道範圍可能已失效，請重新確認監測站設定後，\n從這個時間之後的影片開始判讀。'
            % (os.path.basename(cam['file']), export.fmt_pos(cam['pos']), core.fmt_time(cam['t'], True)))


def mark_time_jumps(evs, files, timing):
    """正式校正發現畫面時間中途跳動的影片：這支影片裡的列車標需人工確認（#52）"""
    jumpy = {f: t.get('msg') for f, t in zip(files, timing) if t.get('msg')}
    idx = {f: i for i, f in enumerate(files)}
    for e in evs:
        a, b = idx.get(e.get('start_file')), idx.get(e.get('end_file'))
        passed = files[a:b + 1] if a is not None and b is not None else [e.get('start_file'), e.get('end_file')]
        msg = next((jumpy[f] for f in passed if f in jumpy), None)      # 事件經過的每一支影片都要看（#68）
        if msg and e.get('valid'):
            e['need_check'] = True
            e['reason'] = (e.get('reason', '') + '；這支影片的畫面時間有跳動（%s），時間可能不準' % msg).strip('；')
            e.setdefault('check_items', []).append('影片的畫面時間有跳動')


def quality_summary(evs, cancelled=False, timing=()):
    """判讀完成的品質摘要文字（#61）"""
    valid = [e for e in evs if e.get('valid')]
    need = [e for e in valid if e.get('need_check')]
    stop = next((t.get('stop') for t in timing if t.get('stop')), None)
    lines = (['判讀中途停止。' + stop, '', '停止前已完成的部分：'] if stop else
             ['%s判讀完成。' % ('已取消，部分' if cancelled else '')]) + [
             '列車 %d 筆、已排除 %d 筆。' % (len(valid), len(evs) - len(valid)),
             '需人工確認 %d 筆%s' % (len(need), '，其中：' if need else '。')]
    cnt = {}
    for e in need:
        for it in (e.get('check_items') or ['其他（請看備註）']):
            cnt[it] = cnt.get(it, 0) + 1
    for it, n in sorted(cnt.items(), key=lambda x: -x[1]):
        lines.append('　・%s：%d 筆' % (it, n))
    no_end = sum(1 for e in valid if e.get('end_known') is False)
    if no_end:
        lines.append('')
        lines.append('注意：%d筆「車尾未確認」，離開時間是程式最後看到列車的時間，\n匯入噪音分析程式前請先確認。' % no_end)
    notes = sorted({t.get('note') for t in timing if t.get('note')})
    if notes:
        lines.append('')
        lines.extend(notes)
    if need:
        lines.append('')
        lines.append('需人工確認的在第2頁以淡黃色標示，原因寫在「備註」欄。\n一筆可能同時有好幾個原因。')
    return '\n'.join(lines)


def open_path(p):
    try:
        if sys.platform.startswith('win'):
            os.startfile(p)  # noqa
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', p])
        else:
            subprocess.Popen(['xdg-open', p])
    except Exception as e:
        messagebox.showerror(APP, '無法開啟：%s\n%s' % (p, e))


def parse_hms(text):
    """'08:15:30' / '08:15:30.5' / '081530' → 當天第幾秒"""
    t = text.strip().replace('：', ':')
    if not t:
        raise ValueError('空白')
    if ':' not in t and t.replace('.', '').isdigit() and len(t.split('.')[0]) == 6:
        t = t[:2] + ':' + t[2:4] + ':' + t[4:]
    parts = t.split(':')
    if len(parts) != 3:
        raise ValueError('格式請用 時:分:秒，例如 08:15:30.5')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s < 60):
        raise ValueError('時間超出範圍')
    return h * 3600 + m * 60 + s


def parse_datetime(text):
    """'2026-01-01 08:00:00' 或 '2026/01/01 08:00:00.5' → datetime"""
    t = text.strip().replace('/', '-').replace('：', ':')
    parts = t.split()
    if len(parts) != 2:
        raise ValueError('格式請用 年-月-日 時:分:秒，例如 2026-01-01 08:00:00')
    d = dt.datetime.strptime(parts[0], '%Y-%m-%d')
    return d + dt.timedelta(seconds=parse_hms(parts[1]))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('%s v%s' % (APP, VERSION))
        self._fonts()
        self.profile = Profile()
        self.files = []            # [{'path','info','osd': dict|None,'manual': datetime|None,'state': str}]
        self.events = []
        self.result_dir = ''
        self.run_files, self.run_timing = [], []
        self.run_profile = None
        self.dirty = False          # 第 2 頁有修改還沒儲存
        self._pv_token = 0          # 預覽：每次要求一個新畫面就加一，舊的讀取看到號碼變了就停下來
        self._pv_shown = None
        self.q = queue.Queue()
        self.cancel_flag = False
        self.worker = None
        self.scanner = None
        self.scan_queue = []
        self.frame_img = None
        self.cur_frame = None
        self.cur_pos = 0.0
        self.disp_scale = 1.0
        self.drag = None
        self._seek_job = None
        self.mode = tk.StringVar(value='ref')
        self.show_excluded = tk.BooleanVar(value=False)
        self.data_root = data_dir()
        self.prof_dir = os.path.join(self.data_root, PROFILE_FOLDER)
        try:
            os.makedirs(self.prof_dir, exist_ok=True)
        except OSError:      # 「文件」不能寫入 → 改存到使用者資料夾
            self.data_root = os.path.join(os.path.expanduser('~'), DATA_FOLDER)
            self.prof_dir = os.path.join(self.data_root, PROFILE_FOLDER)
            os.makedirs(self.prof_dir, exist_ok=True)
        self.migrated = self._migrate_from_app_dir()
        self.settings = load_settings()
        d = self.settings.get('result_location')
        self.out_default = d if isinstance(d, str) and d else None   # None＝影片旁邊
        self.out_location = self.out_default                            # 這次要用的位置

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=6, pady=6)
        self.page1 = ScrollArea(self.nb, *TAB_MIN[1])
        self.page2 = ScrollArea(self.nb, *TAB_MIN[2])
        self.tab1, self.tab2 = self.page1.inner, self.page2.inner
        self.nb.add(self.page1, text='  1. 影片與參考線  ')
        self.nb.add(self.page2, text='  2. 檢視與修改結果  ')
        for seq in ('<MouseWheel>', '<Shift-MouseWheel>', '<Button-4>', '<Button-5>'):
            root.bind_all(seq, self._on_wheel, add='+')
        self._build_tab1()
        self.update_count()
        self._build_tab2()
        self._load_last_profile()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        if os.environ.get('CI') != 'true':      # GitHub Actions 的自動測試不要跳出詢問視窗
            root.after(500, self._first_run_check)
        root.after(150, self._poll)

    def _on_wheel(self, e):
        """滑鼠滾輪捲動整頁（只有視窗太小、出現捲軸時）。在清單上滾動仍是捲清單"""
        try:
            if isinstance(e.widget, (ttk.Treeview, tk.Listbox, ttk.Combobox, ttk.Spinbox, ttk.Scale)):
                return
            page = self.page1 if self.nb.select() == str(self.page1) else self.page2
            if not str(e.widget).startswith(str(page)):
                return
        except (tk.TclError, AttributeError, KeyError):
            return
        if getattr(e, 'num', None) in (4, 5):
            units = -1 if e.num == 4 else 1
        else:
            units = -1 if e.delta > 0 else 1
        page.wheel(units * 3, horizontal=bool(e.state & 0x1))

    # ------------------------------------------------------------ 外觀
    def _fonts(self):
        fams = set(tkfont.families(self.root))
        fam = next((f for f in ('Microsoft JhengHei UI', 'Microsoft JhengHei', 'Noto Sans CJK TC', 'Noto Sans TC', 'PingFang TC')
                    if f in fams), None)
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
            try:
                fnt = tkfont.nametofont(name)
                if fam:
                    fnt.configure(family=fam)
                fnt.configure(size=10)
            except tk.TclError:
                pass
        self.font_family = fam or 'TkDefaultFont'
        st = ttk.Style(self.root)
        if 'vista' in st.theme_names():
            st.theme_use('vista')
        elif 'clam' in st.theme_names():
            st.theme_use('clam')
        st.configure('Treeview', rowheight=24)
        st.configure('Big.TButton', font=(self.font_family, 11, 'bold'))
        st.configure('Hint.TLabel', foreground='#555555')
        st.configure('Warn.TLabel', foreground='#b00000')
        st.configure('Ok.TLabel', foreground='#006400')

    # ------------------------------------------------------------ 分頁 1
    def _build_tab1(self):
        t = self.tab1
        left = ttk.Frame(t)
        left.pack(side='left', fill='y', padx=(4, 8), pady=4)
        ttk.Frame(left, width=360, height=1).pack(side='bottom')   # 撐住寬度 360（不用 pack_propagate，高度才算得出來）
        right = ttk.Frame(t)
        right.pack(side='left', fill='both', expand=True, pady=4)

        box = ttk.LabelFrame(left, text='監測站設定')
        box.pack(fill='x', pady=(0, 6))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Label(r, text='名稱').pack(side='left')
        self.var_name = tk.StringVar(value=self.profile.name)
        ttk.Entry(r, textvariable=self.var_name).pack(side='left', fill='x', expand=True, padx=(6, 0))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(2, 6))
        self.cb_prof = ttk.Combobox(r, state='readonly', width=14, values=self._profile_names())
        self.cb_prof.pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='載入', width=5, command=self.load_profile).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='儲存', width=5, command=self.save_profile).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='進階…', width=6, command=self.advanced).pack(side='left', padx=(4, 0))

        box = vbox = ttk.LabelFrame(left, text='影片（同一台攝影機）')
        box.pack(fill='both', expand=True, pady=(0, 6))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Button(r, text='加入影片…', command=self.add_files).pack(side='left')
        ttk.Button(r, text='移除', width=5, command=self.remove_files).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='清空', width=5, command=self.clear_files).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='依時間排序', command=self.sort_files).pack(side='left', padx=(4, 0))
        fr = ttk.Frame(box); fr.pack(fill='both', expand=True, padx=6, pady=2)
        self.tv_files = ttk.Treeview(fr, columns=('name', 'start'), show='headings', selectmode='extended', height=2)
        self.tv_files.heading('name', text='檔名'); self.tv_files.heading('start', text='畫面上的開始時間')
        self.tv_files.column('name', width=150, anchor='w'); self.tv_files.column('start', width=160, anchor='center')
        sb = ttk.Scrollbar(fr, orient='vertical', command=self.tv_files.yview)
        self.tv_files.configure(yscrollcommand=sb.set)
        self.tv_files.pack(side='left', fill='both', expand=True); sb.pack(side='left', fill='y')
        self.tv_files.tag_configure('bad', foreground='#b00000')
        self.tv_files.tag_configure('manual', foreground='#0050a0')
        self.tv_files.bind('<<TreeviewSelect>>', lambda e: self.on_file_select())
        self.var_count = tk.StringVar(value='')
        self.lbl_count = ttk.Label(box, textvariable=self.var_count, foreground='#1060c0', wraplength=330, justify='left')
        self.lbl_count.pack(fill='x', padx=6)
        hint = ttk.Label(box, style='Hint.TLabel', wraplength=330, justify='left', text=(
            '開始時間讀畫面上的字幕（不看檔名）。紅字＝讀不到，請選取該檔在下面填；藍字＝手動填的。'))
        hint.pack(fill='x', padx=6)
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(4, 6))
        lr = ttk.Frame(r); lr.pack(side='top', fill='x')
        ttk.Label(lr, text='手動開始時間').pack(side='left')
        ttk.Label(lr, text='例：2026-01-01 08:00:00', style='Hint.TLabel').pack(side='left', padx=(8, 0))
        ttk.Button(r, text='清除', width=5, command=self.clear_file_time).pack(side='right')
        ttk.Button(r, text='套用', width=5, command=self.set_file_time).pack(side='right', padx=(4, 4))
        self.var_ftime = tk.StringVar()
        ttk.Entry(r, textvariable=self.var_ftime).pack(side='left', fill='x', expand=True)
        # 清單下面這幾行先排（由下往上），視窗矮的時候縮的是影片清單，不會把手動時間那一行擠掉
        for w in (r, hint, self.lbl_count):
            w.pack_configure(side='bottom', before=fr)

        box = ttk.LabelFrame(left, text='判讀')
        box.pack(fill='x', side='bottom', before=vbox)   # 先排，視窗小時縮的是影片清單，不是判讀按鈕與狀態
        self.var_outloc = tk.StringVar()
        ttk.Label(box, textvariable=self.var_outloc, foreground='#1060c0', wraplength=330, justify='left').pack(fill='x', padx=6, pady=(4, 0))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(2, 0))
        ttk.Button(r, text='選擇存放位置…', command=self.choose_out_location).pack(side='left')
        ttk.Button(r, text='改回影片旁邊', command=self.reset_out_location).pack(side='left', padx=(4, 0))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(6, 0))
        self.btn_cancel = ttk.Button(r, text='取消', width=6, command=self.cancel_run, state='disabled')
        self.btn_cancel.pack(side='right', fill='y', padx=(4, 0))
        self.btn_run = ttk.Button(r, text='開始判讀', style='Big.TButton', command=self.start_run)
        self.btn_run.pack(side='left', fill='x', expand=True)
        self.update_out_location()
        self.pb = ttk.Progressbar(box, maximum=1000)
        self.pb.pack(fill='x', padx=6, pady=(6, 2))
        self.var_status = tk.StringVar(value=STATUS_EMPTY)
        ttk.Label(box, textvariable=self.var_status, wraplength=330, style='Hint.TLabel').pack(fill='x', padx=6, pady=(0, 8))

        r = ttk.Frame(right); r.pack(fill='x')
        ttk.Label(r, text='在畫面上拖曳：').pack(side='left')
        ttk.Radiobutton(r, text='參考線（紅）', value='ref', variable=self.mode).pack(side='left', padx=3)
        ttk.Radiobutton(r, text='軌道範圍（橘）', value='track', variable=self.mode).pack(side='left', padx=3)
        ttk.Radiobutton(r, text='時間字幕位置（藍）', value='osd', variable=self.mode).pack(side='left', padx=3)

        self.cw, self.ch = CANVAS_W, CANVAS_H
        self.canvas = tk.Canvas(right, width=480, height=270, bg='#202020', highlightthickness=0, cursor='crosshair')
        self.canvas.bind('<Configure>', self._on_canvas_size)
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        # 畫面下面的列先排（由下往上），影片畫面拿剩下的空間，視窗縮小時縮的是畫面
        below = ttk.Frame(right); below.pack(side='bottom', fill='x')
        self.canvas.pack(side='top', fill='both', expand=True, pady=(6, 4), anchor='nw')
        right = below
        r = ttk.Frame(right); r.pack(fill='x')
        self.var_pos = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(r, from_=0, to=600, variable=self.var_pos, command=lambda v: self.schedule_seek())
        self.scale.pack(side='left', fill='x', expand=True)
        self.var_postxt = tk.StringVar(value='影片內 --:--')
        ttk.Label(r, textvariable=self.var_postxt, width=18, anchor='e').pack(side='left', padx=(8, 0))
        r = ttk.Frame(right); r.pack(fill='x', pady=(2, 0))
        for lab, d in (('−10 秒', -10), ('−1 秒', -1), ('+1 秒', 1), ('+10 秒', 10)):
            ttk.Button(r, text=lab, width=7, command=lambda d=d: self.step(d)).pack(side='left', padx=(0, 4))
        ttk.Label(r, text='線寬').pack(side='left', padx=(10, 2))
        self.var_width = tk.IntVar(value=self.profile.ref_width)
        sp = ttk.Spinbox(r, from_=3, to=25, width=4, textvariable=self.var_width, command=self.redraw_overlay)
        sp.pack(side='left')
        sp.bind('<KeyRelease>', lambda e: self.redraw_overlay())
        ttk.Button(r, text='教程式認字…', command=self.teach_osd).pack(side='right')
        r = ttk.Frame(right); r.pack(fill='x', pady=(4, 0))     # 時間判讀單獨一行，視窗窄時不會擠掉按鈕
        ttk.Label(r, text='畫面時間判讀：').pack(side='left')
        self.var_osd = tk.StringVar(value='')
        self.lbl_osd = ttk.Label(r, textvariable=self.var_osd, style='Ok.TLabel')
        self.lbl_osd.pack(side='left')
        hints = ttk.Frame(right); hints.pack(fill='x', pady=(8, 0))
        self._hint_labels = []
        for title, color, text in (
                ('參考線', '#c00000', '畫一條橫過軌道的短線。車頭碰到線＝到達時間，車尾離開線＝離開時間。不用剛好畫到車身高度，蓋住兩條軌道即可。'),
                ('軌道範圍', '#b06000', '沿著軌道框一段，參考線左右都要留一段。用來分辨列車（佔滿一長段軌道、橫跨參考線）和汽車、行人、車燈，請盡量避開馬路。'),
                ('時間字幕位置', '#1060c0', '框住畫面上的日期時間字幕（第一個字的左邊到最後一個字的右邊）。下方「畫面時間判讀」和畫面上一樣就表示讀得到；不一樣請調整藍框，或按「教程式認字」。')):
            row = ttk.Frame(hints); row.pack(fill='x', pady=1)
            tk.Label(row, text=title, fg=color, font=(self.font_family, 10, 'bold'), width=12, anchor='nw').pack(side='left', anchor='n')
            lb = ttk.Label(row, text=text, style='Hint.TLabel', wraplength=780, justify='left')
            lb.pack(side='left', fill='x')
            self._hint_labels.append(lb)
        hints.bind('<Configure>', lambda e: [lb.configure(wraplength=max(200, e.width - 110)) for lb in self._hint_labels])

    # ------------------------------------------------------------ 分頁 2
    def _build_tab2(self):
        t = self.tab2
        top = ttk.Frame(t); top.pack(fill='x', padx=4, pady=(4, 4))
        ttk.Button(top, text='開啟先前的結果…', command=self.open_results).pack(side='left')
        ttk.Button(top, text='開啟結果資料夾', command=lambda: self.result_dir and open_path(self.result_dir)).pack(side='right')
        ttk.Button(top, text='輸出噪音分析用 TXT', command=self.export_noise).pack(side='right', padx=(6, 6))
        ttk.Button(top, text='儲存修改並重新輸出 Excel', style='Big.TButton', command=self.save_results).pack(side='right', padx=6)

        # 第二行：顯示已排除、筆數、提示訊息（提示太長會自動換行，不會把按鈕擠掉）
        r2 = ttk.Frame(t); r2.pack(fill='x', padx=4, pady=(0, 4))
        ttk.Checkbutton(r2, text='顯示已排除（非列車／閃爍）', variable=self.show_excluded,
                        command=self.fill_events).pack(side='left', anchor='n')
        self.var_sum = tk.StringVar(value='')
        ttk.Label(r2, textvariable=self.var_sum).pack(side='left', padx=(12, 0), anchor='n')
        self.var_tab2msg = tk.StringVar(value='')
        msg = ttk.Label(r2, textvariable=self.var_tab2msg, foreground='#1060c0', justify='left', wraplength=400)
        msg.pack(side='left', fill='x', expand=True, padx=(16, 0), anchor='n')
        msg.bind('<Configure>', lambda e: msg.configure(wraplength=max(150, e.width - 4)))
        body = ttk.Frame(t); body.pack(fill='both', expand=True, padx=4)
        rf = ttk.Frame(body); rf.pack(side='right', fill='y', padx=(8, 0))
        ttk.Frame(rf, width=500, height=1).pack(side='bottom')     # 撐住寬度 500
        lf = ttk.Frame(body, width=420, height=200); lf.pack(side='left', fill='both', expand=True)
        lf.pack_propagate(False)        # 清單最少 420×200，比這個窄就用清單下方的水平捲軸
        cols = ('no', 'date', 'start', 'end', 'dur', 'dir', 'type', 'cross', 'check', 'reason')
        heads = ('編號', '日期', '車頭到達', '車尾離開', '秒數', '方向', '車種', '交會', '需確認', '備註')
        widths = (46, 56, 86, 86, 46, 70, 100, 44, 56, 240)
        self.tv = ttk.Treeview(lf, columns=cols, show='headings', selectmode='browse')
        for c, h, w in zip(cols, heads, widths):
            self.tv.heading(c, text=h)
            self.tv.column(c, width=w, minwidth=w if c == 'reason' else 20,
                           anchor='w' if c in ('reason', 'type') else 'center', stretch=(c == 'reason'))
        hsb = ttk.Scrollbar(lf, orient='horizontal', command=self.tv.xview)
        hsb.pack(side='bottom', fill='x')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set, xscrollcommand=hsb.set)
        self.tv.pack(side='left', fill='both', expand=True); sb.pack(side='left', fill='y')
        self.tv.tag_configure('need', background='#fff2cc')
        self.tv.tag_configure('excl', foreground='#666666')
        self.tv.bind('<<TreeviewSelect>>', lambda e: self.on_event_select())

        self.shot_canvas = tk.Canvas(rf, width=480, height=270, bg='#202020', highlightthickness=0)
        self.shot_canvas.pack(pady=(0, 4))
        r = ttk.Frame(rf); r.pack(fill='x')
        self.shot_btns = []
        for i, lab in enumerate(('車頭', '中間', '車尾')):
            b = ttk.Button(r, text=lab, width=8, command=lambda i=i: self.show_shot(i))
            b.pack(side='left', padx=(0, 4))
            self.shot_btns.append(b)
        ttk.Button(r, text='播放短片', command=self.play_clip).pack(side='left', padx=(8, 0))
        ttk.Button(r, text='在影片中看這段', command=self.goto_event).pack(side='left', padx=(4, 0))
        self.var_shotlab = tk.StringVar(value='')
        ttk.Label(rf, textvariable=self.var_shotlab, style='Hint.TLabel').pack(fill='x', pady=(2, 6))

        ed = ttk.LabelFrame(rf, text='修改這一筆')
        ed.pack(fill='x')
        g = ttk.Frame(ed); g.pack(fill='x', padx=8, pady=6)
        self.var_es = tk.StringVar(); self.var_ee = tk.StringVar()
        self.var_dir = tk.StringVar(); self.var_type = tk.StringVar()
        self.var_cross = tk.BooleanVar(); self.var_checked = tk.BooleanVar(); self.var_note = tk.StringVar()
        ttk.Label(g, text='車頭到達').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(g, textvariable=self.var_es, width=12).grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(g, text='車尾離開').grid(row=0, column=2, sticky='w', padx=(10, 0))
        ttk.Entry(g, textvariable=self.var_ee, width=12).grid(row=0, column=3, sticky='w', padx=4)
        ttk.Label(g, text='方向').grid(row=1, column=0, sticky='w', pady=2)
        ttk.Combobox(g, textvariable=self.var_dir, values=export.DIRS, width=10, state='readonly').grid(row=1, column=1, sticky='w', padx=4)
        ttk.Label(g, text='車種').grid(row=1, column=2, sticky='w', padx=(10, 0))
        ttk.Combobox(g, textvariable=self.var_type, values=export.TYPES, width=16).grid(row=1, column=3, sticky='w', padx=4)
        ttk.Checkbutton(g, text='雙向交會', variable=self.var_cross).grid(row=2, column=1, sticky='w', pady=2)
        ttk.Checkbutton(g, text='已人工確認', variable=self.var_checked).grid(row=2, column=3, sticky='w')
        ttk.Label(g, text='備註').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(g, textvariable=self.var_note, width=40).grid(row=3, column=1, columnspan=3, sticky='we', padx=4)
        r = ttk.Frame(ed); r.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(r, text='套用修改', command=self.apply_edit).pack(side='left')
        self.btn_valid = ttk.Button(r, text='改為「不是列車」', command=self.toggle_valid)
        self.btn_valid.pack(side='left', padx=6)
        for line in ('淡黃色＝需要人工確認（方向不確定、影片一開始就有車等）。',
                     '看過後勾「已人工確認」，再按「套用修改」（時間例：08:15:30.5）。',
                     '修改完記得按上方「儲存修改並重新輸出 Excel」。'):
            ttk.Label(rf, style='Hint.TLabel', text=line).pack(fill='x', pady=(1, 0))

    # ------------------------------------------------------------ 設定的位置與匯入舊版
    def _migrate_from_app_dir(self):
        """舊版把設定放在程式資料夾裡。新位置還沒有設定、程式資料夾裡有（例如使用者照舊複製過來）→ 自動搬過去。
        程式資料夾裡的「監測站設定」之後不再使用，放一個說明檔指向新位置"""
        old_prof = os.path.join(app_dir(), PROFILE_FOLDER)
        res = None
        if not profile_names_in(self.prof_dir) and os.path.abspath(old_prof) != os.path.abspath(self.prof_dir):
            src_st = os.path.join(app_dir(), SETTINGS_FILE)
            if profile_names_in(old_prof) or os.path.isfile(src_st):
                try:
                    res = import_settings(old_prof, src_st, self.data_root)
                except OSError:
                    res = None
        if os.path.isdir(old_prof) and os.path.abspath(old_prof) != os.path.abspath(self.prof_dir):
            try:
                with open(os.path.join(old_prof, '這個資料夾已不使用.txt'), 'w', encoding='utf-8') as fh:
                    fh.write('從 v1.0.8 起，監測站設定改存在：\n%s\n\n換新版時會自動沿用，不用再複製這個資料夾。\n' % self.prof_dir)
            except OSError:
                pass
        return res

    def _first_run_check(self):
        try:
            if self.migrated and (self.migrated[0] or self.migrated[2]):
                self.var_status.set('已把程式資料夾裡的設定搬到「文件\\%s」，以後換新版會自動沿用。' % DATA_FOLDER)
                return
            asked = os.path.join(self.data_root, ASKED_FILE)
            if self._profile_names() or os.path.exists(asked):
                return
            done = False
            while True:
                if not self.confirm('匯入舊版設定',
                                    '找不到監測站設定（參考線、軌道範圍、時間字幕位置等）。\n\n'
                                    '以前用過舊版：請按「選擇舊版資料夾…」，\n'
                                    '選舊版「啟動列車通過判讀.bat」所在的資料夾，\n設定會自動複製過來。\n\n'
                                    '設定現在改存在「文件\\%s」，\n以後換新版會自動沿用，只要匯入這一次。\n\n'
                                    '第一次使用、沒有舊版：請按「重新設定」。' % DATA_FOLDER,
                                    ok='選擇舊版資料夾…', cancel='重新設定'):
                    break
                if self.import_old(ask_again=True):
                    done = True
                    break
            with open(asked, 'w', encoding='utf-8') as fh:
                fh.write('已詢問過（%s）。要再匯入，請按「進階…」裡的「匯入舊版設定…」。\n'
                         % ('已匯入' if done else '選擇重新設定'))
        except (tk.TclError, OSError):
            pass

    def import_old(self, ask_again=False):
        """選舊版資料夾，把監測站設定複製過來。成功回傳 True"""
        d = filedialog.askdirectory(title='選擇舊版的程式資料夾（「啟動列車通過判讀.bat」所在的資料夾）', mustexist=True)
        if not d:
            return False
        prof, st = find_old_settings(d)
        if not prof and not st:
            messagebox.showwarning(APP, '這個資料夾裡找不到監測站設定：\n%s\n\n請選「啟動列車通過判讀.bat」所在的資料夾，'
                                   '或直接選裡面的「%s」資料夾。' % (d, PROFILE_FOLDER))
            return False
        try:
            copied, skipped, st_done = import_settings(prof, st, self.data_root)
        except OSError as e:
            messagebox.showerror(APP, '複製設定失敗：%s' % e)
            return False
        if st_done:
            self.settings = load_settings()
            v = self.settings.get('result_location')
            self.out_default = self.out_location = v if isinstance(v, str) and v else None
            self.update_out_location()
        self.cb_prof.configure(values=self._profile_names())
        if copied:
            self._load_last_profile()
            if not self.cb_prof.get():
                self.load_profile(copied[0])
        msg = []
        if copied:
            msg.append('已匯入 %d 個監測站：%s' % (len(copied), '、'.join(copied)))
        if skipped:
            msg.append('已經有同名的，沒有覆蓋：%s' % '、'.join(skipped))
        if st_done:
            msg.append('已匯入結果存放的預設位置。')
        if not msg:
            msg.append('沒有需要匯入的設定（都已經有了）。')
        messagebox.showinfo(APP, '\n'.join(msg) + '\n\n設定存在：\n%s' % self.data_root)
        return bool(copied or skipped or st_done)

    # ------------------------------------------------------------ 監測站設定
    def _profile_names(self):
        try:
            return profile_names_in(self.prof_dir)
        except OSError:
            return []

    def _collect_profile(self):
        self.profile.name = self.var_name.get().strip() or '未命名監測站'
        try:
            self.profile.ref_width = max(3, min(25, int(self.var_width.get())))
        except (tk.TclError, ValueError):
            pass
        return self.profile

    def _apply_profile_to_ui(self):
        self.var_name.set(self.profile.name)
        self.var_width.set(self.profile.ref_width)
        self.redraw_overlay()
        self.update_osd_label()

    def save_profile(self):
        p = self._collect_profile()
        if self.cur_frame is not None:
            p.frame_size = [int(self.cur_frame.shape[1]), int(self.cur_frame.shape[0])]
        fn = ''.join('_' if c in '\\/:*?"<>|' else c for c in p.name) + '.json'
        path = os.path.join(self.prof_dir, fn)
        if os.path.exists(path) and not messagebox.askyesno(APP, '「%s」已存在，要覆蓋嗎？' % p.name):
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(p.to_json())
        except OSError as e:
            messagebox.showerror(APP, '無法儲存：%s' % e)
            return
        self.cb_prof['values'] = self._profile_names()
        self.cb_prof.set(fn[:-5])
        self._remember(fn[:-5])
        self.var_status.set('已儲存監測站設定：%s' % p.name)

    def load_profile(self, name=None):
        name = name or self.cb_prof.get()
        if not name:
            messagebox.showinfo(APP, '請先在下拉選單選一個監測站。')
            return
        try:
            with open(os.path.join(self.prof_dir, name + '.json'), encoding='utf-8') as fh:
                self.profile = Profile.from_dict(json.load(fh))
        except Exception as e:
            messagebox.showerror(APP, '讀取設定失敗：%s' % e)
            return
        self.cb_prof.set(name)
        self._apply_profile_to_ui()
        self._remember(name)
        self.var_status.set('已載入監測站設定：%s' % self.profile.name)

    def _remember(self, name):
        try:
            with open(os.path.join(self.prof_dir, '_上次使用.txt'), 'w', encoding='utf-8') as fh:
                fh.write(name)
        except OSError:
            pass

    def _load_last_profile(self):
        try:
            with open(os.path.join(self.prof_dir, '_上次使用.txt'), encoding='utf-8') as fh:
                name = fh.read().strip()
            if name in self._profile_names():
                self.load_profile(name)
        except OSError:
            pass

    def advanced(self):
        p = self._collect_profile()
        w = tk.Toplevel(self.root)
        w.title('進階設定')
        w.transient(self.root)
        w.resizable(False, False)
        items = [('on_level', '佔用開始門檻（參考線亮度差）', '數字越小越敏感；誤判閃爍多時調大'),
                 ('off_level', '佔用結束門檻', '需小於開始門檻'),
                 ('pixel_diff', '軌道範圍：像素變化門檻', ''),
                 ('merge_gap', '合併間隔（秒）', '中斷少於此秒數算同一列車'),
                 ('min_duration', '最短秒數', '少於此秒數視為閃爍（列在已排除）'),
                 ('min_coverage', '最小軌道覆蓋率（0～1）', '低於此值視為非列車（汽車、行人）'),
                 ('min_both', '參考線兩側同時被擋的最小比例（0～1）', '低於此值視為非列車（車燈照射）'),
                 ('static_sec', '列車停止幾秒以上要備註（秒）', '只記錄「曾停止」，不會結束這一筆'),
                 ('max_static', '完全不動多久標需確認（秒）', '預設 1800＝30 分鐘；只標記，事件照樣等車開走才結束'),
                 ('long_event', '很長的佔用（秒）', '超過只標記需確認'),
                 ('clip_before', '短片：事件前秒數', ''),
                 ('clip_after', '短片：事件後秒數', '')]
        vars_ = {}
        for i, (k, lab, hint) in enumerate(items):
            ttk.Label(w, text=lab).grid(row=i, column=0, sticky='w', padx=10, pady=3)
            v = tk.StringVar(value=str(getattr(p, k)))
            ttk.Entry(w, textvariable=v, width=8).grid(row=i, column=1, padx=4)
            ttk.Label(w, text=hint, style='Hint.TLabel').grid(row=i, column=2, sticky='w', padx=(4, 10))
            vars_[k] = v
        n = len(items)
        ttk.Label(w, text='時間字幕格式').grid(row=n, column=0, sticky='w', padx=10, pady=3)
        vf = tk.StringVar(value=p.osd_format)
        ttk.Entry(w, textvariable=vf, width=22).grid(row=n, column=1, columnspan=2, sticky='w', padx=4)
        ttk.Label(w, text='Y 年、M 月、D 日、H 時、M 分、S 秒；其他字元照畫面上的寫（例：YYYY/MM/DD HH:MM:SS）',
                  style='Hint.TLabel').grid(row=n + 1, column=0, columnspan=3, sticky='w', padx=10)
        vc = tk.BooleanVar(value=p.make_clips)
        ttk.Checkbutton(w, text='每筆產生短片（會多花一點時間）', variable=vc).grid(row=n + 2, column=0, columnspan=3, sticky='w', padx=10, pady=4)
        vt = tk.BooleanVar(value=False)
        ttk.Checkbutton(w, text='時間字樣恢復內建（取消「教程式認字」學到的）', variable=vt).grid(row=n + 3, column=0, columnspan=3, sticky='w', padx=10)

        def ok():
            try:
                vals = {k: float(v.get()) for k, v in vars_.items()}
            except ValueError:
                messagebox.showerror(APP, '請輸入數字。', parent=w)
                return
            bad = []
            for k, v in vals.items():
                lo, hi = PARAM_RANGE.get(k, (None, None))
                if lo is not None and not (lo <= v <= hi):
                    bad.append('%s：%g～%g' % (dict((i[0], i[1]) for i in items)[k], lo, hi))
            if bad:
                messagebox.showerror(APP, '下列數值超出合理範圍：\n' + '\n'.join(bad), parent=w)
                return
            if vals['off_level'] >= vals['on_level']:
                messagebox.showerror(APP, '結束門檻需小於開始門檻。', parent=w)
                return
            try:
                osd.tokens(vf.get())
            except ValueError as e:
                messagebox.showerror(APP, '時間字幕格式不對：%s' % e, parent=w)
                return
            for k, v in vals.items():
                setattr(p, k, v)
            p.osd_format = vf.get()
            p.make_clips = bool(vc.get())
            if vt.get():
                p.osd_templates = None
            w.destroy()
            self.update_osd_label()

        def reset():
            d = Profile()
            for k, v in vars_.items():
                v.set(str(getattr(d, k)))
            vf.set(d.osd_format)
            vc.set(d.make_clips)
        box = ttk.LabelFrame(w, text='設定存放位置（所有監測站共用）')
        box.grid(row=n + 4, column=0, columnspan=3, sticky='we', padx=10, pady=(10, 0))
        ttk.Label(box, text=self.data_root, foreground='#1060c0').pack(anchor='w', padx=8, pady=(4, 0))
        ttk.Label(box, style='Hint.TLabel', text='換新版會自動沿用。換電腦或要合併其他電腦的設定，請按「匯入舊版設定…」。').pack(anchor='w', padx=8)
        rb = ttk.Frame(box); rb.pack(anchor='w', padx=8, pady=(4, 8))
        def _imp():
            w.grab_release()
            self.import_old()
            try:
                w.grab_set()
            except tk.TclError:
                pass
        ttk.Button(rb, text='匯入舊版設定…', command=_imp).pack(side='left')
        ttk.Button(rb, text='開啟設定資料夾', command=lambda: open_path(self.data_root)).pack(side='left', padx=(6, 0))
        r = ttk.Frame(w); r.grid(row=n + 5, column=0, columnspan=3, pady=(8, 10))
        ttk.Button(r, text='確定', command=ok).pack(side='left', padx=4)
        ttk.Button(r, text='恢復預設', command=reset).pack(side='left', padx=4)
        ttk.Button(r, text='取消', command=w.destroy).pack(side='left', padx=4)
        w.grab_set()

    # ------------------------------------------------------------ 影片清單
    def _templates(self):
        return osd.templates_from_list(self.profile.osd_templates)

    def add_files(self):
        paths = filedialog.askopenfilenames(title='選擇影片（可一次選多個）', filetypes=VIDEO_TYPES)
        if not paths:
            return
        have = {os.path.normcase(f['path']) for f in self.files}
        cand = []
        for p in paths:
            if os.path.normcase(p) not in have and p not in cand:
                cand.append(p)
        room = MAX_FILES - len(self.files)
        if len(cand) > room:
            messagebox.showwarning(APP, '一次最多加入 %d 支影片，目前已有 %d 支。\n這次只加入前 %d 支，其餘 %d 支沒有加入。\n\n影片更多時請分批判讀。'
                                   % (MAX_FILES, len(self.files), max(0, room), len(cand) - max(0, room)))
            cand = cand[:max(0, room)]
        # 影片資訊與畫面時間都在背景讀，加入很多支也不會卡住
        new = [dict(path=p, info=None, osd=None, manual=None, state='讀取畫面時間中…') for p in cand]
        self.files.extend(new)
        self.refresh_files()
        if new and not self.tv_files.selection():
            self.tv_files.selection_set(str(len(self.files) - len(new)))
        for f in new:
            self.scan_queue.append(f)
        self._start_scanner()

    def _start_scanner(self):
        """背景快速讀每支影片開頭的時間字幕（只讀前 3 秒，顯示用；判讀時會再完整校正一次）"""
        if self.scanner and self.scanner.is_alive():
            return
        prof = Profile.from_dict(json.loads(self._collect_profile().to_json()))

        def work():
            T = osd.templates_from_list(prof.osd_templates)
            while self.scan_queue:
                f = self.scan_queue.pop(0)
                if f.get('info') is None:
                    f['info'] = core.video_info(f['path']) or {}
                try:
                    r = osd.calibrate(f['path'], prof.osd_rect, prof.osd_format, T, dense_seconds=3.0, n_spread=0)
                except Exception as e:
                    r = dict(ok=False, msg=str(e))
                self.q.put(('scan', f, r))
            self.q.put(('scan_done',))
        self.scanner = threading.Thread(target=work, daemon=True)
        self.scanner.start()

    def rescan_all(self):
        for f in self.files:
            f['osd'] = None
            f['state'] = '讀取畫面時間中…'
            if f not in self.scan_queue:
                self.scan_queue.append(f)
        self.refresh_files()
        self._start_scanner()

    def _start_of(self, f):
        if f.get('manual') is not None:
            return f['manual']
        r = f.get('osd')
        if r and r.get('ok'):
            return r['start']
        return None

    def sort_files(self):
        self.files.sort(key=lambda f: (self._start_of(f) is None, self._start_of(f) or dt.datetime.max, os.path.basename(f['path'])))
        self.refresh_files()

    def remove_files(self):
        sel = sorted((int(i) for i in self.tv_files.selection()), reverse=True)
        for i in sel:
            f = self.files.pop(i)
            if f in self.scan_queue:
                self.scan_queue.remove(f)
        self.refresh_files()

    def clear_files(self):
        self.files = []
        self.scan_queue = []
        self.refresh_files()

    def set_file_time(self):
        sel = self.tv_files.selection()
        if len(sel) != 1:
            messagebox.showinfo(APP, '請先在清單選一個檔案。')
            return
        try:
            d = parse_datetime(self.var_ftime.get())
        except ValueError as e:
            messagebox.showerror(APP, '時間格式不對：%s' % e)
            return
        i = int(sel[0])
        self.files[i]['manual'] = d
        self.refresh_files()
        self.tv_files.selection_set(str(i))

    def clear_file_time(self):
        sel = self.tv_files.selection()
        for s in sel:
            self.files[int(s)]['manual'] = None
        self.refresh_files()

    def _row(self, f):
        tag = ()
        if f.get('info') == {}:
            txt, tag = '無法開啟這支影片', ('bad',)
        elif f.get('manual') is not None:
            txt = f['manual'].strftime('%Y-%m-%d %H:%M:%S') + '（手動）'
            tag = ('manual',)
        elif f.get('osd') is None:
            txt = f.get('state') or ''
        elif f['osd'].get('ok'):
            txt = core.fmt_time(f['osd']['offset'], True)[:-2]
        else:
            txt = '讀不到，請手動填'
            tag = ('bad',)
        return (os.path.basename(f['path']), txt), tag

    def update_count(self):
        self.update_out_location()
        n = len(self.files)
        mins = sum(((f.get('info') or {}).get('dur') or 600) for f in self.files) / 60.0
        if n == 0:
            self.var_count.set('最多可加入 %d 支、合計 %d 小時的影片。' % (MAX_FILES, MAX_HOURS))
            return
        lo, hi = (mins * k / 60.0 for k in SEC_PER_VIDEO_MIN)
        total = ('%.1f 小時' % (mins / 60)) if mins >= 60 else ('%d 分鐘' % round(mins))
        pending = any(f.get('info') is None for f in self.files)
        over = mins / 60.0 > MAX_HOURS
        msg = '已加入 %d 支，合計%s %s' % (n, '目前約' if pending else '約', total)
        if over:
            msg += '\n⚠ 超過 %d 小時上限，請移除部分影片，分批判讀。' % MAX_HOURS
        else:
            msg += '\n上限 %d 支、%d 小時；預估約 %d～%d 分鐘' % (MAX_FILES, MAX_HOURS, max(1, round(lo)), max(1, round(hi)))
        self.var_count.set(msg)
        if hasattr(self, 'lbl_count'):
            self.lbl_count.configure(foreground='#b00000' if over else '#1060c0')

    def refresh_files(self):
        st = self.var_status.get() if hasattr(self, 'var_status') else None
        if st in (STATUS_EMPTY, STATUS_READY):          # 只換掉開頭的提示，不蓋掉判讀完成等訊息
            self.var_status.set(STATUS_READY if self.files else STATUS_EMPTY)
        sel = self.tv_files.selection()
        self.tv_files.delete(*self.tv_files.get_children())
        for i, f in enumerate(self.files):
            vals, tag = self._row(f)
            self.tv_files.insert('', 'end', iid=str(i), values=vals, tags=tag)
        self.update_count()
        keep = [s for s in sel if int(s) < len(self.files)]
        if keep:
            self.tv_files.selection_set(keep)
        if not self.files:
            self.canvas.delete('all'); self.cur_frame = None

    def on_file_select(self):
        sel = self.tv_files.selection()
        if len(sel) != 1:
            return
        f = self.files[int(sel[0])]
        st = self._start_of(f)
        self.var_ftime.set(st.strftime('%Y-%m-%d %H:%M:%S') if st else '')
        dur = (f.get('info') or {}).get('dur') or 600
        self.scale.configure(to=max(1.0, dur - 0.5))
        self.schedule_seek(0)

    # ------------------------------------------------------------ 預覽與畫線
    def step(self, d):
        self.var_pos.set(max(0.0, min(float(self.scale.cget('to')), self.var_pos.get() + d)))
        self.schedule_seek()

    def schedule_seek(self, delay=120):
        if self._seek_job:
            self.root.after_cancel(self._seek_job)
        self._seek_job = self.root.after(delay, self.show_frame)

    def _sel_file(self):
        sel = self.tv_files.selection()
        return self.files[int(sel[0])] if sel else None

    def show_frame(self):
        """預覽在背景讀，視窗不會卡住；又拖了別的位置時，舊的讀取會停下來"""
        self._seek_job = None
        f = self._sel_file()
        if not f:
            return
        self._pv_token = getattr(self, '_pv_token', 0) + 1
        tok, path, t = self._pv_token, f['path'], self.var_pos.get()
        if not hasattr(self, 'pv_reader'):
            self.pv_reader = core.PreviewReader()
            self.pv_lock = threading.Lock()

        def prog(p, target):
            self.q.put(('pv_prog', tok, p, target))

        def work():
            with self.pv_lock:
                if tok != self._pv_token:
                    return
                try:
                    fr, pos = self.pv_reader.read(path, t, cancel=lambda: tok != self._pv_token, progress=prog)
                except Exception:
                    fr, pos = None, 0.0
            self.q.put(('pv_done', tok, fr, pos))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(400, lambda: tok == self._pv_token and self._pv_busy(tok))

    def _on_canvas_size(self, e):
        if (e.width, e.height) == (self.cw, self.ch):
            return
        self.cw, self.ch = max(50, e.width), max(30, e.height)
        if getattr(self, '_resize_job', None):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._render_frame)

    def _render_frame(self):
        self._resize_job = None
        fr = self.cur_frame
        if fr is None:
            return
        h, w = fr.shape[:2]
        self.disp_scale = min(self.cw / w, self.ch / h)
        img = Image.fromarray(fr[:, :, ::-1]).resize((max(1, int(w * self.disp_scale)), max(1, int(h * self.disp_scale))), Image.BILINEAR)
        self.frame_img = ImageTk.PhotoImage(img)
        self.canvas.delete('frame')
        self.canvas.configure(bg=ttk.Style(self.root).lookup('TFrame', 'background') or '#dcdad5')  # 畫面外不留黑邊
        self.canvas.create_image(0, 0, image=self.frame_img, anchor='nw', tags='frame')
        self.canvas.tag_lower('frame')
        self.redraw_overlay()

    def _pv_busy(self, tok):
        if getattr(self, '_pv_shown', None) != tok:
            self.var_postxt.set('讀取畫面中…')
            self.var_osd.set('')
            self.canvas.delete('busy')
            cx, cy = self.cw / 2, self.ch / 2
            self.canvas.create_rectangle(cx - 150, cy - 24, cx + 150, cy + 24,
                                         fill='#202020', outline='#ffffff', tags='busy')
            self.canvas.create_text(cx, cy, text='讀取畫面中，請稍候…', fill='#ffffff',
                                    font=(self.font_family, 12, 'bold'), tags='busy')

    def _show_preview(self, fr, pos):
        self.canvas.delete('busy')
        self.cur_frame, self.cur_pos = fr, pos
        self._render_frame()
        self.var_postxt.set('影片內 %s' % export.fmt_pos(pos))
        self.update_osd_label()

    def update_osd_label(self):
        if self.cur_frame is None:
            self.var_osd.set('')
            return
        try:
            d, txt, w = osd.read_datetime(self.cur_frame, self.profile.osd_rect, self.profile.osd_format, self._templates())
        except Exception:
            d, txt, w = None, '', 0
        if d is not None:
            self.var_osd.set(txt)
            self.lbl_osd.configure(style='Ok.TLabel')
        else:
            self.var_osd.set('讀不到（%s）' % txt if txt else '讀不到')
            self.lbl_osd.configure(style='Warn.TLabel')

    def redraw_overlay(self):
        c = self.canvas
        c.delete('ov')
        if self.cur_frame is None:
            return
        s = self.disp_scale
        x0, y0, x1, y1 = self.profile.osd_rect
        c.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s, outline='#30a0ff', width=2, tags='ov')
        x0, y0, x1, y1 = self.profile.track_rect
        c.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s, outline='#ffa000', width=2, tags='ov')
        try:
            wd = max(3, min(25, int(self.var_width.get())))
        except (tk.TclError, ValueError):
            wd = self.profile.ref_width
        (xa, ya), (xb, yb) = self.profile.ref_line
        c.create_line(xa * s, ya * s, xb * s, yb * s, fill='#ff2020', width=max(2, wd * s), tags='ov')
        fnt = (self.font_family, 10, 'bold')
        c.create_text(x0 * s + 4, y0 * s - 3, text='軌道範圍', fill='#ffa000', anchor='sw', tags='ov', font=fnt)
        c.create_text(max(xa, xb) * s + max(2, wd * s) / 2 + 4, min(ya, yb) * s - 3, text='參考線', fill='#ff4040',
                      anchor='sw', tags='ov', font=fnt)

    def _to_video(self, e):
        s = self.disp_scale
        h, w = self.cur_frame.shape[:2]
        return [int(max(0, min(w - 1, e.x / s))), int(max(0, min(h - 1, e.y / s)))]

    def on_press(self, e):
        if self.cur_frame is None:
            return
        self.drag = self._to_video(e)

    def on_drag(self, e):
        if self.cur_frame is None or self.drag is None:
            return
        p = self._to_video(e)
        box = [min(self.drag[0], p[0]), min(self.drag[1], p[1]), max(self.drag[0], p[0]), max(self.drag[1], p[1])]
        m = self.mode.get()
        if m == 'ref':
            self.profile.ref_line = [list(self.drag), p]
        elif m == 'track':
            self.profile.track_rect = box
        else:
            self.profile.osd_rect = box
        self.redraw_overlay()

    def on_release(self, e):
        if self.drag is None:
            return
        if self.cur_frame is not None:
            self.profile.frame_size = [int(self.cur_frame.shape[1]), int(self.cur_frame.shape[0])]
        self.on_drag(e)
        self.drag = None
        if self.mode.get() == 'osd':
            self.update_osd_label()
            self.var_status.set('時間字幕位置已更新；確認「畫面時間判讀」和畫面上一樣後，按「儲存」。')
            self.rescan_all()
            return
        msg = self.check_geometry()
        self.var_status.set(msg or '參考線與軌道範圍已更新（記得按「儲存」保存這個監測站）。')

    def check_geometry(self):
        p = self.profile
        (xa, ya), (xb, yb) = p.ref_line
        x0, y0, x1, y1 = p.track_rect
        if abs(xb - xa) + abs(yb - ya) < 8:
            return '參考線太短，請拖曳畫一條橫過軌道的線。'
        if (x1 - x0) < 20 or (y1 - y0) < 8:
            return '軌道範圍太小，請沿軌道框大一點。'
        rp = p.ref_pos()
        lo, hi = (x0, x1) if p.horizontal() else (y0, y1)
        if not (lo + 15 < rp < hi - 15):
            return '參考線要在軌道範圍中間（兩側都要留一段），方向才判斷得出來。'
        return ''

    def teach_osd(self):
        f = self._sel_file()
        if f is None or self.cur_frame is None:
            messagebox.showinfo(APP, '請先選一支影片，畫面上要看得到時間字幕。')
            return
        try:
            d, txt, _w = osd.read_datetime(self.cur_frame, self.profile.osd_rect, self.profile.osd_format, self._templates())
        except Exception:
            d, txt = None, ''
        s = simpledialog.askstring(APP, '請輸入「現在這個畫面」上顯示的日期時間\n（例：2026-01-01 08:00:00）。\n'
                                   '程式會往後讀 13 秒，學會這台攝影機的數字字型。',
                                   initialvalue=(d.strftime('%Y-%m-%d %H:%M:%S') if d else ''), parent=self.root)
        if not s:
            return
        try:
            shown = parse_datetime(s)
        except ValueError as e:
            messagebox.showerror(APP, str(e))
            return
        try:
            T, missing = osd.learn(f['path'], self.cur_pos, shown.replace(microsecond=0), self.profile.osd_rect, self.profile.osd_format)
        except Exception as e:
            messagebox.showerror(APP, '學習失敗：%s' % e)
            return
        self.profile.osd_templates = osd.templates_to_list(T)
        self.update_osd_label()
        msg = '已學會這台攝影機的數字字型。'
        if missing:
            msg += '\n（%s 這幾個數字沒出現，暫用內建字樣）' % '、'.join(str(x) for x in missing)
        messagebox.showinfo(APP, msg + '\n記得按「儲存」保存這個監測站。')
        self.rescan_all()

    # ------------------------------------------------------------ 執行
    def start_run(self):
        if self.worker and self.worker.is_alive():
            return
        if self.dirty and not self.confirm('確認', '第 2 頁的修改還沒儲存，重新判讀後這些修改會遺失。\n\n確定要開始判讀嗎？', ok='開始判讀', warn=True):
            return
        if not self.files:
            messagebox.showinfo(APP, '請先加入影片。')
            return
        msg = self.check_geometry()
        if msg:
            messagebox.showerror(APP, msg)
            return
        if self.scan_queue or any(f.get('info') is None for f in self.files):
            messagebox.showinfo(APP, '還在讀取影片的畫面時間，請等清單上的時間都出現後再開始。')
            return
        hours = sum(((f.get('info') or {}).get('dur') or 600) for f in self.files) / 3600.0
        if hours > MAX_HOURS:
            messagebox.showerror(APP, '目前加入的影片合計約 %.1f 小時，超過一次 %d 小時的上限。\n請移除部分影片，分批判讀。' % (hours, MAX_HOURS))
            return
        broken = [os.path.basename(f['path']) for f in self.files if f.get('info') == {}]
        if broken:
            messagebox.showerror(APP, '這些影片無法開啟，請先移除：\n' + '\n'.join(broken[:15]))
            return
        size_err = self.check_sizes()
        if size_err:
            messagebox.showerror(APP, size_err + '\n\n參考線、軌道範圍是用像素位置畫的，畫面大小不同時位置一定會錯，所以不能一起判讀。\n'
                                 '請移除大小不同的影片，或分開判讀（大小不同的攝影機請另外設定一個監測站）。')
            return
        issues = self.check_files()
        if issues and not self.confirm('影片檢查', '開始判讀前發現下列狀況：\n\n' + '\n'.join(issues[:15])
                                       + ('\n…（共 %d 項）' % len(issues) if len(issues) > 15 else '')
                                       + '\n\n確定要照目前的清單判讀嗎？', ok='仍要判讀', warn=True):
            return
        prof = self._collect_profile()
        files = [f['path'] for f in self.files]
        manual = [f.get('manual') for f in self.files]
        if self.out_location and not os.path.isdir(self.out_location):
            messagebox.showerror(APP, '找不到結果存放位置：\n%s\n\n可能是資料夾被改名、刪除，或隨身碟／網路磁碟沒有接上。\n'
                                 '請按「選擇存放位置…」重新選，或按「改回影片旁邊」。' % self.out_location)
            return
        root_dir = result_root(self.out_location, files[0])
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        safe = ''.join('_' if c in '\\/:*?"<>|' else c for c in prof.name)
        out = os.path.join(root_dir, '%s_%s' % (safe, stamp))
        self._made_root = None if os.path.isdir(root_dir) else root_dir   # 取消時只收掉這次才建立的空資料夾
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            messagebox.showerror(APP, '無法建立結果資料夾：\n%s\n\n%s\n\n請按「選擇存放位置…」換一個可以寫入的位置。' % (out, e))
            return
        self.cancel_flag = False
        self.run_t0 = time.time()
        self.btn_run.configure(state='disabled'); self.btn_cancel.configure(state='normal')
        self.pb['value'] = 0
        prof_copy = Profile.from_dict(json.loads(prof.to_json()))
        self.worker = threading.Thread(target=self._work, args=(files, manual, prof_copy, out), daemon=True)
        self.worker.start()

    def check_sizes(self):
        """畫面大小：影片之間不一致、或和監測站設定（畫參考線時的畫面）不同 → 回傳錯誤說明（#50，不能「仍要判讀」）"""
        sizes = {}
        for f in self.files:
            i = f.get('info') or {}
            sizes.setdefault((i.get('w'), i.get('h')), []).append(os.path.basename(f['path']))
        if len(sizes) > 1:
            return ('影片的畫面大小不一致：' + '；'.join('%s×%s 有 %d 支（例：%s）' % (k[0], k[1], len(v), v[0])
                                                    for k, v in sizes.items()))
        fs = self.profile.frame_size
        if fs and sizes:
            k = next(iter(sizes))
            if list(k) != list(fs):
                return '這個監測站的參考線是在 %d×%d 的畫面上畫的，但這些影片是 %s×%s。' % (fs[0], fs[1], k[0], k[1])
        return ''

    def check_files(self):
        """判讀前的檢查：時間順序倒退／重疊／中間缺一段。回傳說明清單（空的＝沒問題）"""
        out = []
        prev = None
        for f in self.files:
            st = self._start_of(f)
            if st is None:
                continue
            dur = (f.get('info') or {}).get('dur') or 600
            name = os.path.basename(f['path'])
            if prev is not None:
                p_st, p_end, p_name = prev
                gap = (st - p_end).total_seconds()
                if st < p_st:
                    out.append('● 時間倒退：%s（%s）排在 %s（%s）後面。可以按「依時間排序」。'
                               % (name, st.strftime('%m-%d %H:%M:%S'), p_name, p_st.strftime('%m-%d %H:%M:%S')))
                elif gap < -2:
                    out.append('● 時間重疊 %d 秒：%s 和 %s（可能重複加入同一段錄影）' % (-gap, p_name, name))
                elif gap > 5:
                    if gap >= 3600:
                        g = '%d 小時 %d 分' % (gap // 3600, gap % 3600 // 60)
                    elif gap >= 60:
                        g = '%d 分 %d 秒' % (gap // 60, gap % 60)
                    else:
                        g = '%d 秒' % gap
                    out.append('● 中間缺 %s：%s 結束到 %s 開始之間沒有錄影（缺檔？）' % (g, p_name, name))
            prev = (st, st + dt.timedelta(seconds=dur), name)
        return out

    def cancel_run(self):
        self.cancel_flag = True
        self.var_status.set('取消中…（已完成的部分仍會輸出）')

    def _work(self, files, manual, prof, out):
        try:
            T = osd.templates_from_list(prof.osd_templates)
            timing, bad = [], []
            for i, (f, m) in enumerate(zip(files, manual)):
                self.q.put(('prog', 0.08 * i / len(files), '讀取畫面時間 %d/%d：%s' % (i + 1, len(files), os.path.basename(f))))
                if m is not None:
                    timing.append(dict(offset=m.timestamp(), source='手動', support=None, msg=''))
                    continue
                r = osd.calibrate(f, prof.osd_rect, prof.osd_format, T, cancel=lambda: self.cancel_flag)
                if not r.get('ok'):
                    bad.append('%s：%s' % (os.path.basename(f), r.get('msg', '')))
                timing.append(dict(offset=r.get('offset'), source='畫面字幕', support=r.get('support'), msg=r.get('msg', '')))
                if self.cancel_flag:
                    break
            if bad:
                self.q.put(('stop', '下列影片讀不到可靠的畫面時間，請在清單選取後手動填開始時間，或調整「時間字幕位置」：\n\n'
                            + '\n'.join(bad[:15]), out))
                return
            if self.cancel_flag:
                self.q.put(('stop', '已取消。', out))
                return
            # 正式讀完畫面時間後再檢查一次順序與重疊（加入影片時只快速讀開頭，#48）
            files, timing, sorted_note, overlap = final_order(files, timing)
            if overlap:
                self.q.put(('stop', '正式讀取畫面時間後，發現下列影片的時間重疊（可能重複加入同一段錄影），請移除重複的再判讀：\n\n'
                            + '\n'.join(overlap[:15]), out))
                return
            bases = [t['offset'] for t in timing]
            prog = lambda fr, msg: self.q.put(('prog', 0.08 + fr * 0.8, '判讀中：' + msg))
            info = {}
            try:
                evs = core.process(files, bases, prof, prog, lambda: self.cancel_flag, info=info)
            except core.TimeOrderError as ex:                       # 判讀途中發現時間倒退／重疊（#69）
                self.q.put(('stop', str(ex), out))
                return
            mark_time_jumps(evs, files, timing)
            cam = info.get('camera')
            if cam:                                                    # 攝影機位置改變 → 判讀已停止（#71）
                for f, t in zip(files, timing):
                    if f == cam['file']:
                        t['stop'] = camera_stop_text(cam)
            core.number_events(evs)
            prog2 = lambda fr, msg: self.q.put(('prog', 0.88 + fr * 0.12, msg))
            core.save_frames_and_clips(evs, prof, out, prog2, lambda: self.cancel_flag, files=files, bases=bases)
            for t in timing:
                if sorted_note:
                    t.setdefault('note', sorted_note)
            export.save_json(os.path.join(out, 'results.json'), evs, prof, files, timing)
            export.write_excel(os.path.join(out, '列車通過紀錄.xlsx'), evs, prof, files, timing)
            export.write_csv(os.path.join(out, '列車通過紀錄.csv'), evs)
            self.q.put(('done', evs, out, files, timing, prof, self.cancel_flag))
        except Exception as e:
            self.q.put(('error', '%s\n\n%s' % (e, traceback.format_exc())))

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                kind = m[0]
                if kind == 'prog':
                    self.pb['value'] = int(m[1] * 1000)
                    txt = m[2]
                    el = time.time() - getattr(self, 'run_t0', time.time())
                    if m[1] > 0.1 and el > 20:             # 進度一成以後才估，比較準
                        left = el * (1 - m[1]) / m[1]
                        txt += '\n已經 %d 分鐘，預估還要約 %d 分鐘。' % (el // 60, max(1, round(left / 60)))
                    self.var_status.set(txt)
                elif kind == 'scan':
                    _, f, r = m
                    f['osd'] = r
                    f['state'] = ''
                    # 只更新這一列（不要整個清單重畫，影片多時才不會越來越慢）
                    i = next((k for k, x in enumerate(self.files) if x is f), None)
                    if i is not None and self.tv_files.exists(str(i)):
                        vals, tag = self._row(f)
                        self.tv_files.item(str(i), values=vals, tags=tag)
                    self.update_count()
                elif kind == 'pv_prog':
                    _, tok, p, target = m
                    if tok == self._pv_token:
                        self.var_postxt.set('讀取畫面中… %d%%' % min(99, 100 * p / max(1e-6, target)))
                elif kind == 'pv_done':
                    _, tok, fr, pos = m
                    if tok == self._pv_token:
                        self._pv_shown = tok
                        if fr is not None:
                            self._show_preview(fr, pos)
                        else:
                            self.canvas.delete('busy')
                            self.var_postxt.set('讀不到這個位置的畫面')
                elif kind == 'regen_done':
                    _, e, work_ev, err = m
                    e.pop('media_busy', None)
                    if not err:
                        e['shots'], e['clip'] = work_ev.get('shots') or [], work_ev.get('clip') or ''
                        e['media_tag'] = work_ev.get('media_tag')
                    why = e.pop('_regen_why', 'time')
                    self.var_tab2msg.set(('截圖與短片產生失敗：%s' % err) if err else
                                         ('已依修改後的時間（%s～%s）重新產生截圖與短片。' % (core.fmt_time(e['start']), core.fmt_time(e['end']))
                                          if why == 'time' else '已重新產生這一筆的截圖與短片。'))
                    _i, cur = self._sel_event()
                    if cur is e:
                        self.show_shot(0)
                    if e.pop('play_after', False) and e.get('clip'):
                        open_path(os.path.join(self.result_dir, '短片', e['clip']))
                elif kind == 'scan_done':
                    if self.scan_queue:
                        self._start_scanner()
                elif kind == 'done':
                    _, evs, out, files, timing, prof, cancelled = m
                    self._run_finished()
                    self.pb['value'] = 1000
                    self.events, self.result_dir, self.run_files, self.run_timing, self.run_profile = evs, out, files, timing, prof
                    self.set_dirty(False)       # 判讀完已自動輸出 Excel
                    nv = sum(1 for e in evs if e['valid'])
                    self.root.after(300, lambda evs=evs, c=cancelled, t=timing: self.show_summary(evs, c, t))
                    stopped = any(t.get('stop') for t in timing)
                    if stopped:
                        self.var_status.set('⚠ 攝影機位置改變，判讀已停止。\n停止前：列車%d筆、已排除%d筆。\n結果資料夾：%s'
                                            % (nv, len(evs) - nv, os.path.basename(out)))
                    else:
                        self.var_status.set('%s完成：列車 %d 筆、已排除 %d 筆。\n結果資料夾：%s' % (
                            '已取消，部分' if cancelled else '', nv, len(evs) - nv, os.path.basename(out)))
                    self.fill_events()
                    self.nb.select(self.page2)
                elif kind == 'stop':
                    self._run_finished()
                    self._remove_empty(m[2])
                    if getattr(self, '_made_root', None):          # 這次才建立的空「判讀結果」資料夾也收掉
                        self._remove_empty(self._made_root)
                    self.var_status.set('沒有開始判讀。')
                    messagebox.showwarning(APP, m[1])
                elif kind == 'error':
                    self._run_finished()
                    self.var_status.set('發生錯誤，請看錯誤訊息。')
                    messagebox.showerror(APP, '判讀時發生錯誤：\n' + m[1][:1500])
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def _run_finished(self):
        self.btn_run.configure(state='normal'); self.btn_cancel.configure(state='disabled')

    # ------------------------------------------------------------ 結果存放位置
    def update_out_location(self):
        if not hasattr(self, 'var_outloc'):
            return
        if self.out_location:
            txt = result_root(self.out_location, '')
        elif self.files:
            txt = result_root(None, self.files[0]['path'])
        else:
            txt = '影片旁邊的「%s」資料夾' % RESULT_FOLDER
        txt = ('結果存放（預設）：' if self.out_location == self.out_default else '結果存放（只用這一次）：') + txt
        self.var_outloc.set(txt)

    def choose_out_location(self):
        init = self.out_location or (os.path.dirname(self.files[0]['path']) if self.files else None)
        d = filedialog.askdirectory(title='選擇判讀結果要存放的位置（會在裡面建立「%s」資料夾）' % RESULT_FOLDER,
                                    initialdir=init, mustexist=True)
        if not d:
            return
        d = os.path.abspath(d)
        self.out_location = d
        if d != self.out_default:
            if self.confirm('結果存放位置', '判讀結果會存在：\n%s\n\n要把這個位置設成以後的預設嗎？\n'
                            '（選「只用這一次」，下次開程式會回到原本的預設）' % result_root(d, ''),
                            ok='設為預設', cancel='只用這一次'):
                self._set_default_location(d)
        self.update_out_location()

    def reset_out_location(self):
        self.out_location = None
        if self.out_default is not None:
            if self.confirm('結果存放位置', '這次的結果會放在影片旁邊的「%s」資料夾。\n\n'
                            '要把以後的預設也改回影片旁邊嗎？\n（目前的預設是：%s）'
                            % (RESULT_FOLDER, result_root(self.out_default, '')),
                            ok='預設也改回', cancel='只用這一次'):
                self._set_default_location(None)
        self.update_out_location()

    def _set_default_location(self, d):
        self.settings['result_location'] = d
        if save_settings(self.settings):
            self.out_default = d
        else:
            messagebox.showwarning(APP, '無法儲存預設位置（程式資料夾和使用者資料夾都不能寫入），這次仍會使用您選的位置。')

    def show_summary(self, evs, cancelled, timing):
        need = any(e.get('need_check') for e in evs if e.get('valid')) or any(t.get('stop') for t in timing)
        self.confirm('判讀完成', quality_summary(evs, cancelled, timing), ok='知道了', cancel=None,
                     warn=need, icon=None if need else '✓')

    def _remove_empty(self, d):
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass

    # ------------------------------------------------------------ 結果
    def fill_events(self):
        self.tv.delete(*self.tv.get_children())
        show_x = self.show_excluded.get()
        nv = nx = nn = 0
        for i, e in enumerate(self.events):
            if e['valid']:
                nv += 1
                if e.get('need_check') and not e.get('checked'):
                    nn += 1
            else:
                nx += 1
            if not e['valid'] and not show_x:
                continue
            need = e['valid'] and e.get('need_check') and not e.get('checked')
            d = dt.datetime.fromtimestamp(e['start'])
            vals = (e['no'] if e['valid'] else 'X%d' % e['no'], d.strftime('%m-%d'), core.fmt_time(e['start']),
                    core.fmt_time(e['end']), '%.1f' % (e['end'] - e['start']), e.get('direction', ''),
                    e.get('train_type', ''), '是' if e.get('crossing') else '',
                    '是' if need else ('已確認' if e.get('checked') else ''),
                    (e.get('reason') or '') + (('；' + e['note']) if e.get('note') else ''))
            self.tv.insert('', 'end', iid=str(i), values=vals, tags=('need',) if need else (('excl',) if not e['valid'] else ()))
        self.var_sum.set('列車 %d 筆（需確認 %d）、已排除 %d 筆' % (nv, nn, nx))

    def _sel_event(self):
        sel = self.tv.selection()
        return (int(sel[0]), self.events[int(sel[0])]) if sel else (None, None)

    def on_event_select(self):
        i, e = self._sel_event()
        if e is None:
            return
        self.var_es.set(core.fmt_time(e['start'])); self.var_ee.set(core.fmt_time(e['end']))
        self.var_dir.set(e.get('direction', '')); self.var_type.set(e.get('train_type', ''))
        self.var_cross.set(bool(e.get('crossing'))); self.var_checked.set(bool(e.get('checked')))
        self.var_note.set(e.get('note', ''))
        self.btn_valid.configure(text='改為「不是列車」' if e['valid'] else '改為「是列車」')
        has_peak = any('2變化最大' in x for x in (e.get('shots') or []))
        self.shot_btns[1].configure(text='變化最大' if has_peak else '中間')
        self.shot_btns[0].configure(text='車頭' if e.get('start_known', True) else '第一畫面')   # 車頭／車尾未確認（#70）
        self.shot_btns[2].configure(text='車尾' if e.get('end_known', True) else '最後畫面')
        self.show_shot(1 if has_peak else 0)          # 已排除的先顯示「變化最大」，最看得出為什麼被排除

    def show_shot(self, k):
        i, e = self._sel_event()
        self.shot_canvas.delete('all')
        if e is None:
            return
        shots = e.get('shots') or []
        want = (('1車頭', '1第一畫面'), ('2中間', '2變化最大'), ('3車尾', '3最後畫面'))[k]   # 已排除的第二張是「變化最大」（#60）
        name = next((s for s in shots if any(w in s for w in want)), None)
        if not name:
            self.var_shotlab.set('（沒有截圖）')
            return
        path = os.path.join(self.result_dir, '截圖', name)
        try:
            im = Image.open(path)
            im.thumbnail((480, 270))
            self.shot_img = ImageTk.PhotoImage(im)
            self.shot_canvas.create_image(240, 135, image=self.shot_img)
            self.var_shotlab.set(name)
        except Exception:
            self.var_shotlab.set('找不到截圖：%s' % name)

    def play_clip(self):
        i, e = self._sel_event()
        if e is None:
            return
        if e.get('media_busy'):
            e['play_after'] = True
            self.var_tab2msg.set('短片還在依修改後的時間產生中，做好會自動播放…')
            return
        if not e.get('clip'):
            messagebox.showinfo(APP, '這一筆沒有短片。已排除的事件不產生短片，請看截圖或按「在影片中看這段」。'
                                if not e.get('valid') else '這一筆沒有短片（可在「進階」開啟產生短片）。')
            return
        open_path(os.path.join(self.result_dir, '短片', e['clip']))

    def goto_event(self):
        """切到第 1 頁，定位到這一筆開始前 2 秒"""
        i, e = self._sel_event()
        if e is None:
            return
        for k, f in enumerate(self.files):
            if os.path.normcase(f['path']) == os.path.normcase(e.get('start_file') or ''):
                self.tv_files.selection_set(str(k))
                self.root.update_idletasks()
                self.var_pos.set(max(0.0, (e.get('start_pos') or 0) - 2))
                self.schedule_seek(10)
                self.nb.select(self.page1)
                return
        messagebox.showinfo(APP, '這一筆的影片不在第 1 頁的清單中：\n%s' % (e.get('start_file') or ''))

    def _parse_event_time(self, text, ref_ts):
        s = parse_hms(text)
        d = dt.datetime.fromtimestamp(ref_ts)
        t = dt.datetime(d.year, d.month, d.day).timestamp() + s
        if t - ref_ts > 43200:            # 跨午夜：與原本時間差超過 12 小時，視為前一天／後一天
            t -= 86400
        elif ref_ts - t > 43200:
            t += 86400
        return t

    def apply_edit(self):
        i, e = self._sel_event()
        if e is None:
            messagebox.showinfo(APP, '請先在清單選一筆。')
            return
        if self._media_busy():
            return
        try:
            ns = self._parse_event_time(self.var_es.get(), e['start'])
            ne = self._parse_event_time(self.var_ee.get(), e['end'])
        except ValueError as ex:
            messagebox.showerror(APP, '時間格式不對：%s' % ex)
            return
        if ne < ns:
            messagebox.showerror(APP, '車尾離開不能早於車頭到達。')
            return
        time_changed = abs(ns - e['start']) > 0.05 or abs(ne - e['end']) > 0.05
        if time_changed:
            loc_s, loc_e = self._locate(ns), self._locate(ne)
            if loc_s is None or loc_e is None:
                bad = core.fmt_time(ns if loc_s is None else ne)
                messagebox.showerror(APP, '%s 沒有對應的錄影畫面（不在這次判讀的影片範圍內，或落在兩支影片中間沒有錄影的缺口）。\n'
                                     '請確認時間是否正確。' % bad)
                return
            e['start_file'], e['start_pos'] = loc_s
            e['end_file'], e['end_pos'] = loc_e
        e['start'], e['end'] = ns, ne
        e['direction'] = self.var_dir.get()
        e['train_type'] = self.var_type.get().strip()
        e['crossing'] = bool(self.var_cross.get())
        e['checked'] = bool(self.var_checked.get())
        e['note'] = self.var_note.get().strip()
        self._refresh_keep(e)
        if time_changed:
            self.regen_media(e)

    def _locate(self, ts):
        """畫面時間 → (影片檔, 影片內秒數)；用這次判讀時每支影片的時間對照"""
        best = None
        for f, tm in zip(self.run_files, self.run_timing):
            off = tm.get('offset')
            if off is None:
                continue
            if not hasattr(self, '_dur_cache'):
                self._dur_cache = {}
            if f not in self._dur_cache:
                self._dur_cache[f] = ((core.video_info(f) or {}).get('dur') or 0.0)
            pos = ts - off
            if -0.05 <= pos <= self._dur_cache[f] + 0.05:     # 只接受真的有錄影的時間，不吸附到旁邊的影片（#58）
                best = (f, min(max(0.0, pos), self._dur_cache[f]))
                break
        return best

    def regen_media(self, e, why='time'):
        """依這一筆目前的時間重做截圖與短片（時間改過、或改成「是列車」時），在背景做。
        背景只處理副本，做完由畫面執行緒一次套用；做的時候不能再套用修改或儲存（#55）"""
        e['media_busy'] = True
        e['_regen_why'] = why
        prof = self.run_profile or self.profile
        files = list(self.run_files)
        bases = [t.get('offset') for t in self.run_timing] if self.run_timing else None
        out = self.result_dir
        work_ev = {k: v for k, v in e.items() if k not in ('shots', 'clip')}
        if not work_ev.get('media_tag'):       # v1.0.8 以前的結果：沿用原本的檔名前綴（例：002、X001）
            old = (e.get('shots') or [''])[0].split('_')[0] or (e.get('clip') or '').split('.')[0]
            if old:
                work_ev['media_tag'] = old
        self.var_tab2msg.set('依修改後的時間重新產生截圖與短片中…（完成前不能再套用修改或儲存）' if why == 'time'
                             else '產生這一筆的短片中…（完成前不能再套用修改或儲存）')

        def work():
            try:
                core.save_frames_and_clips([work_ev], prof, out, files=files, bases=bases)
                err = ''
            except Exception as ex:
                err = str(ex)
            self.q.put(('regen_done', e, work_ev, err))
        threading.Thread(target=work, daemon=True).start()

    def _media_busy(self):
        if any(ev.get('media_busy') for ev in self.events or []):
            messagebox.showinfo(APP, '截圖與短片還在更新，請等上方顯示「已重新產生」後再操作。')
            return True
        return False

    def toggle_valid(self):
        i, e = self._sel_event()
        if e is None:
            return
        if self._media_busy():
            return
        e['valid'] = not e['valid']
        tag = '人工改為列車' if e['valid'] else '人工改為非列車'
        note = e.get('note') or ''
        if tag not in note:
            e['note'] = (note + '；' + tag).strip('；')
        self._refresh_keep(e)
        if e['valid'] and not e.get('clip') and (self.run_profile or self.profile).make_clips and self.result_dir:
            self.regen_media(e, why='valid')        # 已排除的沒有短片（#60），改成列車時補做
        if not e['valid'] and not self.show_excluded.get():
            # 不自動打開「顯示已排除」（使用者要求）；那一筆從清單消失，改選它原本位置的下一筆
            kids = self.tv.get_children()
            if kids:
                nxt = next((k for k in kids if self.events[int(k)]['start'] >= e['start']), kids[-1])
                self.tv.selection_set(nxt); self.tv.see(nxt)
            else:
                self.shot_canvas.delete('all'); self.var_shotlab.set('')
            self.var_tab2msg.set('已把 %s～%s 那一筆改為「不是列車」。要再看到它，請勾「顯示已排除」。'
                                 % (core.fmt_time(e['start']), core.fmt_time(e['end'])))

    def set_dirty(self, v):
        self.dirty = v
        self.root.title('%s v%s%s' % (APP, VERSION, '　（有修改還沒儲存）' if v else ''))

    def _refresh_keep(self, e):
        self.set_dirty(True)
        self.var_tab2msg.set('')
        core.number_events(self.events)        # 依時間重新排序、編號
        i = next(k for k, x in enumerate(self.events) if x is e)
        self.fill_events()
        if self.tv.exists(str(i)):
            self.tv.selection_set(str(i)); self.tv.see(str(i))

    def save_results(self):
        if not self.result_dir:
            messagebox.showinfo(APP, '還沒有結果。')
            return
        if self._media_busy():
            return
        prof = self.run_profile or self.profile
        try:
            export.save_json(os.path.join(self.result_dir, 'results.json'), self.events, prof, self.run_files, self.run_timing)
            export.write_excel(os.path.join(self.result_dir, '列車通過紀錄.xlsx'), self.events, prof, self.run_files, self.run_timing)
            export.write_csv(os.path.join(self.result_dir, '列車通過紀錄.csv'), self.events)
        except PermissionError:
            messagebox.showerror(APP, '無法寫入，Excel 檔可能正開著。請先關閉「列車通過紀錄.xlsx」再試一次。')
            return
        except Exception as ex:
            messagebox.showerror(APP, '儲存失敗：%s' % ex)
            return
        self.set_dirty(False)
        messagebox.showinfo(APP, '已儲存：\n%s' % os.path.join(self.result_dir, '列車通過紀錄.xlsx'))

    def export_noise(self):
        """噪音分析程式用的記事本檔：日期,進入時間,離開時間,10,FR,,A（只含「是列車」的）"""
        if not self.result_dir or not self.events:
            messagebox.showinfo(APP, '還沒有結果。')
            return
        valid = [e for e in self.events if e.get('valid')]
        if not valid:
            messagebox.showinfo(APP, '沒有判定為列車的事件，不需要輸出。')
            return
        unchecked = sum(1 for e in valid if e.get('need_check') and not e.get('checked'))
        no_end = sum(1 for e in valid if e.get('end_known') is False and not e.get('checked'))
        if unchecked and not self.confirm('確認', '還有 %d 筆「需人工確認」還沒勾「已人工確認」。%s\n\n'
                                          '這些也會照目前的時間輸出。確定要輸出嗎？'
                                          % (unchecked, ('\n\n其中%d筆「車尾未確認」：離開時間只是程式最後看到列車的時間，\n'
                                                         '不是真的車尾離開（影片結束、取消、錄影中斷或長時間沒有動靜）。' % no_end) if no_end else ''),
                                          ok='照樣輸出', warn=True):
            return
        path = filedialog.asksaveasfilename(title='儲存噪音分析用的記事本檔', initialdir=self.result_dir,
                                            initialfile='列車進入及離開時間.txt', defaultextension='.txt',
                                            filetypes=[('記事本', '*.txt')])
        if not path:
            return
        try:
            n = export.write_noise_txt(path, self.events)
        except PermissionError:
            messagebox.showerror(APP, '無法寫入，檔案可能正開著。請先關閉再試一次。')
            return
        except Exception as ex:
            messagebox.showerror(APP, '輸出失敗：%s' % ex)
            return
        note = '（第 2 頁的修改還沒按「儲存修改並重新輸出 Excel」，但這份 TXT 已經是修改後的時間。）' if self.dirty else ''
        messagebox.showinfo(APP, '已輸出 %d 筆列車：\n%s\n\n已排除（非列車）的沒有輸出。%s' % (n, path, note))

    def _results_start_dir(self):
        """「開啟先前的結果…」從哪個資料夾開始：剛判讀（或剛開啟）的結果所在的「判讀結果」→
        目前存放位置的「判讀結果」→ 預設位置的「判讀結果」；都不存在就交給作業系統"""
        cands = []
        if self.result_dir:
            cands.append(os.path.dirname(os.path.abspath(self.result_dir)))
        if self.files:
            cands.append(result_root(self.out_location, self.files[0]['path']))
        elif self.out_location:
            cands.append(result_root(self.out_location, ''))
        if self.out_default:
            cands.append(result_root(self.out_default, ''))
        for d in cands:
            if d and os.path.isdir(d):
                return d
        return None

    def open_results(self):
        if self.dirty and not self.confirm('確認', '目前的結果有修改還沒儲存，開啟別的結果會放棄這些修改。\n\n確定要繼續嗎？', ok='確定', warn=True):
            return
        p = filedialog.askopenfilename(title='選擇結果資料夾裡的 results.json', initialdir=self._results_start_dir(), filetypes=[('判讀結果', 'results.json'), ('JSON', '*.json')])
        if not p:
            return
        try:
            evs, prof, files, timing = export.load_json(p)
        except Exception as ex:
            messagebox.showerror(APP, '讀取失敗：%s' % ex)
            return
        self.events, self.run_profile, self.run_files, self.run_timing = evs, prof, files, timing
        self.result_dir = os.path.dirname(p)
        self.set_dirty(False)
        self.fill_events()

    def confirm(self, title, msg, ok='確定', cancel='取消', warn=False, icon=None):
        """自訂確認視窗（按鈕文字固定是中文，不受作業系統語言影響）。確定→True，取消／關掉視窗→False。
        cancel=None：只有一個按鈕（通知用）"""
        w = tk.Toplevel(self.root)
        w.title(title)
        w.transient(self.root)
        w.resizable(False, False)
        res = {'v': False}
        body = ttk.Frame(w, padding=(18, 16, 18, 8)); body.pack(fill='both')
        tk.Label(body, text=icon or ('⚠' if warn else '？'), fg='#c05000' if warn else '#1060c0',
                 font=(self.font_family, 22, 'bold')).pack(side='left', anchor='n', padx=(0, 12))
        ttk.Label(body, text=msg, justify='left', wraplength=440).pack(side='left', fill='x')
        bar = ttk.Frame(w, padding=(18, 4, 18, 14)); bar.pack(fill='x')

        def done(v):
            res['v'] = v
            w.destroy()
        b_ok = None
        if cancel:
            b_cancel = ttk.Button(bar, text=cancel, width=max(10, len(cancel) * 2 + 2), command=lambda: done(False))
            b_cancel.pack(side='right')
        b_ok = ttk.Button(bar, text=ok, width=max(10, len(ok) * 2 + 2), command=lambda: done(True))
        b_ok.pack(side='right', padx=(0, 8) if cancel else 0)
        if not cancel:
            b_cancel = b_ok
        w.protocol('WM_DELETE_WINDOW', lambda: done(False))
        w.bind('<Escape>', lambda e: done(False))
        w.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - w.winfo_height()) // 3
        w.geometry('+%d+%d' % (max(0, x), max(0, y)))
        (b_cancel if warn else b_ok).focus_set()      # 有未儲存的修改時，預設按鈕是「取消」
        w.grab_set()
        self.root.wait_window(w)
        return res['v']

    def _stop_preview(self):
        self._pv_token = getattr(self, '_pv_token', 0) + 1

    def on_close(self):
        """關閉前一律確認：確定＝關閉，取消＝不關閉"""
        if self.worker and self.worker.is_alive():
            msg = '判讀還在進行，關閉後這次判讀不會有結果。\n\n確定要關閉嗎？'
        elif self.dirty:
            msg = ('第 2 頁的修改還沒儲存（還沒按「儲存修改並重新輸出 Excel」），關閉後這些修改會遺失。\n\n'
                   '要保留修改，請按「取消」，再按「儲存修改並重新輸出 Excel」。\n\n確定要關閉嗎？')
        elif self.result_dir:
            msg = ('判讀結果已經存在：\n%s\n\n下次可以按第 2 頁的「開啟先前的結果…」，選這個資料夾裡的 results.json 打開。\n\n'
                   '確定要關閉嗎？' % self.result_dir)
        else:
            msg = '確定要關閉程式嗎？'
        warn = bool(self.dirty or (self.worker and self.worker.is_alive()))
        if not self.confirm('確認關閉', msg, ok='確定關閉', cancel='取消', warn=warn):
            return
        if self.worker and self.worker.is_alive():
            self.cancel_flag = True
        self.root.destroy()


def main():
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry('%dx%d' % (min(1320, sw - 40), min(820, sh - 80)))   # 螢幕比較小時一開始就不要超出螢幕
    root.minsize(640, 420)                                           # 再小就用捲軸看（ScrollArea）
    App(root)
    root.mainloop()
