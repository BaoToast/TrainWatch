"""列車通過判讀：視窗介面（tkinter）"""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

from PIL import Image, ImageTk

from . import core, export, osd
from .core import Profile

VERSION = '1.0.3'
APP = '列車通過判讀'
VIDEO_TYPES = [('影片', '*.mkv *.mp4 *.avi *.mov *.ts *.h264 *.264 *.dav'), ('所有檔案', '*.*')]

CANVAS_W, CANVAS_H = 900, 506


def app_dir():
    """程式的最上層資料夾（「啟動列車通過判讀.bat」所在處），監測站設定存在這裡"""
    home = os.environ.get('TRAINWATCH_HOME')
    if home and os.path.isdir(home):
        return os.path.abspath(home)
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        self.prof_dir = os.path.join(app_dir(), '監測站設定')
        try:
            os.makedirs(self.prof_dir, exist_ok=True)
        except OSError:      # 程式放在不能寫入的地方（例如 Program Files）→ 改存到使用者資料夾
            self.prof_dir = os.path.join(os.path.expanduser('~'), '列車通過判讀_監測站設定')
            os.makedirs(self.prof_dir, exist_ok=True)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=6, pady=6)
        self.tab1 = ttk.Frame(self.nb)
        self.tab2 = ttk.Frame(self.nb)
        self.nb.add(self.tab1, text='  1. 影片與參考線  ')
        self.nb.add(self.tab2, text='  2. 檢視與修改結果  ')
        self._build_tab1()
        self._build_tab2()
        self._load_last_profile()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(150, self._poll)

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
        left = ttk.Frame(t, width=360)
        left.pack(side='left', fill='y', padx=(4, 8), pady=4)
        left.pack_propagate(False)
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

        box = ttk.LabelFrame(left, text='影片（同一台攝影機）')
        box.pack(fill='both', expand=True, pady=(0, 6))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Button(r, text='加入影片…', command=self.add_files).pack(side='left')
        ttk.Button(r, text='移除', width=5, command=self.remove_files).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='清空', width=5, command=self.clear_files).pack(side='left', padx=(4, 0))
        ttk.Button(r, text='依時間排序', command=self.sort_files).pack(side='left', padx=(4, 0))
        fr = ttk.Frame(box); fr.pack(fill='both', expand=True, padx=6, pady=2)
        self.tv_files = ttk.Treeview(fr, columns=('name', 'start'), show='headings', selectmode='extended', height=8)
        self.tv_files.heading('name', text='檔名'); self.tv_files.heading('start', text='畫面上的開始時間')
        self.tv_files.column('name', width=150, anchor='w'); self.tv_files.column('start', width=160, anchor='center')
        sb = ttk.Scrollbar(fr, orient='vertical', command=self.tv_files.yview)
        self.tv_files.configure(yscrollcommand=sb.set)
        self.tv_files.pack(side='left', fill='both', expand=True); sb.pack(side='left', fill='y')
        self.tv_files.tag_configure('bad', foreground='#b00000')
        self.tv_files.tag_configure('manual', foreground='#0050a0')
        self.tv_files.bind('<<TreeviewSelect>>', lambda e: self.on_file_select())
        ttk.Label(box, style='Hint.TLabel', wraplength=330, justify='left', text=(
            '開始時間是讀畫面上的時間字幕，不看檔名。讀不到（紅字）時請選取該檔，'
            '在下面填畫面時間；藍字＝手動填的。')).pack(fill='x', padx=6)
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(4, 2))
        ttk.Label(r, text='手動開始時間').pack(side='left')
        self.var_ftime = tk.StringVar()
        ttk.Entry(r, textvariable=self.var_ftime).pack(side='left', fill='x', expand=True, padx=(6, 0))
        r = ttk.Frame(box); r.pack(fill='x', padx=6, pady=(2, 6))
        ttk.Label(r, text='例：2026-01-01 08:00:00', style='Hint.TLabel').pack(side='left')
        ttk.Button(r, text='清除', width=5, command=self.clear_file_time).pack(side='right')
        ttk.Button(r, text='套用', width=5, command=self.set_file_time).pack(side='right', padx=(0, 4))

        box = ttk.LabelFrame(left, text='判讀')
        box.pack(fill='x')
        self.btn_run = ttk.Button(box, text='開始判讀', style='Big.TButton', command=self.start_run)
        self.btn_run.pack(fill='x', padx=6, pady=(8, 4))
        self.btn_cancel = ttk.Button(box, text='取消', command=self.cancel_run, state='disabled')
        self.btn_cancel.pack(fill='x', padx=6)
        self.pb = ttk.Progressbar(box, maximum=1000)
        self.pb.pack(fill='x', padx=6, pady=(6, 2))
        self.var_status = tk.StringVar(value='請先加入影片，再在右邊畫參考線與軌道範圍。')
        ttk.Label(box, textvariable=self.var_status, wraplength=330, style='Hint.TLabel').pack(fill='x', padx=6, pady=(0, 8))

        r = ttk.Frame(right); r.pack(fill='x')
        ttk.Label(r, text='在畫面上拖曳：').pack(side='left')
        ttk.Radiobutton(r, text='參考線（紅）', value='ref', variable=self.mode).pack(side='left', padx=3)
        ttk.Radiobutton(r, text='軌道範圍（橘）', value='track', variable=self.mode).pack(side='left', padx=3)
        ttk.Radiobutton(r, text='時間字幕位置（藍）', value='osd', variable=self.mode).pack(side='left', padx=3)
        ttk.Label(r, text='線寬').pack(side='left', padx=(10, 2))
        self.var_width = tk.IntVar(value=self.profile.ref_width)
        sp = ttk.Spinbox(r, from_=3, to=25, width=4, textvariable=self.var_width, command=self.redraw_overlay)
        sp.pack(side='left')
        sp.bind('<KeyRelease>', lambda e: self.redraw_overlay())

        self.canvas = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg='#202020', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(pady=(6, 4), anchor='w')
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        r = ttk.Frame(right); r.pack(fill='x')
        self.var_pos = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(r, from_=0, to=600, variable=self.var_pos, command=lambda v: self.schedule_seek())
        self.scale.pack(side='left', fill='x', expand=True)
        self.var_postxt = tk.StringVar(value='影片內 --:--')
        ttk.Label(r, textvariable=self.var_postxt, width=14, anchor='e').pack(side='left', padx=(8, 0))
        r = ttk.Frame(right); r.pack(fill='x', pady=(2, 0))
        for lab, d in (('−10 秒', -10), ('−1 秒', -1), ('+1 秒', 1), ('+10 秒', 10)):
            ttk.Button(r, text=lab, width=7, command=lambda d=d: self.step(d)).pack(side='left', padx=(0, 4))
        ttk.Label(r, text='　畫面時間判讀：').pack(side='left')
        self.var_osd = tk.StringVar(value='')
        self.lbl_osd = ttk.Label(r, textvariable=self.var_osd, style='Ok.TLabel')
        self.lbl_osd.pack(side='left')
        ttk.Button(r, text='教程式認字…', command=self.teach_osd).pack(side='right')
        hints = ttk.Frame(right); hints.pack(fill='x', pady=(8, 0))
        for title, color, text in (
                ('參考線', '#c00000', '畫一條橫過軌道的短線。車頭碰到線＝到達時間，車尾離開線＝離開時間。不用剛好畫到車身高度，蓋住兩條軌道即可。'),
                ('軌道範圍', '#b06000', '沿著軌道框一段，參考線左右都要留一段。用來分辨列車（佔滿一長段軌道、橫跨參考線）和汽車、行人、車燈，請盡量避開馬路。'),
                ('時間字幕位置', '#1060c0', '框住畫面上的日期時間字幕（第一個字的左邊到最後一個字的右邊）。下方「畫面時間判讀」和畫面上一樣就表示讀得到；不一樣請調整藍框，或按「教程式認字」。')):
            row = ttk.Frame(hints); row.pack(fill='x', pady=1)
            tk.Label(row, text=title, fg=color, font=(self.font_family, 10, 'bold'), width=12, anchor='nw').pack(side='left', anchor='n')
            ttk.Label(row, text=text, style='Hint.TLabel', wraplength=780, justify='left').pack(side='left', fill='x')

    # ------------------------------------------------------------ 分頁 2
    def _build_tab2(self):
        t = self.tab2
        top = ttk.Frame(t); top.pack(fill='x', padx=4, pady=(4, 4))
        ttk.Button(top, text='開啟先前的結果…', command=self.open_results).pack(side='left')
        ttk.Checkbutton(top, text='顯示已排除（非列車／閃爍）', variable=self.show_excluded,
                        command=self.fill_events).pack(side='left', padx=12)
        self.var_sum = tk.StringVar(value='')
        ttk.Label(top, textvariable=self.var_sum).pack(side='left', padx=8)
        ttk.Button(top, text='開啟結果資料夾', command=lambda: self.result_dir and open_path(self.result_dir)).pack(side='right')
        ttk.Button(top, text='儲存修改並重新輸出 Excel', style='Big.TButton', command=self.save_results).pack(side='right', padx=6)

        body = ttk.Frame(t); body.pack(fill='both', expand=True, padx=4)
        rf = ttk.Frame(body, width=500); rf.pack(side='right', fill='y', padx=(8, 0))
        rf.pack_propagate(False)
        lf = ttk.Frame(body); lf.pack(side='left', fill='both', expand=True)
        cols = ('no', 'date', 'start', 'end', 'dur', 'dir', 'type', 'cross', 'check', 'reason')
        heads = ('編號', '日期', '車頭到達', '車尾離開', '秒數', '方向', '車種', '交會', '需確認', '備註')
        widths = (46, 56, 86, 86, 46, 70, 100, 44, 56, 240)
        self.tv = ttk.Treeview(lf, columns=cols, show='headings', selectmode='browse')
        for c, h, w in zip(cols, heads, widths):
            self.tv.heading(c, text=h)
            self.tv.column(c, width=w, anchor='w' if c in ('reason', 'type') else 'center', stretch=(c == 'reason'))
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side='left', fill='both', expand=True); sb.pack(side='left', fill='y')
        self.tv.tag_configure('need', background='#fff2cc')
        self.tv.tag_configure('excl', foreground='#666666')
        self.tv.bind('<<TreeviewSelect>>', lambda e: self.on_event_select())

        self.shot_canvas = tk.Canvas(rf, width=480, height=270, bg='#202020', highlightthickness=0)
        self.shot_canvas.pack(pady=(0, 4))
        r = ttk.Frame(rf); r.pack(fill='x')
        for i, lab in enumerate(('車頭', '中間', '車尾')):
            ttk.Button(r, text=lab, width=6, command=lambda i=i: self.show_shot(i)).pack(side='left', padx=(0, 4))
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
                     '看過後勾「已人工確認」，再按「套用修改」。',
                     '時間格式：時:分:秒，例如「08:15:30.5」。',
                     '修改完記得按上方「儲存修改並重新輸出 Excel」。'):
            ttk.Label(rf, style='Hint.TLabel', text=line).pack(fill='x', pady=(2, 0))

    # ------------------------------------------------------------ 監測站設定
    def _profile_names(self):
        try:
            return sorted(f[:-5] for f in os.listdir(self.prof_dir) if f.lower().endswith('.json'))
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
                 ('static_sec', '畫面靜止幾秒就結束（秒）', '列車經過後攝影機亮度跳動用'),
                 ('long_event', '很長的佔用（秒）', '超過標記需確認，並重新學習背景'),
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
        r = ttk.Frame(w); r.grid(row=n + 4, column=0, columnspan=3, pady=(8, 10))
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
        new = []
        for p in paths:
            if os.path.normcase(p) in have:
                continue
            info = core.video_info(p)
            if info is None:
                messagebox.showwarning(APP, '無法開啟：%s' % os.path.basename(p))
                continue
            new.append(dict(path=p, info=info, osd=None, manual=None, state='讀取畫面時間中…'))
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

    def refresh_files(self):
        sel = self.tv_files.selection()
        self.tv_files.delete(*self.tv_files.get_children())
        for i, f in enumerate(self.files):
            tag = ()
            if f.get('manual') is not None:
                txt = f['manual'].strftime('%Y-%m-%d %H:%M:%S') + '（手動）'
                tag = ('manual',)
            elif f.get('osd') is None:
                txt = f.get('state') or ''
            elif f['osd'].get('ok'):
                txt = core.fmt_time(f['osd']['offset'], True)[:-2]
            else:
                txt = '讀不到，請手動填'
                tag = ('bad',)
            self.tv_files.insert('', 'end', iid=str(i), values=(os.path.basename(f['path']), txt), tags=tag)
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
        dur = f['info'].get('dur') or 600
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
        self._seek_job = None
        f = self._sel_file()
        if not f:
            return
        fr, pos = core.read_frame_at(f['path'], self.var_pos.get())
        if fr is None:
            return
        self.cur_frame, self.cur_pos = fr, pos
        h, w = fr.shape[:2]
        self.disp_scale = min(CANVAS_W / w, CANVAS_H / h)
        img = Image.fromarray(fr[:, :, ::-1]).resize((int(w * self.disp_scale), int(h * self.disp_scale)), Image.BILINEAR)
        self.frame_img = ImageTk.PhotoImage(img)
        self.canvas.delete('frame')
        self.canvas.create_image(0, 0, image=self.frame_img, anchor='nw', tags='frame')
        self.canvas.tag_lower('frame')
        self.var_postxt.set('影片內 %s' % export.fmt_pos(pos))
        self.redraw_overlay()
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
        prof = self._collect_profile()
        files = [f['path'] for f in self.files]
        manual = [f.get('manual') for f in self.files]
        base = os.path.dirname(files[0])
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        safe = ''.join('_' if c in '\\/:*?"<>|' else c for c in prof.name)
        out = os.path.join(base, '判讀結果_%s_%s' % (safe, stamp))
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            messagebox.showerror(APP, '無法在影片資料夾建立結果資料夾：%s' % e)
            return
        self.cancel_flag = False
        self.btn_run.configure(state='disabled'); self.btn_cancel.configure(state='normal')
        self.pb['value'] = 0
        prof_copy = Profile.from_dict(json.loads(prof.to_json()))
        self.worker = threading.Thread(target=self._work, args=(files, manual, prof_copy, out), daemon=True)
        self.worker.start()

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
            bases = [t['offset'] for t in timing]
            prog = lambda fr, msg: self.q.put(('prog', 0.08 + fr * 0.8, '判讀中：' + msg))
            evs = core.process(files, bases, prof, prog, lambda: self.cancel_flag)
            core.number_events(evs)
            prog2 = lambda fr, msg: self.q.put(('prog', 0.88 + fr * 0.12, msg))
            core.save_frames_and_clips(evs, prof, out, prog2, files=files)
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
                    self.var_status.set(m[2])
                elif kind == 'scan':
                    _, f, r = m
                    f['osd'] = r
                    f['state'] = ''
                    self.refresh_files()
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
                    self.var_status.set('%s完成：列車 %d 筆、已排除 %d 筆。結果在影片旁的「%s」資料夾。' % (
                        '已取消，部分' if cancelled else '', nv, len(evs) - nv, os.path.basename(out)))
                    self.fill_events()
                    self.nb.select(self.tab2)
                elif kind == 'stop':
                    self._run_finished()
                    self._remove_empty(m[2])
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
        self.show_shot(0)

    def show_shot(self, k):
        i, e = self._sel_event()
        self.shot_canvas.delete('all')
        if e is None:
            return
        shots = e.get('shots') or []
        want = ('1車頭', '2中間', '3車尾')[k]
        name = next((s for s in shots if want in s), None)
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
        if not e.get('clip'):
            messagebox.showinfo(APP, '這一筆沒有短片（可在「進階」開啟產生短片）。')
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
                self.nb.select(self.tab1)
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
        try:
            ns = self._parse_event_time(self.var_es.get(), e['start'])
            ne = self._parse_event_time(self.var_ee.get(), e['end'])
        except ValueError as ex:
            messagebox.showerror(APP, '時間格式不對：%s' % ex)
            return
        if ne < ns:
            messagebox.showerror(APP, '車尾離開不能早於車頭到達。')
            return
        e['start'], e['end'] = ns, ne
        e['direction'] = self.var_dir.get()
        e['train_type'] = self.var_type.get().strip()
        e['crossing'] = bool(self.var_cross.get())
        e['checked'] = bool(self.var_checked.get())
        e['note'] = self.var_note.get().strip()
        self._refresh_keep(e)

    def toggle_valid(self):
        i, e = self._sel_event()
        if e is None:
            return
        e['valid'] = not e['valid']
        tag = '人工改為列車' if e['valid'] else '人工改為非列車'
        note = e.get('note') or ''
        if tag not in note:
            e['note'] = (note + '；' + tag).strip('；')
        if not e['valid']:
            self.show_excluded.set(True)       # 讓剛改的那筆還看得到
        self._refresh_keep(e)

    def set_dirty(self, v):
        self.dirty = v
        self.root.title('%s v%s%s' % (APP, VERSION, '　（有修改還沒儲存）' if v else ''))

    def _refresh_keep(self, e):
        self.set_dirty(True)
        core.number_events(self.events)        # 依時間重新排序、編號
        i = next(k for k, x in enumerate(self.events) if x is e)
        self.fill_events()
        if self.tv.exists(str(i)):
            self.tv.selection_set(str(i)); self.tv.see(str(i))

    def save_results(self):
        if not self.result_dir:
            messagebox.showinfo(APP, '還沒有結果。')
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

    def open_results(self):
        if self.dirty and not self.confirm('確認', '目前的結果有修改還沒儲存，開啟別的結果會放棄這些修改。\n\n確定要繼續嗎？', ok='確定', warn=True):
            return
        p = filedialog.askopenfilename(title='選擇結果資料夾裡的 results.json', filetypes=[('判讀結果', 'results.json'), ('JSON', '*.json')])
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

    def confirm(self, title, msg, ok='確定', cancel='取消', warn=False):
        """自訂確認視窗（按鈕文字固定是中文，不受作業系統語言影響）。確定→True，取消／關掉視窗→False"""
        w = tk.Toplevel(self.root)
        w.title(title)
        w.transient(self.root)
        w.resizable(False, False)
        res = {'v': False}
        body = ttk.Frame(w, padding=(18, 16, 18, 8)); body.pack(fill='both')
        tk.Label(body, text='⚠' if warn else '？', fg='#c05000' if warn else '#1060c0',
                 font=(self.font_family, 22, 'bold')).pack(side='left', anchor='n', padx=(0, 12))
        ttk.Label(body, text=msg, justify='left', wraplength=440).pack(side='left', fill='x')
        bar = ttk.Frame(w, padding=(18, 4, 18, 14)); bar.pack(fill='x')

        def done(v):
            res['v'] = v
            w.destroy()
        b_cancel = ttk.Button(bar, text=cancel, width=10, command=lambda: done(False))
        b_cancel.pack(side='right')
        b_ok = ttk.Button(bar, text=ok, width=10, command=lambda: done(True))
        b_ok.pack(side='right', padx=(0, 8))
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
    root.geometry('1320x820')
    root.minsize(1200, 760)
    App(root)
    root.mainloop()
