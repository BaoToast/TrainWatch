"""
v1.0.21：#120 拖曳加入影片、#122／#123 第 2 頁篩選、#124 只有已排除的做截圖、#125 拆成兩筆、#126 參考線靠近範圍一端。
只用虛構的事件與合成影片。
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainwatch import core  # noqa: E402

try:
    import tkinter as tk
    _r = tk.Tk()
    _r.destroy()
    HAVE_TK = True
except Exception:
    HAVE_TK = False

BASE = 1767225600.0      # 虛構日期


def _ev(t0, dur, valid=True, reason='', need=False, items=(), note=''):
    return {'no': 0, 'start': BASE + t0, 'end': BASE + t0 + dur, 'valid': valid, 'direction': '往右' if valid else '',
            'train_type': '', 'crossing': False, 'need_check': need, 'checked': False, 'note': note,
            'reason': reason, 'shots': [], 'clip': '', 'check_items': list(items),
            'start_file': 'v.mkv', 'end_file': 'v.mkv', 'start_pos': t0, 'end_pos': t0 + dur}


class Geometry(unittest.TestCase):
    def test_reference_near_one_end_allowed(self):
        # 細長的斜向範圍，參考線靠近一端（使用者實際的畫法，#126）
        p = core.Profile(track_quad=[[513, 146], [556, 150], [391, 285], [326, 243]], ref_line=[[355, 239], [401, 270]])
        d = core.Detector(p, core.work_crop(p, 640, 360))
        self.assertTrue(d.short_side)
        p2 = core.Profile(track_rect=[100, 140, 540, 220], ref_line=[[320, 140], [320, 220]])
        self.assertFalse(core.Detector(p2, core.work_crop(p2, 640, 360)).short_side)

    def test_short_side_train_not_excluded(self):
        import cv2
        import numpy as np
        p = core.Profile(track_rect=[100, 140, 540, 220], ref_line=[[530, 140], [530, 220]])   # 參考線離右邊只有 10
        crop = core.work_crop(p, 640, 360)
        det = core.Detector(p, crop)
        self.assertTrue(det.short_side)
        x0, y0, x1, y1 = crop
        det.bg = np.full((y1 - y0, x1 - x0), 100, np.float32)
        for i in range(141):
            t = i / 10
            im = np.full((360, 640), 100, np.uint8)
            head = int(530 + (t - 5) * 80)
            cv2.rectangle(im, (head - 200, 146), (head, 214), 230, -1)
            det.feed(t, im[y0:y1, x0:x1], 0, 'synthetic', t)
        ev = [e for e in det.finish() if e['valid']]
        self.assertEqual(len(ev), 1, ev)
        self.assertLess(abs(ev[0]['start'] - 5), 0.3)


class Dropped(unittest.TestCase):
    def test_expand_dropped(self):
        from trainwatch import gui
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, '9月10日'))
            for n in ('b.mkv', 'a.MP4', 'x.txt', os.path.join('9月10日', 'c.mkv')):
                open(os.path.join(d, n), 'w').close()
            vids, skipped = gui.expand_dropped([os.path.join(d, 'x.txt'), os.path.join(d, 'b.mkv')])
            self.assertEqual([os.path.basename(v) for v in vids], ['b.mkv'])
            self.assertEqual(skipped, 1)
            vids, skipped = gui.expand_dropped([d])                 # 整個資料夾（含子資料夾）
            self.assertEqual(sorted(os.path.basename(v) for v in vids), ['a.MP4', 'b.mkv', 'c.mkv'])
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class Page2(unittest.TestCase):
    def setUp(self):
        from trainwatch import gui
        self.gui = gui
        self.root = tk.Tk()
        self.root.geometry('1366x768')
        os.environ['CI'] = 'true'
        self.app = gui.App(self.root)
        self.calls = []
        self._mb = (gui.messagebox.showinfo, gui.messagebox.askyesno)
        gui.messagebox.showinfo = lambda *a, **k: self.calls.append(a[1] if len(a) > 1 else '')
        gui.messagebox.askyesno = lambda *a, **k: True
        evs = [_ev(100, 3), _ev(300, 12, need=True, items=['方向不確定']), _ev(500, 0.2, False, '時間太短（0.2 秒，可能是閃爍）'),
               _ev(700, 4, False, '軌道範圍內變化範圍太小（10%，可能是汽車、行人或樹葉）'), _ev(900, 0.1, False, '時間太短（0.1 秒）'),
               _ev(1100, 5, False, '方向不確定', note='人工改為非列車'), _ev(1300, 15)]
        core.number_events(evs)
        self.app.events = evs
        self.app.result_dir = tempfile.mkdtemp()
        self.app.run_files = ['v.mkv']
        self.app.run_timing = [{'offset': BASE}]
        self.app.fill_events()
        self.root.update()

    def tearDown(self):
        self.gui.messagebox.showinfo, self.gui.messagebox.askyesno = self._mb
        shutil.rmtree(self.app.result_dir, ignore_errors=True)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _shown(self):
        return [self.app.events[int(k)] for k in self.app.tv.get_children()]

    def _pick(self, prefix):
        a = self.app
        lab = next(v for v in a.cb_filter.cget('values') if v.startswith(prefix))
        a.var_filter.set(lab); a._on_filter()

    def test_filters(self):
        a = self.app
        self.assertEqual(len(self._shown()), 3)                      # 全部（沒勾顯示已排除）：3 筆列車
        self._pick('只看已排除（4）')
        self.assertEqual(len(self._shown()), 4)
        self._pick('已排除｜時間太短（閃爍）（2）')
        self.assertEqual([round(e['end'] - e['start'], 1) for e in self._shown()], [0.2, 0.1])
        self._pick('已排除｜人工改為不是列車（1）')
        self.assertEqual(len(self._shown()), 1)
        self._pick('需確認｜方向不確定（1）')
        self.assertEqual(len(self._shown()), 1)
        self._pick('全部')
        a.var_long.set(True); a.var_long_sec.set('10'); a.fill_events()
        self.assertEqual(sorted(round(e['end'] - e['start']) for e in self._shown()), [12, 15])
        self.assertIn('顯示 2 筆', a.var_sum.get())

    def test_pick_all_excluded_with_filter(self):
        a = self.app
        self._pick('只看已排除')
        a.pick_all_excluded()
        self.assertEqual(sum(1 for e in a.events if e.get('_pick')), 4)

    def test_split_event(self):
        a = self.app
        e = next(x for x in a.events if round(x['end'] - x['start']) == 12)
        e['shots'] = ['002_1車頭.jpg', '002_2中間.jpg', '002_3車尾.jpg']
        e['clip'] = '002.mp4'
        a._locate = lambda ts: ('v.mkv', ts - BASE)
        n0 = len(a.events)
        a._do_split(e, BASE + 304, BASE + 306.5, ('v.mkv', 304), ('v.mkv', 306.5))
        self.assertEqual(len(a.events), n0 + 1)
        two = sorted([x for x in a.events if '人工拆成兩筆' in x.get('note', '')], key=lambda x: x['start'])
        self.assertEqual([(x['start'] - BASE, x['end'] - BASE) for x in two], [(300, 304), (306.5, 312)])
        self.assertEqual(two[0]['shots'], ['002_1車頭.jpg'])
        self.assertEqual(two[1]['shots'], ['002_3車尾.jpg'])
        self.assertEqual(two[0]['clip'], two[1]['clip'])
        self.assertNotEqual(two[0].get('media_tag'), two[1].get('media_tag'))
        self.assertTrue(a.dirty)
        nos = [x['no'] for x in a.events if x['valid']]
        self.assertEqual(nos, sorted(nos))                          # 依時間重新編號


class ExcludedOnlyShots(unittest.TestCase):
    def test_only_excluded_get_shots(self):
        from trainwatch import selftest
        d = tempfile.mkdtemp()
        try:
            f = os.path.join(d, 'v.avi')
            selftest.make_video(f, seconds=12, trains=((5, 1),))
            p = core.Profile(make_clips=False, make_shots=True, shots_excluded_only=True)
            evs = [dict(no=1, valid=True, start=5.0, end=7.0, start_file=f, end_file=f, start_pos=5.0, end_pos=7.0),
                   dict(no=1, valid=False, start=9.0, end=9.3, start_file=f, end_file=f, start_pos=9.0, end_pos=9.3)]
            core.save_frames_and_clips(evs, p, d, files=[f], bases=[0.0])
            self.assertEqual(evs[0]['shots'], [])
            self.assertEqual(len(evs[1]['shots']), 3)
            self.assertEqual(evs[0]['clip'], '')
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(sys.platform.startswith('win') and HAVE_TK, '只在 Windows 測（GitHub 打包時會跑）')
class WindowsDrop(unittest.TestCase):
    def test_wm_dropfiles(self):
        """模擬從檔案總管拖檔案進來：送一個真的 WM_DROPFILES 給視窗"""
        import ctypes
        from ctypes import wintypes
        from trainwatch import gui
        root = tk.Tk()
        try:
            got = []
            self.assertTrue(gui.enable_file_drop(root, got.extend))
            names = ['C:\\影片\\a.mkv', 'C:\\影片\\9月10日\\b.mkv']
            text = ('\0'.join(names) + '\0\0').encode('utf-16-le')
            hdr = 20                                   # DROPFILES：pFiles, pt.x, pt.y, fNC, fWide
            k32 = ctypes.windll.kernel32
            k32.GlobalAlloc.restype = ctypes.c_void_p
            k32.GlobalLock.restype = ctypes.c_void_p
            k32.GlobalLock.argtypes = [ctypes.c_void_p]
            k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            h = k32.GlobalAlloc(0x0042, hdr + len(text))
            ptr = k32.GlobalLock(h)
            ctypes.memmove(ptr, (ctypes.c_uint32 * 5)(hdr, 0, 0, 0, 1), hdr)
            ctypes.memmove(ptr + hdr, text, len(text))
            k32.GlobalUnlock(h)
            u = ctypes.windll.user32
            u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            u.SendMessageW(root.winfo_id(), 0x0233, h, 0)
            end = time.time() + 2
            while time.time() < end and not got:
                root.update(); time.sleep(0.02)
            self.assertEqual(got, names)
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
