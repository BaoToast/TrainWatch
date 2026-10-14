"""
#105：第 2 頁勾選多筆，一次改為列車／不是列車。需要畫面（沒有畫面的環境自動略過）。
只用虛構的事件資料。
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter as tk
    _root_probe = tk.Tk()
    _root_probe.destroy()
    HAVE_TK = True
except Exception:
    HAVE_TK = False


def _pump(root, seconds):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def _ev(t0, dur, valid, reason=''):
    return {'no': 0, 'start': t0, 'end': t0 + dur, 'valid': valid, 'direction': '往北' if valid else '',
            'train_type': '', 'crossing': False, 'need_check': False, 'checked': False, 'note': '',
            'reason': reason, 'shots': [], 'clip': ''}


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class TestBatchValid(unittest.TestCase):
    def setUp(self):
        from trainwatch import gui, core
        self.gui, self.core = gui, core
        self.root = tk.Tk()
        self.root.geometry('1280x760')
        os.environ['CI'] = 'true'
        self.app = gui.App(self.root)
        self.calls = []
        self._mb = (gui.messagebox.showinfo, gui.messagebox.askyesno)
        gui.messagebox.showinfo = lambda *a, **k: self.calls.append(('info', a[1] if len(a) > 1 else ''))
        gui.messagebox.askyesno = lambda *a, **k: (self.calls.append(('ask', a[1] if len(a) > 1 else '')), True)[1]
        self.dir = tempfile.mkdtemp()
        base = 1767225600.0    # 虛構日期
        evs = [_ev(base + 100, 20, True), _ev(base + 300, 1.2, False, '太短'), _ev(base + 500, 8, False, '變化範圍太小'),
               _ev(base + 700, 15, True), _ev(base + 900, 6, False, '變化範圍太小')]
        core.number_events(evs)
        self.app.events = evs
        self.app.result_dir = self.dir
        self.app.run_files = []
        self.app.run_timing = []
        self.app.nb.select(self.app.page2)
        self.app.fill_events()
        _pump(self.root, 0.3)

    def tearDown(self):
        self.gui.messagebox.showinfo, self.gui.messagebox.askyesno = self._mb
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def _click_pick(self, iid):
        x, y, w, h = self.app.tv.bbox(iid, 'pick')
        ev = type('E', (), {'x': x + w // 2, 'y': y + h // 2})()
        return self.app._on_tv_click(ev)

    def test105a_need_show_excluded(self):
        self.app.pick_all_excluded()
        self.assertTrue(any('顯示已排除' in c[1] for c in self.calls), '沒顯示已排除時要提醒先勾')
        self.assertFalse(any(e.get('_pick') for e in self.app.events))

    def test105b_pick_all_excluded_and_make_valid(self):
        self.app.show_excluded.set(True); self.app.fill_events(); _pump(self.root, 0.2)
        sel0 = self.app.tv.get_children()[0]
        self.app.tv.selection_set(sel0); _pump(self.root, 0.2)
        self.app.pick_all_excluded()
        self.assertEqual(sum(1 for e in self.app.events if e.get('_pick')), 3)
        self.assertEqual(self.app.tv.set(self.app.tv.get_children()[1], 'pick'), '☑')
        self.app.batch_valid(True); _pump(self.root, 0.2)
        self.assertTrue(all(e['valid'] for e in self.app.events), '5 筆都是列車')
        self.assertEqual([e['no'] for e in self.app.events], [1, 2, 3, 4, 5], '依時間重新編號')
        self.assertTrue(all('人工改為列車' in e['note'] for e in self.app.events if e['reason']))
        self.assertFalse(any(e.get('media_busy') for e in self.app.events), '不補做短片')
        self.assertFalse(any(e.get('_pick') for e in self.app.events), '改完清除勾選')
        self.assertEqual(self.app.tv.selection(), (sel0,), '目前看的那一筆不變')
        self.assertTrue(self.app.dirty)
        self.app.save_results(); _pump(self.root, 0.2)
        with open(os.path.join(self.dir, 'results.json'), encoding='utf-8') as fh:
            data = json.load(fh)
        self.assertEqual(sum(1 for e in data['events'] if e['valid']), 5)
        self.assertFalse(any('_pick' in e for e in data['events']), '勾選不存進結果')
        self.assertTrue(os.path.exists(os.path.join(self.dir, '列車通過紀錄.xlsx')))

    def test105c_click_toggles_without_changing_selection(self):
        self.app.show_excluded.set(True); self.app.fill_events(); _pump(self.root, 0.2)
        kids = self.app.tv.get_children()
        self.app.tv.selection_set(kids[0]); _pump(self.root, 0.2)
        self.assertEqual(self._click_pick(kids[2]), 'break')
        self.assertEqual(self.app.tv.set(kids[2], 'pick'), '☑')
        self.assertEqual(self.app.tv.selection(), (kids[0],))
        self._click_pick(kids[2])
        self.assertEqual(self.app.tv.set(kids[2], 'pick'), '☐', '再點一下取消')
        # 只勾一筆改為列車
        self._click_pick(kids[2]); self.app.batch_valid(True); _pump(self.root, 0.2)
        self.assertEqual(sum(1 for e in self.app.events if e['valid']), 3)

    def test105d_make_not_train_and_nothing_picked(self):
        self.app.batch_valid(True)
        self.assertTrue(any('還沒有勾選' in c[1] for c in self.calls))
        kids = self.app.tv.get_children()          # 只顯示列車 2 筆
        self._click_pick(kids[0]); self._click_pick(kids[1])
        self.app.batch_valid(False); _pump(self.root, 0.2)
        self.assertEqual(sum(1 for e in self.app.events if e['valid']), 0)
        self.assertIn('顯示已排除', self.app.var_tab2msg.get())


if __name__ == '__main__':
    unittest.main()
