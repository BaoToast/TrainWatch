"""
#97：程式在背景時跳出的確認視窗。需要畫面（GitHub Actions 的 Windows 有；沒有畫面的環境自動略過）。

python -m unittest discover -s tests -v
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter as tk
    _root_probe = tk.Tk()
    _root_probe.destroy()
    HAVE_TK = True
except Exception:   # 沒有 tkinter 或沒有畫面
    HAVE_TK = False


def _pump(root, seconds):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.02)


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class TestBackgroundPrompt(unittest.TestCase):
    def setUp(self):
        from trainwatch import gui
        self.gui = gui
        self.root = tk.Tk()
        self.root.geometry('900x600')
        os.environ['CI'] = 'true'            # 不要跳出第一次使用的詢問
        self.app = gui.App(self.root)
        self.calls = []
        # 原生訊息框會擋住測試：改成記錄下來
        self._mb = (gui.messagebox.showerror, gui.messagebox.showwarning)
        gui.messagebox.showerror = lambda *a, **k: self.calls.append('showerror')
        gui.messagebox.showwarning = lambda *a, **k: self.calls.append('showwarning')
        _pump(self.root, 0.3)

    def tearDown(self):
        self.gui.messagebox.showerror, self.gui.messagebox.showwarning = self._mb
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _need(self):
        frame = np.full((480, 640, 3), 90, np.uint8)
        return [{'file': 'x.mp4', 'frame': frame, 'verdict': 'none', 'pos': 1.0, 'disp': ''}]

    def test97a_poll_survives_handler_error(self):
        """一則訊息處理失敗（例如確認視窗開不起來），之後的訊息仍要照常處理"""
        def boom(*a, **k):
            raise tk.TclError('grab failed: window "x" was not viewable')
        self.app._app_active = lambda: True      # 自動測試的電腦上程式不一定在前景，固定成在前景
        self.app.confirm = boom
        ev, ans = threading.Event(), {}
        self.app.q.put(('ask_pos', self._need(), self.app.profile, ev, ans))
        _pump(self.root, 0.6)
        self.app.q.put(('prog', 0.5, '進度訊息還有在更新'))
        _pump(self.root, 0.6)
        self.assertEqual(self.app.var_status.get(), '進度訊息還有在更新')
        self.assertTrue(ev.is_set(), '確認失敗也要讓背景判讀知道（不能一直等）')
        self.assertIn('showerror', self.calls, '開不起來要告訴使用者')

    def test97b_defer_until_user_returns(self):
        """程式不在最前面時先不開視窗，顯示提醒列；使用者回來才開"""
        active = {'v': False}
        self.app._app_active = lambda: active['v']
        self.app.confirm = lambda *a, **k: (self.calls.append(a[0]), True)[1]
        ev, ans = threading.Event(), {}
        self.app.q.put(('ask_pos', self._need(), self.app.profile, ev, ans))
        _pump(self.root, 1.0)
        self.assertEqual(self.calls, [], '程式在背景時不可以直接跳出視窗')
        self.assertTrue(self.app.attn_bar.winfo_ismapped(), '要顯示黃色提醒列')
        self.assertIn('確認攝影機位置', self.app.var_attn.get())
        self.assertFalse(ev.is_set())
        active['v'] = True                     # 使用者切回這個程式
        _pump(self.root, 1.0)
        self.assertEqual(self.calls, ['確認攝影機位置'])
        self.assertTrue(ev.is_set())
        self.assertTrue(ans.get('ok'))
        self.assertFalse(self.app.attn_bar.winfo_ismapped(), '處理完提醒列要消失')

    def test97c_button_opens_immediately(self):
        """提醒列上的按鈕：按了就開（人就在程式上）"""
        self.app._app_active = lambda: False
        self.app.confirm = lambda *a, **k: (self.calls.append(a[0]), True)[1]
        ev, ans = threading.Event(), {}
        self.app.q.put(('ask_pos', self._need(), self.app.profile, ev, ans))
        _pump(self.root, 0.6)
        self.assertEqual(self.calls, [])
        self.app.btn_attn.invoke()
        _pump(self.root, 0.3)
        self.assertEqual(self.calls, ['確認攝影機位置'])
        self.assertTrue(ev.is_set())

    def test97d_foreground_shows_at_once(self):
        """程式本來就在最前面：和以前一樣立刻跳出"""
        self.app._app_active = lambda: True
        self.app.confirm = lambda *a, **k: (self.calls.append(a[0]), True)[1]
        ev, ans = threading.Event(), {}
        self.app.q.put(('ask_pos', self._need(), self.app.profile, ev, ans))
        _pump(self.root, 0.5)
        self.assertEqual(self.calls, ['確認攝影機位置'])
        self.assertFalse(self.app.attn_bar.winfo_ismapped())

    def test97e_summary_and_stop_also_wait(self):
        """判讀完成摘要、沒有開始判讀的訊息，在背景時也一樣等使用者回來"""
        active = {'v': False}
        self.app._app_active = lambda: active['v']
        self.app.show_summary = lambda *a: self.calls.append('判讀完成')
        self.app.q.put(('done', [], '', [], [], self.app.profile, False))
        _pump(self.root, 1.0)
        self.assertEqual(self.calls, [])
        self.assertIn('判讀完成', self.app.var_attn.get())
        active['v'] = True
        _pump(self.root, 1.0)
        self.assertEqual(self.calls, ['判讀完成'])

    def test97f_confirm_retries_grab(self):
        """確認視窗還沒顯示出來時 grab_set 失敗，不可以讓確認流程中斷"""
        orig = tk.Toplevel.grab_set
        n = {'fail': 2}

        def flaky(w):
            if n['fail'] > 0:
                n['fail'] -= 1
                raise tk.TclError('grab failed: window not viewable')
            return orig(w)
        tk.Toplevel.grab_set = flaky
        try:
            def close_later():
                for w in self.root.winfo_children():
                    if isinstance(w, tk.Toplevel):
                        w.event_generate('<Escape>')
            self.root.after(900, close_later)
            r = self.app.confirm('測試', '內容', ok='好', cancel='取消')
        finally:
            tk.Toplevel.grab_set = orig
        self.assertFalse(r)
        self.assertEqual(n['fail'], 0, 'grab 失敗後要再試')


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class TestRunLock(unittest.TestCase):
    """#98 判讀中鎖住第 1 頁；#99 位置基準只記進這次判讀用的監測站"""
    def setUp(self):
        from trainwatch import gui
        self.gui = gui
        self.root = tk.Tk()
        os.environ['CI'] = 'true'
        self.app = gui.App(self.root)
        _pump(self.root, 0.3)

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _buttons(self, text):
        out = []
        def walk(w):
            for c in w.winfo_children():
                try:
                    if str(c.cget('text')) == text:
                        out.append(c)
                except tk.TclError:
                    pass
                walk(c)
        walk(self.root)
        return out

    def test98a_lock_and_unlock(self):
        add = self._buttons('加入影片…')[0]
        clear = self._buttons('清空')[0]
        self.assertEqual(str(add.cget('state')), 'normal')
        self.app.btn_run.configure(state='disabled')
        self.app._set_run_lock(True)
        _pump(self.root, 0.2)
        for w in (add, clear, self.app.cb_prof):
            self.assertEqual(str(w.cget('state')), 'disabled')
        self.assertTrue(self.app.lbl_lock.winfo_ismapped(), '要顯示判讀中不能修改的說明')
        self.app._run_finished()
        _pump(self.root, 0.2)
        self.assertEqual(str(add.cget('state')), 'normal')
        self.assertEqual(str(self.app.cb_prof.cget('state')), 'readonly', '下拉選單要恢復成原本的唯讀狀態')
        self.assertFalse(self.app.lbl_lock.winfo_ismapped())

    def test98b_done_message_unlocks(self):
        self.app._set_run_lock(True)
        self.app._app_active = lambda: False      # 摘要視窗排隊就好，不要真的開
        self.app.q.put(('done', [], '', [], [], self.app.profile, False))
        _pump(self.root, 0.6)
        self.assertEqual(str(self._buttons('加入影片…')[0].cget('state')), 'normal')

    def test98c_canvas_drag_ignored_while_running(self):
        self.app.cur_frame = np.zeros((270, 480, 3), np.uint8)
        before = [list(p) for p in self.app.profile.ref_line]
        self.app._set_run_lock(True)

        class E:
            x, y = 10, 10
        self.app.on_press(E())
        E.x, E.y = 200, 200
        self.app.on_drag(E())
        self.assertEqual([list(p) for p in self.app.profile.ref_line], before)

    def test99_anchors_not_saved_to_other_station(self):
        self.app._run_prof_name = '甲站'
        self.app.profile.name = '乙站'
        old = list(self.app.profile.camera_anchors or [])
        self.app.q.put(('anchors', [{'note': '甲站確認的畫面'}]))
        _pump(self.root, 0.5)
        self.assertEqual(list(self.app.profile.camera_anchors or []), old, '甲站確認的位置不可以寫進乙站')


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class TestDoneWithUnsaved(unittest.TestCase):
    """#100 判讀完成時第 2 頁還有沒儲存的修改：先問，不要默默蓋掉"""
    def setUp(self):
        from trainwatch import gui
        self.gui = gui
        self.root = tk.Tk()
        os.environ['CI'] = 'true'
        self.app = gui.App(self.root)
        self.app._app_active = lambda: True
        self.app.show_summary = lambda *a: None
        self.asked = []
        _pump(self.root, 0.3)

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _done(self):
        self.app.events = [{'no': 1, 'valid': True, 'start': 0, 'end': 1, '舊的': True}]
        self.app.set_dirty(True)
        new = [{'no': 1, 'valid': True, 'start': 5, 'end': 9}]
        self.app.q.put(('done', new, '', [], [], self.app.profile, False))
        return new

    def test100a_ask_then_discard(self):
        self.app.confirm = lambda title, *a, **k: (self.asked.append(title), True)[1]   # 按「不儲存」
        new = self._done()
        _pump(self.root, 0.8)
        self.assertIn('第 2 頁的修改還沒儲存', self.asked)
        self.assertIs(self.app.events, new)
        self.assertFalse(self.app.dirty)

    def test100b_ask_then_save(self):
        self.app.confirm = lambda title, *a, **k: (self.asked.append(title), False)[1]  # 按「儲存」（也是 Esc／關視窗）
        saved = []
        self.app.save_results = lambda: (saved.append(list(self.app.events)), self.app.set_dirty(False))
        new = self._done()
        _pump(self.root, 0.8)
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0][0].get('舊的'), '要存的是上一次的結果')
        self.assertIs(self.app.events, new)
