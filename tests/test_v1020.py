"""
v1.0.20：#114 軌道範圍任意四邊形（拖曳四個角）、#115 參考線被擋住一部分時自動判斷、
#116「如果列車有可能長停車，請打勾」、v1.0.19 設定的轉換。只用虛構的合成畫面。
"""
import json
import os
import sys
import time
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainwatch import core  # noqa: E402

try:
    import tkinter as tk
    _r = tk.Tk()
    _r.destroy()
    HAVE_TK = True
except Exception:
    HAVE_TK = False

QUAD = [[120, 250], [470, 90], [560, 170], [200, 330]]      # 斜向、近寬遠窄的軌道


def _homography(quad, W, H):
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    return cv2.getPerspectiveTransform(src, np.float32(quad))


def _to_img(M, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), M).reshape(-1, 2)


class QuadGeometry(unittest.TestCase):
    def test_quad_points_axes_and_direction(self):
        p = core.Profile(track_quad=QUAD)
        self.assertEqual(p.track_points().tolist(), np.float32(QUAD).tolist())
        self.assertTrue(p.horizontal())
        self.assertFalse(p.track_is_rect())
        self.assertEqual(p.quad_problem(), '')
        # 沿著四邊形的「寬」方向是往右上：主要是往右
        self.assertEqual(p.direction_label(1), '往右')
        self.assertEqual(p.direction_label(-1), '往左')
        b = p.track_bounds()
        self.assertEqual(b, [120, 90, 560, 330])

    def test_bad_quads_rejected(self):
        bow = core.Profile(track_quad=[[100, 100], [400, 100], [100, 200], [400, 200]])      # 交叉（蝴蝶結）
        self.assertIn('交叉', bow.quad_problem())
        concave = core.Profile(track_quad=[[100, 100], [400, 100], [250, 130], [100, 200]])  # 往內凹
        self.assertIn('凹', concave.quad_problem())

    def test_axis_aligned_quad_uses_plain_rectangle(self):
        p = core.Profile(track_quad=[[100, 140], [540, 140], [540, 220], [100, 220]], ref_line=[[320, 140], [320, 220]])
        det = core.Detector(p, core.work_crop(p, 640, 360))
        self.assertIsNone(det.track_maps)

    def test_v1019_profile_converted(self):
        d = json.loads(core.Profile(track_rect=[265, 198, 535, 283], track_angle=-38, occluded_track=True).to_json())
        d.pop('track_quad')                                     # v1.0.19 存的檔案沒有這個欄位
        p = core.Profile.from_dict(d)
        self.assertEqual(len(p.track_points()), 4)
        self.assertFalse(p.track_is_rect())
        q = p.track_points()
        self.assertTrue(np.allclose(q.mean(axis=0), [400, 240.5], atol=0.01))   # 繞中心旋轉
        p2 = core.Profile.from_dict(json.loads(core.Profile(track_quad=QUAD).to_json()))
        self.assertEqual(p2.track_points().tolist(), np.float32(QUAD).tolist())

    def test_train_through_quad_entry_and_exit(self):
        W, H = 380, 90
        M = _homography(QUAD, W, H)
        ref = _to_img(M, [[W / 2, 0], [W / 2, H]])
        p = core.Profile(track_quad=QUAD, ref_line=ref.round(1).tolist())
        crop = core.work_crop(p, 640, 360)
        det = core.Detector(p, crop)
        x0, y0, x1, y1 = crop
        det.bg = np.full((y1 - y0, x1 - x0), 100, np.float32)
        for i in range(121):
            t = i / 10
            im = np.full((360, 640), 100, np.uint8)
            head = W / 2 + (t - 5) * 60                         # 5 秒時車頭碰到參考線，車長 120（2 秒）
            tail = head - 120
            a, b = max(0.0, tail), min(float(W), head)
            if b > a:
                poly = _to_img(M, [[a, 8], [b, 8], [b, H - 8], [a, H - 8]])
                cv2.fillConvexPoly(im, np.rint(poly).astype(np.int32), 230)
            det.feed(t, im[y0:y1, x0:x1], 0, 'synthetic', t)
        ev = [e for e in det.finish() if e['valid']]
        self.assertEqual(len(ev), 1, ev)
        self.assertLess(abs(ev[0]['start'] - 5), 0.3)
        self.assertLess(abs(ev[0]['end'] - 7), 0.3)
        self.assertEqual(ev[0]['direction'], '往右')


class PartialReference(unittest.TestCase):
    """#115：參考線被擋住一部分時，程式同時跑一般判讀與遮擋判讀，自動選一份（不用勾選）"""

    def _run(self, wall, occluded, obj='train'):
        p = core.Profile(track_rect=[100, 140, 540, 220], ref_line=[[320, 140], [320, 220]], ref_height=0)
        crop = core.work_crop(p, 640, 360)
        det = core.Detector(p, crop, occluded=occluded)
        x0, y0, x1, y1 = crop
        for i in range(141):
            t = i / 10
            im = np.full((360, 640), 100, np.uint8)
            if obj == 'train':
                head = int(320 + (t - 5) * 80)
                cv2.rectangle(im, (head - 200, 146), (head, 214), 230, -1)
            if wall:                                              # 參考線左邊被牆擋住：列車只出現在右側
                cv2.rectangle(im, (100, 130), (316, 230), 100, -1)
            det.feed(t, im[y0:y1, x0:x1], 0, 'synthetic', t)
        return [e for e in det.finish() if e['valid']]

    def test_hidden_side_needs_occlusion_mode(self):
        self.assertEqual(self._run(wall=True, occluded=False), [])     # 一般判讀：只有一側 → 排除
        ev = self._run(wall=True, occluded=True)
        self.assertEqual(len(ev), 1, ev)
        self.assertLess(abs(ev[0]['start'] - 5), 0.4)

    def test_unhidden_train_found_by_both(self):
        self.assertEqual(len(self._run(wall=False, occluded=False)), 1)
        self.assertEqual(len(self._run(wall=False, occluded=True)), 1)

    @staticmethod
    def _evs(starts, valid=True):
        return [dict(start=float(t), end=float(t) + 3, valid=valid, reason='') for t in starts]

    def test_choose_occlusion(self):
        n = self._evs(range(0, 1000, 100))                          # 一般判讀 10 筆
        o = self._evs(range(0, 1000, 100)) + self._evs([50, 450])   # 遮擋判讀多 2 筆（例如車燈）→ 照用一般判讀
        use, st = core.choose_occlusion(n, o)
        self.assertFalse(use)
        self.assertEqual((st['normal'], st['occluded']), (10, 12))
        n2 = self._evs([0, 300])                                    # 大部分列車一般判讀都沒找到 → 改用遮擋判讀
        use, st = core.choose_occlusion(n2, o)
        self.assertTrue(use)
        self.assertLess(st['share'], 0.5)
        self.assertFalse(core.choose_occlusion([], self._evs([1, 200]))[0])   # 只多 2 筆不換

    def test_process_reports_choice(self):
        from trainwatch import selftest
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            f = str(pathlib.Path(tmp) / 'v.avi')
            selftest.make_video(f, seconds=20, trains=((5, 1), (12, -1)))
            info = {}
            evs = core.process([f], [0], core.Profile(), info=info)
            self.assertFalse(info['occlusion']['used'])
            self.assertEqual(sum(e['valid'] for e in evs), 2)


@unittest.skipUnless(HAVE_TK, '沒有畫面，略過 GUI 測試')
class Page1Ui(unittest.TestCase):
    def setUp(self):
        from trainwatch import gui
        self.gui = gui
        self.root = tk.Tk()
        self.root.geometry('1280x760')
        self.app = gui.App(self.root)
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_longstop_checkbox(self):
        a = self.app
        self.assertTrue(a.profile.passing_only)           # 新監測站預設：短暫通過
        self.assertFalse(a.var_longstop.get())
        a.var_longstop.set(True); a.set_longstop()
        self.assertFalse(a.profile.passing_only)
        a.var_longstop.set(False)
        self.assertTrue(a._collect_profile().passing_only)
        self.assertFalse(hasattr(a, 'var_angle'))           # 角度輸入框已拿掉（#114）

    def test_drag_corner_and_redraw_rectangle(self):
        a = self.app
        a.cur_frame = np.full((360, 640, 3), 90, np.uint8)
        a.disp_scale = 1.0
        a.profile.track_rect = [100, 140, 540, 220]
        a.profile.ref_line = [[320, 140], [320, 220]]
        a.mode.set('track')
        E = lambda x, y: types.SimpleNamespace(x=x, y=y)
        a.on_press(E(542, 138))                            # 按在右上角附近
        self.assertEqual(a.drag_corner, 1)
        a.on_drag(E(520, 100)); a.on_release(E(520, 100))
        self.assertEqual(a.profile.track_points()[1].tolist(), [520.0, 100.0])
        self.assertEqual(a.check_geometry(), '')
        a.on_press(E(100, 220)); a.on_drag(E(400, 140)); a.on_release(E(400, 140))   # 左下角拖到右上方 → 交叉
        self.assertIn('交叉', a.check_geometry())
        a.on_press(E(300, 300)); a.on_drag(E(60, 120)); a.on_release(E(60, 120))     # 空白處重新框長方形
        self.assertIsNone(a.profile.track_quad)
        self.assertEqual(a.profile.track_rect, [60, 120, 300, 300])


if __name__ == '__main__':
    unittest.main(verbosity=2)
