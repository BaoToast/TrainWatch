"""
單元測試：python -m unittest discover -s tests -v

真實影片的回歸測試只在設定環境變數 TW_SAMPLES（放樣本影片的資料夾）時執行；
樣本影片是使用者的監測資料，不可以放進 repository。
"""
import datetime as dt
import glob
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainwatch import core, export, osd, selftest  # noqa: E402


class TestOsdFormat(unittest.TestCase):
    def test_tokens_and_roundtrip(self):
        fmt = 'YYYY-MM-DD HH:MM:SS'
        d = dt.datetime(2026, 1, 2, 23, 4, 7)
        self.assertEqual(osd.render(d, fmt), '2026-01-02 23:04:07')
        self.assertEqual(osd.parse('2026-01-02 23:04:07', fmt), d)

    def test_other_format(self):
        fmt = 'DD/MM/YY HH:MM:SS'
        d = dt.datetime(2026, 3, 5, 7, 8, 9)
        self.assertEqual(osd.render(d, fmt), '05/03/26 07:08:09')
        self.assertEqual(osd.parse('05/03/26 07:08:09', fmt), d)

    def test_invalid(self):
        self.assertIsNone(osd.parse('2026-13-02 23:04:07', 'YYYY-MM-DD HH:MM:SS'))
        self.assertIsNone(osd.parse('2026-01-02 25:04:07', 'YYYY-MM-DD HH:MM:SS'))
        with self.assertRaises(ValueError):
            osd.tokens('HH:MM:SS')

    def test_read_rendered_digits(self):
        """用內建字樣拼出一張字幕圖，應該讀回同一個字串"""
        import cv2
        fmt = osd.DEFAULT_FORMAT
        T = osd.default_templates()
        text = '2026-01-02 13:58:59'
        x0, y0 = 14, 8
        img = np.zeros((40, 260), np.uint8)
        for i, ch in enumerate(text):
            if ch.isdigit():
                cell = (T[int(ch)] * 80 + 150).clip(0, 255).astype(np.uint8)
                img[y0:y0 + 16, x0 + i * 10:x0 + i * 10 + 10] = cell
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        d, txt, w = osd.read_datetime(frame, osd.DEFAULT_RECT, fmt, T)
        self.assertEqual(txt, text)
        self.assertEqual(d, dt.datetime(2026, 1, 2, 13, 58, 59))


class TestCore(unittest.TestCase):
    def test_number_events(self):
        evs = [dict(start=3, valid=True), dict(start=1, valid=False), dict(start=2, valid=True)]
        core.number_events(evs)
        self.assertEqual([(e['start'], e['no']) for e in evs], [(1, 1), (2, 1), (3, 2)])

    def test_fmt_time(self):
        ts = dt.datetime(2026, 1, 2, 7, 6, 5).timestamp() + 0.35
        self.assertEqual(core.fmt_time(ts), '07:06:05.3')
        self.assertEqual(core.fmt_time(ts, True), '2026-01-02 07:06:05.3')

    def test_profile_json(self):
        p = core.Profile(name='站A')
        q = core.Profile.from_dict(__import__('json').loads(p.to_json()))
        self.assertEqual(q.name, '站A')
        self.assertEqual(q.osd_rect, p.osd_rect)

    def test_selftest_video(self):
        """假列車影片：兩筆、時間誤差 < 0.4 秒、方向正確、有截圖與短片"""
        r = selftest.run()
        self.assertTrue(r['ok'], r['problems'])


class TestExport(unittest.TestCase):
    def test_roundtrip(self):
        d = tempfile.mkdtemp()
        base = dt.datetime(2026, 1, 2, 10, 0, 0).timestamp()
        evs = [dict(no=1, start=base + 10, end=base + 25, start_file='a.mkv', end_file='a.mkv', start_pos=10, end_pos=25,
                    direction='往右', valid=True, need_check=True, reason='方向不確定', shots=['001_1車頭.jpg'], clip='001.mp4'),
               dict(no=1, start=base + 40, end=base + 40.2, start_file='a.mkv', end_file='a.mkv', start_pos=40, end_pos=40.2,
                    direction='無法判定', valid=False, reason='時間太短')]
        prof = core.Profile()
        timing = [dict(offset=base, source='畫面字幕', support=1.0, msg='')]
        export.save_json(os.path.join(d, 'results.json'), evs, prof, ['a.mkv'], timing)
        e2, p2, f2, t2 = export.load_json(os.path.join(d, 'results.json'))
        self.assertEqual(len(e2), 2)
        self.assertEqual(e2[0]['direction'], '往右')
        export.write_excel(os.path.join(d, 'x.xlsx'), e2, p2, f2, t2)
        export.write_csv(os.path.join(d, 'x.csv'), e2)
        from openpyxl import load_workbook
        wb = load_workbook(os.path.join(d, 'x.xlsx'))
        self.assertEqual(wb.sheetnames, ['列車通過紀錄', '已排除', '設定與影片'])
        ws = wb['列車通過紀錄']
        self.assertEqual(ws.cell(2, 3).value, '10:00:10.0')
        self.assertEqual(ws.cell(2, 9).value, '是')
        self.assertEqual(wb['已排除'].max_row, 2)
        with open(os.path.join(d, 'x.csv'), encoding='utf-8-sig') as fh:
            self.assertEqual(len(fh.read().strip().splitlines()), 2)


class TestGuiHelpers(unittest.TestCase):
    def test_parse(self):
        try:
            from trainwatch import gui
        except Exception as e:  # 沒有 tkinter 的環境
            self.skipTest(str(e))
        self.assertAlmostEqual(gui.parse_hms('08:15:30.5'), 29730.5)
        self.assertAlmostEqual(gui.parse_hms('081530'), 29730)
        self.assertEqual(gui.parse_datetime('2026/01/02 08:00:00'), dt.datetime(2026, 1, 2, 8, 0, 0))
        with self.assertRaises(ValueError):
            gui.parse_hms('25:00:00')


def _expected():
    root = os.environ.get('TW_SAMPLES')
    p = os.path.join(root, 'expected.json') if root else ''
    return p if p and os.path.exists(p) else None


@unittest.skipUnless(_expected(), '沒有設定 TW_SAMPLES，或樣本資料夾裡沒有 expected.json')
class TestSamples(unittest.TestCase):
    """樣本資料夾放影片＋expected.json（逐格截圖核對過的答案）。兩者都是真實監測資料，不放進 repository。
    expected.json 格式：[{"files": ["檔名開頭", ...], "events": [{"start": "時:分:秒.x", "end": "…", "direction": "往右"|null}]}]"""
    def test_samples(self):
        root = os.environ['TW_SAMPLES']
        import json
        with open(_expected(), encoding='utf-8') as fh:
            cases = json.load(fh)

        def sec(s):
            h, m, x = s.split(':')
            return int(h) * 3600 + int(m) * 60 + float(x)
        for case in cases:
            keys = case['files']
            exp = [(x['start'], x['end'], x['direction']) for x in case['events']]
            files = [glob.glob(os.path.join(root, k + '*'))[0] for k in keys]
            bases = []
            for f in files:
                r = osd.calibrate(f, osd.DEFAULT_RECT)
                self.assertTrue(r['ok'], (f, r))
                self.assertGreaterEqual(r['support'], 0.95)
                bases.append(r['offset'])
            evs = [e for e in core.number_events(core.process(files, bases, core.Profile())) if e['valid']]
            got = [(core.fmt_time(e['start']), core.fmt_time(e['end']), e['direction']) for e in evs]
            self.assertEqual(len(got), len(exp), (keys, got))
            for g, x in zip(got, exp):
                self.assertLess(abs(sec(g[0]) - sec(x[0])), 0.5, (keys, g, x))
                self.assertLess(abs(sec(g[1]) - sec(x[1])), 0.5, (keys, g, x))
                if x[2]:
                    self.assertEqual(g[2], x[2], (keys, g, x))


    def test_shots_and_clips_every_sample(self):
        """每支樣本影片（含每秒實際格數和檔頭不符的）：每一筆都要有 3 張截圖和短片，
        而且車頭截圖上的畫面時間要和判讀的車頭時間一致（差 1 秒內）"""
        root = os.environ['TW_SAMPLES']
        import cv2
        T = osd.default_templates()
        for f in sorted(glob.glob(os.path.join(root, '*.mkv'))):
            r = osd.calibrate(f, osd.DEFAULT_RECT)
            prof = core.Profile()
            evs = [e for e in core.number_events(core.process([f], [r['offset']], prof)) if e['end'] - e['start'] >= 0.6]
            out = tempfile.mkdtemp()
            core.save_frames_and_clips(evs, prof, out, files=[f])
            for e in evs:
                self.assertEqual(len(e['shots']), 3, (f, core.fmt_time(e['start'])))
                self.assertTrue(e['clip'] and os.path.getsize(os.path.join(out, '短片', e['clip'])) > 0, (f, core.fmt_time(e['start'])))
                img = cv2.imdecode(np.fromfile(os.path.join(out, '截圖', e['shots'][0]), np.uint8), cv2.IMREAD_COLOR)
                d, txt, _w = osd.read_datetime(img, osd.DEFAULT_RECT, osd.DEFAULT_FORMAT, T)
                self.assertIsNotNone(d, (f, txt))
                self.assertLess(abs(d.timestamp() - e['start']), 1.0, (f, txt, core.fmt_time(e['start'])))
            fr, pos = core.read_frame_at(f, 550.0)        # 預覽：影片後段也要讀得到
            self.assertIsNotNone(fr, f)
            self.assertLess(abs(pos - 550.0), 0.2, f)


if __name__ == '__main__':
    unittest.main()
