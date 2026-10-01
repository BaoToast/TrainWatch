"""列車通過判讀 — 程式進入點

  啟動列車通過判讀.bat                    （免安裝版）開啟視窗
  python main.py                          開啟視窗
  python main.py --selftest 結果.json      自我測試（GitHub 打包時自動執行）
"""
import os
import sys
import traceback


def _fatal(msg):
    """啟動失敗：寫錯誤紀錄，並用 Windows 內建訊息框顯示（不依賴 tkinter，tkinter 壞掉也看得到）"""
    home = os.environ.get('TRAINWATCH_HOME') or os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(home, '錯誤紀錄.txt'), 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n\n')
    except OSError:
        pass
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg[-1800:], '列車通過判讀', 0x10)
            return
        except Exception:
            pass
    print(msg, file=sys.stderr)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) >= 2 and sys.argv[1] == '--selftest':
        out = sys.argv[2] if len(sys.argv) > 2 else 'selftest.json'
        try:
            from trainwatch import selftest
            rep = selftest.run(out)
            sys.exit(0 if rep['ok'] else 1)
        except Exception:
            with open(out, 'w', encoding='utf-8') as fh:
                fh.write('{"ok": false, "error": %r}' % traceback.format_exc())
            sys.exit(2)
    try:
        from trainwatch.gui import main
        main()
    except Exception:
        _fatal('程式無法啟動：\n\n' + traceback.format_exc()[-3000:])
