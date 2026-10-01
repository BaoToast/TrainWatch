"""列車通過判讀 — 程式進入點（打包成 exe 用）

  TrainWatch.exe                     開啟視窗
  TrainWatch.exe --selftest 結果.json  自我測試（GitHub 打包後自動執行，確認 exe 真的能判讀）
"""
import sys
import traceback


def _fatal(msg):
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk(); r.withdraw()
        messagebox.showerror('列車通過判讀', msg)
    except Exception:
        print(msg, file=sys.stderr)


if __name__ == '__main__':
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
        _fatal('程式無法啟動：\n\n' + traceback.format_exc()[-2000:])
