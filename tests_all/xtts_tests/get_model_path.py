import os
import sys
from pathlib import Path

def get_tests_output_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件所在目录的绝对路径
    two_levels_up = os.path.abspath(os.path.join(current_dir, "../"))  # 上上级目录的路径
    return two_levels_up

def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)

if __name__ == "__main__":
    print(get_user_data_dir("tts"))
    print(get_tests_output_path())
    print(len("港风回忆，今日心动。Nostalgic vibes, today's excitement."))