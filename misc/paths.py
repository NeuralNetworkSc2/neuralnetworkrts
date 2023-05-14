import logging
import os
import platform
import re
from pathlib import Path

logger = logging.getLogger(__name__)

BASEDIR = {"Windows": "C:/Program Files (x86)/StarCraft II"}

USERPATH = {"Windows": "\\Documents\\StarCraft II\\ExecuteInfo.txt"}

BINPATH = {"Windows": "SC2_x64.exe"}

CWD = {"Windows": "Support64"}

PF = os.environ.get("SC2PF", platform.system())

def latest_executeble(versions_dir):
    latest = max((int(p.name[4:]), p) for p in versions_dir.iterdir() if p.is_dir() and p.name.startswith("Base"))
    version, path = latest
    return path / BINPATH[PF]

class _MetaPaths(type):
    def __setup(self):
        if PF not in BASEDIR:
            exit(1)
        try:
            base = os.environ.get("SC2PATH")
            if base is None:
                base = BASEDIR[PF]
            self.BASE = Path(base).expanduser()
            self.EXECUTABLE = latest_executeble(self.BASE / "Versions")
            self.CWD = self.BASE / CWD[PF] if CWD[PF] else None

            self.REPLAYS = self.BASE / "Replays"
            if (self.BASE / "maps").exists():
                self.MAPS = self.BASE / "maps"
            else:
                self.MAPS = self.BASE / "Maps"
        except FileNotFoundError as e:
            logger.critical(f"Игра не найдена: Файл '{e.filename}' отсутствует.")
            exit(1)

    def __getattr__(self, attr):
        self.__setup()
        return getattr(self, attr)

class Paths(metaclass=_MetaPaths):
    "Класс-родитель"
