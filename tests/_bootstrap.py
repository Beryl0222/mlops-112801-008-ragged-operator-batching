"""测试公共辅助：在 ``python -I`` 下保证能导入工作区的 segbatch。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
