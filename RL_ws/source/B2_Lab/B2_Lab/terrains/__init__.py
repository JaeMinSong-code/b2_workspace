"""로컬 terrain 설정 패키지.

원래 이 프로젝트는 Isaac Sim 4.5용 IsaacLab 내부(`isaaclab.terrains.config.rough`)에
사용자가 직접 추가해 둔 ``CUSTOM_TERRAINS_CFG`` / ``CURRICULUM_TERRAINS_CFG`` 를
import 했습니다. Isaac Sim 5.1 / IsaacLab 2.3.2 에는 이 설정이 없어 ImportError 가
발생하므로, 프로젝트 내부(local)로 옮겨 관리합니다.
"""

from .custom_terrains import (  # noqa: F401
    CURRICULUM_TERRAINS_CFG,
    CUSTOM_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
)
