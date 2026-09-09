"""프로젝트 로컬 terrain 생성 설정.

주의: ``CUSTOM_TERRAINS_CFG`` / ``CURRICULUM_TERRAINS_CFG`` 의 원본(4.5 IsaacLab 내부에
직접 추가돼 있던 정의)은 이 머신에 남아 있지 않아, IsaacLab 2.3.2 의
``ROUGH_TERRAINS_CFG`` 를 기반으로 재구성한 것입니다. 학습에서 실제로 사용할 경우
sub-terrain 비율/파라미터를 로봇(B2)에 맞게 다시 튜닝하세요.

- ``ROUGH_TERRAINS_CFG`` : IsaacLab 기본값을 그대로 재노출.
- ``CUSTOM_TERRAINS_CFG`` : curriculum 없이 섞인 지형(play/평가용). ``curriculum=False``.
- ``CURRICULUM_TERRAINS_CFG`` : 난이도 상승 curriculum 학습용. ``curriculum=True``.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# IsaacLab 2.3.2 기본 rough terrain 을 그대로 재노출 (호환성 유지)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # noqa: F401


# ---------------------------------------------------------------------------
# CUSTOM_TERRAINS_CFG
#   curriculum 을 쓰지 않고 여러 지형을 고르게 섞어 배치한다. play 스크립트에서
#   ``.replace(num_rows=2, num_cols=2, curriculum=False)`` 형태로 사용된다.
# ---------------------------------------------------------------------------
CUSTOM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""curriculum 없이 여러 지형을 섞어 배치한 설정 (재구성본)."""


# ---------------------------------------------------------------------------
# CURRICULUM_TERRAINS_CFG
#   난이도가 row 방향으로 상승하는 curriculum 학습용. ``curriculum=True``.
# ---------------------------------------------------------------------------
CURRICULUM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""row 방향으로 난이도가 상승하는 curriculum terrain 설정 (재구성본)."""


# ---------------------------------------------------------------------------
# PLAY_TERRAINS_CFG
#   play/평가 전용. CURRICULUM_TERRAINS_CFG 와 동일한 6종 지형을 쓰되,
#   각 지형의 난이도 range 를 '중간대'로 좁혀(가장 쉬운·가장 어려운 극단 제외)
#   "적당히 쉬운 ~ 적당히 어려운" 만 남긴다. grid 도 작게 줄여 생성/렌더 부담↓.
#   curriculum=False → 각 타일이 좁힌 range 안에서 랜덤 난이도(중간대)로 생성.
# ---------------------------------------------------------------------------
PLAY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),          # 타일 크기는 유지(계단/플랫폼 형상 보존)
    border_width=20.0,
    num_rows=3,               # 학습(10) 대비 축소
    num_cols=6,               # 6종 각 1열 정도
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # 계단 0.05~0.23 → 중간대 0.10~0.16
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.10, 0.16),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.10, 0.16),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 박스 0.05~0.2 → 중간대 0.08~0.14
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.08, 0.14), platform_width=2.0
        ),
        # 러프 0.02~0.10 → 중간대 0.04~0.075
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.04, 0.075), noise_step=0.02, border_width=0.25
        ),
        # 경사 0.0~0.4 → 중간대 0.12~0.26
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.12, 0.26), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.12, 0.26), platform_width=2.0, border_width=0.25
        ),
    },
)
"""play/평가 전용: 각 지형 난이도를 중간대로 좁힌 소형 mixed terrain."""
