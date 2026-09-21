"""Extreme Parkour env config for the B2 quadruped (IsaacLab direct workflow).

Subclasses the existing B2LabFlatEnvCfg so the velocity-tracking pipeline is left
completely intact. Enable this config to train the Extreme-Parkour teacher/student.

Reference: Cheng et al., "Extreme Parkour with Legged Robots" (ICRA 2024).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from B2_Lab.terrains import CURRICULUM_TERRAINS_CFG

from .b2_lab_env_cfg import B2LabFlatEnvCfg


@configclass
class B2LabParkourEnvCfg(B2LabFlatEnvCfg):
    # ------------------------------------------------------------------ toggles
    parkour_mode: bool = True
    # Phase-1 teacher does not use depth; disable to skip the (slow) camera and
    # avoid needing --enable_cameras. Phase-2 vision training sets this True.
    enable_depth: bool = True

    # --- Extreme-Parkour observation layout (policy obs is a single flat vector) ---
    # [ proprio(49) | scandots(121) | priv_explicit(3) | priv_latent(33) | history(490) ] = 696
    # NOTE: the height scanner GridPattern (size 1.0m, res 0.1m) yields an 11x11=121
    # grid (endpoints included), not 10x10=100.
    num_prop_pk: int = 49
    num_scan_pk: int = 121          # existing height scanner: 11x11 grid
    num_priv_pk: int = 3            # base linear velocity (explicit privileged)
    num_priv_latent_pk: int = 33    # existing privileged_obs_buf (fric/gains/contact/com)
    num_hist_pk: int = 10
    num_actor_obs_pk: int = 49 + 121 + 3 + 33 + 10 * 49  # = 696

    # --- depth camera (student / phase 2) ---
    depth_width: int = 87
    depth_height: int = 58
    depth_near: float = 0.05
    depth_far: float = 2.0

    # Forward-facing depth camera mounted on the base. Geometry approximates the
    # RealSense used in Extreme Parkour (~87 deg HFOV, resized to 87x58, far=2m).
    # rot: identity-forward with a slight downward pitch; convention="ros".
    # NOTE: focal_length/aperture and rot likely need TUNING on the real B2 mount.
    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_depth_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.05),
            rot=(0.9961947, 0.0, 0.0871557, 0.0),  # ~10 deg pitch-down about y (w,x,y,z)
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=22.78,   # ~87 deg HFOV with focal_length=12
            clipping_range=(0.05, 2.0),
        ),
        width=87,
        height=58,
        update_period=0.0,  # refresh each control step; rendered only when RTX sensors active
    )

    # --- parkour command sampling ---
    class parkour_commands:
        speed_range = [0.5, 1.5]        # commanded forward speed magnitude (m/s)
        heading_range = [-math.pi, math.pi]  # (unused with waypoint goals; kept for reference)

    # --- world-space goal waypoints (Extreme-Parkour navigation) ---
    class goals:
        num_goals = 8              # forward waypoints per env
        spacing = 1.5              # m between consecutive goals (along +x from spawn)
        reach_threshold = 0.5      # m xy-distance to advance to the next goal
        edge_threshold = 0.12      # normalized foot-scan height range flagged as an edge

    # ------------------------------------------------------------------ rewards
    # Extreme-Parkour reward weights (per second; env multiplies by dt internally).
    class rewards(B2LabFlatEnvCfg.rewards):
        class scales:
            # task
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # base regularization
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            # joint / actuation regularization
            dof_acc = -2.5e-7
            torques = -0.00001
            delta_torques = -1.0e-7
            action_rate = -0.1
            hip_pos = -0.5
            dof_error = -0.04
            # contact
            collision = -10.0
            feet_stumble = -1.0
            feet_edge = -1.0        # active only once terrain exposes env.feet_at_edge
            # terminal
            termination = -1.0

        tracking_sigma = 0.2
        reward_container_name = "B2quadParkourReward"


@configclass
class B2LabParkourTerrainEnvCfg(B2LabParkourEnvCfg):
    """Parkour on obstacle terrain (stairs/boxes/slopes) with difficulty curriculum.

    Goals are laid out forward from spawn, so the robot must traverse obstacles to
    reach them; feet_edge activates on the terrain discontinuities. Heavier than the
    plane variant (terrain generation) — use for the actual parkour training run.
    """

    terrain_curriculum: bool = True
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CURRICULUM_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
