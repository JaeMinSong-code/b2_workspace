# obstacles/ladder.py
from __future__ import annotations

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema


def _ensure_translate_op(xformable: UsdGeom.Xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def _ensure_rotate_xyz_op(xformable: UsdGeom.Xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            return op
    return xformable.AddRotateXYZOp()


def _set_tr_rxyz(prim, pos_xyz, rpy_deg):
    xf = UsdGeom.Xformable(prim)
    _ensure_translate_op(xf).Set(Gf.Vec3d(*pos_xyz))
    _ensure_rotate_xyz_op(xf).Set(Gf.Vec3f(*rpy_deg))  # degrees (roll, pitch, yaw)


def _define_static_box(stage, prim_path: str, size_xyz, pos_xyz):
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    _ensure_translate_op(xformable).Set(Gf.Vec3d(*pos_xyz))

    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Geom")
    cube.CreateSizeAttr(1.0)

    cube_xf = UsdGeom.Xformable(cube.GetPrim())
    # 기존 ScaleOp가 있을 수 있으니 "있으면 set, 없으면 add"로 처리
    scale_op = None
    for op in cube_xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = cube_xf.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*size_xyz))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())


def spawn_ladder_boxes_per_env(
    num_envs: int,
    base_x: float = 2.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    ladder_width: float = 1.45,
    ladder_height: float = 2.20,
    rail_thickness: float = 0.08,
    rung_thickness: float = 0.07,
    rung_count: int = 8,
    root_name: str = "Ladder",
    roll_deg: float = 0.0,
    pitch_deg: float = 15.0,  # ✅ 사다리 기울기(앞/뒤): +면 +Y축?가 아니라 "RotateXYZ의 Y축 회전" (pitch)
    yaw_deg: float = 0.0,
):
    stage = omni.usd.get_context().get_stage()

    # 로컬 좌표계에서 레일/가로봉 배치 (루트가 base_xyz로 이동 + 회전함)
    y_left = -0.5 * ladder_width
    y_right = 0.5 * ladder_width
    rail_center_z = 0.5 * ladder_height

    if rung_count <= 1:
        rung_z_list = [0.5 * ladder_height]
    else:
        rung_z_list = [(ladder_height * i / (rung_count - 1)) for i in range(rung_count)]

    for i in range(num_envs):
        env_path = f"/World/envs/env_{i}"
        ladder_root = f"{env_path}/{root_name}"

        # ✅ 사다리 루트 프림에만 위치/회전 적용
        root_prim = UsdGeom.Xform.Define(stage, ladder_root).GetPrim()
        _set_tr_rxyz(root_prim, (base_x, base_y, base_z), (roll_deg, pitch_deg, yaw_deg))

        # ✅ 자식 박스들은 "로컬 좌표"로만 배치
        _define_static_box(
            stage,
            f"{ladder_root}/RailLeft",
            (rail_thickness, rail_thickness, ladder_height),
            (0.0, y_left, rail_center_z),
        )
        _define_static_box(
            stage,
            f"{ladder_root}/RailRight",
            (rail_thickness, rail_thickness, ladder_height),
            (0.0, y_right, rail_center_z),
        )

        for k, z in enumerate(rung_z_list):
            _define_static_box(
                stage,
                f"{ladder_root}/Rung_{k:02d}",
                (rung_thickness, ladder_width, rung_thickness),
                (0.0, 0.0, z),
            )
