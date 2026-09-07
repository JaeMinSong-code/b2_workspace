"""USD flatten 유틸리티.

layered/referenced USD(예: Isaac Sim URDF importer 결과물, base/physics/sensor 로 나뉜
b2_description) 를 외부 의존이 없는 단일 .usd 파일로 합성(flatten)해서 저장한다.

사용법 (Isaac Sim 5.1 python.sh 로 실행):

    /home/js/isaacsim/python.sh scripts/tools/flatten_usd.py \
        --src /path/to/robot.usd \
        --dst /path/to/robot_flatten.usd

flatten 이 제대로 됐는지(외부 파일 의존 0) 자동 검증까지 수행한다.
OmniPBR.mdl 같은 Kit 내장 머티리얼은 런타임에 해결되므로 unresolved 로 떠도 정상이다.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Flatten a layered/referenced USD into a single self-contained file.")
parser.add_argument("--src", required=True, help="입력 USD 경로 (layered/referenced 가능)")
parser.add_argument("--dst", required=True, help="출력 flatten USD 경로")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdPhysics, UsdUtils  # noqa: E402


def main():
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    stage = Usd.Stage.Open(src)
    if stage is None:
        raise RuntimeError(f"입력 USD 를 열 수 없습니다: {src}")

    # 모든 sublayer / reference / payload / variant 를 단일 레이어로 합성
    flat_layer = stage.Flatten()
    flat_layer.Export(dst)
    print(f"[flatten] exported -> {dst}  ({os.path.getsize(dst)/1e6:.2f} MB)")

    # --- 검증: 외부 파일 의존이 남아있지 않은지 ---
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(dst))
    ext_layers = [l.identifier for l in layers if os.path.abspath(l.identifier) != dst]
    print(f"[verify] external USD layers : {len(ext_layers)} {ext_layers}")
    print(f"[verify] external asset deps : {len(assets)}")
    print(f"[verify] unresolved          : {list(unresolved)}  (OmniPBR.mdl 등 Kit 내장은 정상)")

    # --- 검증: articulation / joint / body 구조 유지 ---
    v = Usd.Stage.Open(dst)
    art = [p.GetPath().pathString for p in v.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    rev = [p.GetName() for p in v.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint"]
    bodies = [p.GetName() for p in v.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    dp = v.GetDefaultPrim()
    print(f"[verify] defaultPrim   : {dp.GetPath() if dp else None}")
    print(f"[verify] articulation  : {art}")
    print(f"[verify] revolute joints ({len(rev)}): {rev}")
    print(f"[verify] rigid bodies  ({len(bodies)})")

    ok = len(ext_layers) == 0 and len(assets) == 0
    print(f"[result] self-contained: {'OK' if ok else 'FAIL (외부 의존 남음)'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
