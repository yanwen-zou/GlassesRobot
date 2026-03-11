import math
import tempfile
import time
from pathlib import Path

import pybullet as p
import pybullet_data
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


URDF_DIR = Path(__file__).resolve().parent / "ur_description"
URDF_PATH = URDF_DIR / "urdf" / "ur5_robot.urdf"
ROBOT_IP = "192.168.2.102"
POSITION_STEP = 0.005
ROTATION_STEP = 0.05
SIM_DT = 1.0 / 240.0
CONTROL_DT = 0.05
ESC_KEY = 27
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
MAX_JOINT_SPEED = 0.3


def build_pybullet_urdf() -> str:
    urdf_text = URDF_PATH.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace("package://ur_description/", f"{URDF_DIR.as_posix()}/")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="ur5_robot_pybullet_",
        delete=False,
        encoding="utf-8",
    ) as fp:
        fp.write(urdf_text)
        return fp.name


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def key_is_active(keys: dict[int, int], code: int) -> bool:
    return code in keys and keys[code] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)


def key_was_triggered(keys: dict[int, int], code: int) -> bool:
    return code in keys and keys[code] & p.KEY_WAS_TRIGGERED


def apply_key_delta(target_pos: list[float], target_euler: list[float], keys: dict[int, int]) -> bool:
    moved = False

    if key_was_triggered(keys, ord("w")):
        target_pos[0] += POSITION_STEP
        moved = True
    if key_was_triggered(keys, ord("s")):
        target_pos[0] -= POSITION_STEP
        moved = True
    if key_was_triggered(keys, ord("a")):
        target_pos[1] += POSITION_STEP
        moved = True
    if key_was_triggered(keys, ord("d")):
        target_pos[1] -= POSITION_STEP
        moved = True
    if key_was_triggered(keys, ord("q")):
        target_pos[2] += POSITION_STEP
        moved = True
    if key_was_triggered(keys, ord("e")):
        target_pos[2] -= POSITION_STEP
        moved = True

    if key_was_triggered(keys, ord("i")):
        target_euler[0] += ROTATION_STEP
        moved = True
    if key_was_triggered(keys, ord("k")):
        target_euler[0] -= ROTATION_STEP
        moved = True
    if key_was_triggered(keys, ord("j")):
        target_euler[1] += ROTATION_STEP
        moved = True
    if key_was_triggered(keys, ord("l")):
        target_euler[1] -= ROTATION_STEP
        moved = True
    if key_was_triggered(keys, ord("u")):
        target_euler[2] += ROTATION_STEP
        moved = True
    if key_was_triggered(keys, ord("o")):
        target_euler[2] -= ROTATION_STEP
        moved = True

    for idx in range(3):
        target_euler[idx] = wrap_to_pi(target_euler[idx])
    return moved


def reset_pybullet_joints(client: int, robot_id: int, joint_indices: list[int], joints: list[float]) -> None:
    for joint_idx, joint_pos in zip(joint_indices, joints):
        p.resetJointState(robot_id, joint_idx, joint_pos, physicsClientId=client)


def get_ee_target_from_robot(client: int, robot_id: int, ee_link_index: int) -> tuple[list[float], list[float]]:
    ee_state = p.getLinkState(
        robot_id,
        ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=client,
    )
    return list(ee_state[4]), list(p.getEulerFromQuaternion(ee_state[5]))


def limit_joint_step(target_joints: list[float], actual_joints: list[float]) -> list[float]:
    max_step = MAX_JOINT_SPEED * CONTROL_DT
    limited = []
    for target_joint, actual_joint in zip(target_joints, actual_joints):
        delta = max(-max_step, min(max_step, target_joint - actual_joint))
        limited.append(actual_joint + delta)
    return limited


def main() -> None:
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(SIM_DT, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    robot_id = p.loadURDF(build_pybullet_urdf(), useFixedBase=True, physicsClientId=client)

    joint_indices = []
    joint_lower = []
    joint_upper = []
    joint_ranges = []
    ee_link_index = -1

    for joint_idx in range(p.getNumJoints(robot_id, physicsClientId=client)):
        joint_info = p.getJointInfo(robot_id, joint_idx, physicsClientId=client)
        if joint_info[12].decode("utf-8") == "ee_link":
            ee_link_index = joint_idx
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(joint_idx)
            joint_lower.append(joint_info[8])
            joint_upper.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])

    if ee_link_index < 0:
        raise RuntimeError("Failed to find ee_link in URDF.")

    current_joints = list(rtde_r.getActualQ())
    reset_pybullet_joints(client, robot_id, joint_indices, current_joints)
    target_pos, target_euler = get_ee_target_from_robot(client, robot_id, ee_link_index)
    last_servo_time = 0.0

    print("UR5 real robot keyboard control started")
    print(f"Robot IP: {ROBOT_IP}")
    print("PyBullet is only used for IK and visualization.")
    print("W/S: X  A/D: Y  Q/E: Z")
    print("I/K: Roll  J/L: Pitch  U/O: Yaw")
    print("R: sync target to actual robot pose")
    print("ESC: quit")

    try:
        while p.isConnected(client):
            keys = p.getKeyboardEvents(physicsClientId=client)

            if key_is_active(keys, ESC_KEY):
                break

            if key_was_triggered(keys, ord("r")):
                current_joints = list(rtde_r.getActualQ())
                reset_pybullet_joints(client, robot_id, joint_indices, current_joints)
                target_pos, target_euler = get_ee_target_from_robot(client, robot_id, ee_link_index)

            actual_joints = list(rtde_r.getActualQ())
            reset_pybullet_joints(client, robot_id, joint_indices, actual_joints)
            target_pos, target_euler = get_ee_target_from_robot(client, robot_id, ee_link_index)
            moved = apply_key_delta(target_pos, target_euler, keys)

            if moved and time.time() - last_servo_time >= CONTROL_DT:
                target_quat = p.getQuaternionFromEuler(target_euler)
                joint_targets = list(
                    p.calculateInverseKinematics(
                        robot_id,
                        ee_link_index,
                        targetPosition=target_pos,
                        targetOrientation=target_quat,
                        lowerLimits=joint_lower,
                        upperLimits=joint_upper,
                        jointRanges=joint_ranges,
                        restPoses=actual_joints,
                        maxNumIterations=200,
                        residualThreshold=1e-4,
                        physicsClientId=client,
                    )[: len(joint_indices)]
                )
                joint_targets = limit_joint_step(joint_targets, actual_joints)
                rtde_c.servoJ(
                    joint_targets,
                    0.5,
                    0.5,
                    CONTROL_DT,
                    SERVO_LOOKAHEAD,
                    SERVO_GAIN,
                )
                reset_pybullet_joints(client, robot_id, joint_indices, joint_targets)
                last_servo_time = time.time()

            p.stepSimulation(physicsClientId=client)
            time.sleep(SIM_DT)
    finally:
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
