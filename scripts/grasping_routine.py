import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from src.orca_core.orca_core.hand_factory import load_hand
from src.orca_core.orca_core.utils.cli import add_hand_arguments, connect_hand, shutdown_hand

_REPLAY_FILE = Path("./replay_sequences/grasping.yaml")
_STEP_TIME = 0.01
THRESHOLD = 0.5


def ease_in_out(t: float) -> float:
    return 0.5 * (1 - np.cos(np.pi * t))


def open_hand(hand, start: np.ndarray, end: np.ndarray, transition_time: float) -> None:
    n_steps = max(1, int(transition_time / _STEP_TIME))
    start_time = time.time()
    for step in range(n_steps + 1):
        alpha = ease_in_out(step / n_steps)
        pose = (1 - alpha) * start + alpha * end
        hand.set_joint_positions(pose)
        remaining = start_time + step * _STEP_TIME - time.time()
        if remaining > 0:
            time.sleep(remaining)

def close_hand(hand, start: np.ndarray, end: np.ndarray, transition_time: float) -> None:
    """Curl fingers from start toward end, stopping each finger independently when it exceeds THRESHOLD force.

    end acts as the maximum curl limit — no finger moves beyond it even if no
    contact is detected, preventing damage.
    """
    n_steps = max(1, int(transition_time / _STEP_TIME))
    start_time = time.time()

    hand.start_tactile_stream(resultant=True)
    hand.wait_for_first_frame() if hasattr(hand, "wait_for_first_frame") else time.sleep(0.1)

    # Map finger name -> list of joint indices using the naming convention {finger}_{joint_type}
    finger_joints: dict[str, list[int]] = {}
    for i, joint_id in enumerate(hand.config.joint_ids):
        finger = joint_id.split('_')[0]
        finger_joints.setdefault(finger, []).append(i)

    # Tracks the frozen pose for each finger that has hit the threshold
    stopped: dict[str, np.ndarray] = {}

    current_pose = start.copy()

    for step in range(n_steps + 1):
        reading = hand.get_tactile_forces()
        if reading is not None:
            for finger, fvec in reading.forces.items():
                if finger not in stopped and fvec[2] > THRESHOLD:
                    print(f"{finger}: contact detected ({fvec[2]:.2f} N), stopping")
                    stopped[finger] = current_pose.copy()

        alpha = ease_in_out(step / n_steps)
        current_pose = (1 - alpha) * start + alpha * end

        # Hold stopped fingers at their frozen positions
        for finger, frozen in stopped.items():
            for idx in finger_joints.get(finger, []):
                current_pose[idx] = frozen[idx]

        hand.set_joint_positions(current_pose)

        remaining = start_time + step * _STEP_TIME - time.time()
        if remaining > 0:
            time.sleep(remaining)

    hand.stop_tactile_stream()


def main() -> int:
    parser = argparse.ArgumentParser(description="Grasping routine to grab a glass.")
    add_hand_arguments(parser)
    parser.add_argument(
        "--force-calibrate",
        action="store_true",
        help="Run calibration even if calibration.yaml already exists.",
    )
    parser.add_argument("--transition-time", type=float, default=2.0)
    args = parser.parse_args()

    try:
        replay_data = yaml.safe_load(_REPLAY_FILE.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        print(f"Replay file not found: {_REPLAY_FILE}")
        return 1

    waypoints = replay_data.get("waypoints", [])
    if not waypoints:
        print("No waypoints found in the replay file.")
        return 1

    relaxed = np.asarray(waypoints[0], dtype=np.float64)
    open_pose = np.asarray(waypoints[1], dtype=np.float64)
    closed_pose = np.asarray(waypoints[2], dtype=np.float64)
    high_five = np.asarray(waypoints[3], dtype=np.float64)

    hand = load_hand(args.config_path, mock=args.mock)
    try:
        connect_hand(hand)
        hand.init_joints(force_calibrate=args.force_calibrate or args.mock)

        hand.set_joint_positions(relaxed)
        time.sleep(args.transition_time)

        key_input = ''
        while key_input != 'q':
            key_input = input("Hand is in relaxed pose. Press w to open, s to close")
            curr_pose = hand.get_joint_position().as_array(hand.config.joint_ids)
            if key_input == 'w':
                #print(key_input)
                open_hand(hand, curr_pose, open_pose, args.transition_time)
                #curr_pose = open_pose
            elif key_input == 's':
                close_hand(hand, curr_pose, closed_pose, args.transition_time)
                #curr_pose = closed_pose
            elif key_input == 'd':
                close_hand(hand, curr_pose, relaxed, args.transition_time)
                #curr_pose = relaxed
            elif key_input == 'a':
                open_hand(hand, curr_pose, high_five, args.transition_time)
                #curr_pose = high_five
    
    finally:
        shutdown_hand(hand)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
