import argparse
import time
import curses
import numpy as np
from i2rt.robots.get_robot import get_yam_robot


def main(stdscr):
    parser = argparse.ArgumentParser(description="Keyboard control for YAM robot arm")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--step", type=float, default=0.05, help="Step size in radians per key press")
    args = parser.parse_args()

    # Initialize robot
    robot = get_yam_robot(channel=args.channel, zero_gravity_mode=True)
    
    # Get current joint positions as initial target
    current_pos = robot.get_joint_pos()
    target_pos = current_pos.copy()
    num_joints = len(target_pos)
    
    # Selected joint (0-indexed, default to joint 0)
    selected_joint = 0
    
    # Step size
    step = args.step
    
    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    
    running = True
    last_print = 0.0
    
    try:
        print("Starting robot control loop")
        while running:
            t0 = time.monotonic()
            
            # Non-blocking key read
            key = stdscr.getch()
            if key != -1:
                # Select joint with number keys 1-7
                if ord('1') <= key <= ord('7'):
                    joint_idx = key - ord('1')
                    if joint_idx < num_joints:
                        selected_joint = joint_idx
                
                # Control selected joint
                elif key in (curses.KEY_RIGHT, ord('+'), ord('=')):  # Increase position
                    target_pos[selected_joint] += step
                elif key in (curses.KEY_LEFT, ord('-'), ord('_')):  # Decrease position
                    target_pos[selected_joint] -= step
                
                # Adjust step size
                elif key in (curses.KEY_UP, ord('k')):  # Increase step
                    step *= 1.25
                elif key in (curses.KEY_DOWN, ord('j')):  # Decrease step
                    step = max(step / 1.25, 1e-4)
                
                # Reset target to current position
                elif key == ord('r'):
                    current_pos = robot.get_joint_pos()
                    target_pos = current_pos.copy()
                
                # Reset all joints to zero
                elif key == ord('z'):
                    target_pos = np.zeros(num_joints)
                
                # Quit
                elif key in (ord('q'), 27):  # 'q' or ESC
                    running = False
            
            # Send command to robot
            robot.command_joint_pos(target_pos)
            
            # Periodic status display
            now = time.monotonic()
            if now - last_print > 0.1:
                current_pos = robot.get_joint_pos()
                
                # Get terminal size
                height, width = stdscr.getmaxyx()
                
                # Safe addstr wrapper
                def safe_addstr(y, x, text):
                    if y < height and x < width:
                        try:
                            # Truncate text if too long
                            max_len = width - x
                            if max_len > 0:
                                text = text[:max_len]
                                stdscr.addstr(y, x, text)
                        except curses.error:
                            pass  # Ignore errors if we can't write
                
                stdscr.erase()
                safe_addstr(0, 0, "Keyboard Joint Control (q to quit)")
                safe_addstr(1, 0, f"Selected Joint: {selected_joint + 1} (use 1-{num_joints} to select)")
                safe_addstr(2, 0, f"Step size: {step:.5f} rad (↑/k increase, ↓/j decrease)")
                
                # Display joint positions
                safe_addstr(4, 0, "Joint | Current Deg | Target Deg | Error Deg")
                safe_addstr(5, 0, "------|--------------|-------------|----------")
                
                # Only display joints that fit on screen
                max_joints_to_show = min(num_joints, height - 8)
                for i in range(max_joints_to_show):
                    marker = " <--" if i == selected_joint else ""
                    current = np.rad2deg(current_pos[i])
                    target = np.rad2deg(target_pos[i])
                    error = target - current
                    safe_addstr(6 + i, 0, 
                        f"  {i+1}   | {current:+8.4f}  | {target:+8.4f}  | {error:+8.4f}{marker}")
                
                # Display controls if there's space
                controls_start = 6 + max_joints_to_show + 1
                if controls_start < height:
                    safe_addstr(controls_start, 0, "Controls:")
                    if controls_start + 1 < height:
                        safe_addstr(controls_start + 1, 0, "  1-7: Select joint")
                    if controls_start + 2 < height:
                        safe_addstr(controls_start + 2, 0, "  ←/- or →/+: Decrease/Increase")
                    if controls_start + 3 < height:
                        safe_addstr(controls_start + 3, 0, "  ↑/k or ↓/j: Adjust step size")
                    if controls_start + 4 < height:
                        safe_addstr(controls_start + 4, 0, "  r: Reset to current, z: Zero, q: Quit")
                
                stdscr.refresh()
                last_print = now
            
            # Sleep to maintain reasonable loop rate
            elapsed = time.monotonic() - t0
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)
    
    finally:
        # Clean shutdown
        try:
            current_pos = robot.get_joint_pos()
            robot.command_joint_pos(current_pos)  # Hold current position
            robot.close()
            print("\nRobot closed safely.")
        except Exception as e:
            print(f"\nError during shutdown: {e}")


if __name__ == "__main__":
    curses.wrapper(main)
