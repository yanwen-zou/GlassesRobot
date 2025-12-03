import time
import numpy as np
def move_piper_to_init_angles():
    """Move Piper arm to a known joint pose before ball calibration."""
    try:
        from piper_sdk import C_PiperInterface_V2
    except Exception as exc:
        print(f"[WARN] piper_sdk not available; skip Piper init. ({exc})")
        return
    try:
        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort()
        while not piper.EnablePiper():
            time.sleep(0.01)
        # target angles in degrees
        target_deg = [-30.0, 18.0, -10.0, 0.0, -2.0, 0.0]
        factor = 57295.7795  # rad -> scaled int (1000 * deg)
        rad = np.deg2rad(target_deg)
        joints = [int(r * factor) for r in rad]
        piper.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        piper.JointCtrl(*joints)
        print(f"[INFO] Moved Piper joints to deg {target_deg}")
    except Exception as exc:
        print(f"[WARN] Piper init move failed: {exc}")

def main():
    import sys
    import time
    import math
    from piper_sdk import C_PiperInterface_V2
    move_piper_to_init_angles()

if __name__ == "__main__":
    main()