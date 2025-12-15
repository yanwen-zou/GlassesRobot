import time
from pathlib import Path
import sys

import numpy as np

# Ensure we import PyRoki and snippets from the local repo.
here = Path(__file__).resolve()
repo_root = here.parents[4]
gh_root = repo_root / "glasses_hardware" / "i2rt"
pyroki_src = gh_root / "pyroki" / "src"
snippets_root = gh_root / "pyroki_snippets"
for path in (pyroki_src, snippets_root, gh_root, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
sys.modules.pop("pyroki", None)
sys.modules.pop("pyroki_snippets", None)

import pyroki as pk
import pyroki_snippets as pks
import viser
import yourdfpy
from viser.extras import ViserUrdf
from i2rt.robots.utils import YAM_XML_PATH



def main():

    """Main function for basic IK."""


    xml_path = Path(YAM_XML_PATH)          # yam.xml
    urdf_path = xml_path.with_suffix(".urdf")
    assets_root = urdf_path.parent

    def filename_handler(fname: str, **kwargs) -> str:
        if fname.startswith("package://"):
            rel = fname[len("package://"):]
            print(f"Resolved {fname} to assets root:{assets_root / rel}")
            return str((assets_root / rel).resolve())
        return fname

    urdf = yourdfpy.URDF.load(str(urdf_path), filename_handler=filename_handler)


    target_link_name = "link_6"


    # Create robot.

    robot = pk.Robot.from_urdf(urdf)


    # Set up visualizer.

    server = viser.ViserServer()

    server.scene.add_grid("/ground", width=2, height=2)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")


    # Create interactive controller with initial position.

    ik_target = server.scene.add_transform_controls(

        "/ik_target", scale=0.2, position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)

    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)


    while True:

        # Solve IK.

        start_time = time.time()

        solution = pks.solve_ik(

            robot=robot,

            target_link_name=target_link_name,

            target_position=np.array(ik_target.position),

            target_wxyz=np.array(ik_target.wxyz),

        )


        # Update timing handle.

        elapsed_time = time.time() - start_time

        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)


        # Update visualizer.

        urdf_vis.update_cfg(solution)



if __name__ == "__main__":

    main()
