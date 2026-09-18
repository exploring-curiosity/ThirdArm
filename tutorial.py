import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.switch import Switch
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

load_dotenv(Path(__file__).with_name('.env'))


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to the .env file next to tutorial.py.")
    return value


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=_require_env('VIAM_API_KEY'),
        api_key_id=_require_env('VIAM_API_KEY_ID')
    )

    return await RobotClient.at_address(_require_env('VIAM_MACHINE_ADDRESS'), opts)

# The arm-position-saver switch exposes three positions:
#   0 = idle, 1 = update config (overwrites the saved pose!), 2 = go to saved pose.
# Only ever send 2 — sending 1 would clobber the pose saved in the Viam app.
GO_TO = 2


async def goto_saved_pose(machine, name):
    """Drive an arm-position-saver switch to its saved pose.

    set_position(2) blocks until the arm finishes moving, and the module
    returns itself to idle (0) on completion.
    """
    switch = Switch.from_robot(machine, name)
    print(f"moving to '{name}'...")
    await switch.set_position(GO_TO)
    print(f"reached '{name}'.")


async def main():
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")

        # Start from a known state: open gripper, arm at start-position.
        print("opening gripper...")
        await gripper.open()
        await goto_saved_pose(machine, "start-position")

        # Reach the cuboid and settle before closing on it.
        await goto_saved_pose(machine, "reach-cuboid")
        pose = await arm.get_end_position()
        print(f"  pose at reach-cuboid: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f}")

        print("holding for 2s...")
        await asyncio.sleep(1)

        # Pick up the cuboid.
        print("grabbing...")
        grabbed = await gripper.grab()
        if not grabbed:
            print("  WARNING: grab() reported nothing grasped — continuing anyway.")
        else:
            print("  grabbed.")

        # Carry it back to start.
        await goto_saved_pose(machine, "start-position")

        # Return it and let go.
        await goto_saved_pose(machine, "reach-cuboid")
        print("releasing...")
        await asyncio.sleep(1)
        await gripper.open()
        print("  released.")

        # Retreat to start.
        await goto_saved_pose(machine, "start-position")

        pose = await arm.get_end_position()
        print(f"  pose back at start-position: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f}")


if __name__ == '__main__':
    asyncio.run(main())
