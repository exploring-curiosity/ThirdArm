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

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # home-pose
        home_pose = Switch.from_robot(machine, "home-pose")
        home_pose_return_value = await home_pose.get_position()
        print(f"home-pose get_position return value: {home_pose_return_value}")

        # arm
        arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # wall-front
        wall_front = Gripper.from_robot(machine, "wall-front")
        wall_front_return_value = await wall_front.is_moving()
        print(f"wall-front is_moving return value: {wall_front_return_value}")

        # wall-side
        wall_side = Gripper.from_robot(machine, "wall-side")
        wall_side_return_value = await wall_side.is_moving()
        print(f"wall-side is_moving return value: {wall_side_return_value}")

        # ceiling
        ceiling = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

if __name__ == '__main__':
    asyncio.run(main())
