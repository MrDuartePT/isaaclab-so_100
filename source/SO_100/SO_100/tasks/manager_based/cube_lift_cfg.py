# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses

from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils  # For the debug visualization
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from ament_index_python.packages import get_package_share_directory

from SO_100.config.so_101_base_env import SO100BaseEnvCfg
from SO_100.config.so_101_cfg import SO101_CFG, SO101_RGB_SENSOR, SO101_DEPTH_SENSOR # isort: skip

from SO_100.config import mdp
from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from .SO100 import SO100_CFG  # Corrected import # isort: skip


##
# MDP settings
##s

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching reward with lower weight
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2)

    # Lifting reward with higher weight
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=25.0)

    # Action penalty to encourage smooth movements
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # Joint velocity penalty to prevent erratic movements
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


# @configclass
class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     # Stage 1: Focus on reaching
#     # Start with higher reaching reward, then gradually decrease it
#     reaching_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 6000}
#     )

#     # Stage 2: Transition to lifting
#     # Start with lower lifting reward, gradually increase to encourage lifting behavior
#     lifting_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "lifting_object", "weight": 35.0, "num_steps": 8000}
#     )

    # Stage 4: Stabilize the policy
    # Gradually increase action penalties to encourage smoother, more stable movements
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    )


##
# Environment configuration
##
@configclass
class SO100LiftEnvCfg(SO100BaseEnvCfg):
    """Configuration for the lifting environment."""
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

@configclass
class SO100CubeLiftEnvCfg(SO100LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SO101 as robot
        _robot_cfg = dataclasses.replace(SO101_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        # Set initial rotation if needed
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        self.scene.robot = _robot_cfg

        self.scene.robot_rgb = dataclasses.replace(SO101_RGB_SENSOR, prim_path="{ENV_REGEX_NS}/Robot/gripper/camera_bottom_screw_frame/camera_link/Camera_RGB")
        self.scene.robot_depth = dataclasses.replace(SO101_DEPTH_SENSOR, prim_path="{ENV_REGEX_NS}/Robot/gripper/camera_bottom_screw_frame/camera_link/Camera_Depth")

        # Set actions for the specific robot type (SO100)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True
        )

        # Set gripper action with wider range for better visibility
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            open_command_expr={"Jaw": 0.5},  # Fully open
            close_command_expr={"Jaw": 0.0}  # flly closed
        )
        
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper"
        # Disable debug visualization for the target pose command
        self.commands.object_pose.debug_vis = False

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Configure end-effector marker
        marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        # Properly replace the frame marker configuration
        marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            )
        }
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        # Updated FrameTransformerCfg for alternate USD structure
        self.scene.ee_frame = FrameTransformerCfg(
            # Original path in comments for reference
            # prim_path="{ENV_REGEX_NS}/Robot/SO_100/SO_5DOF_ARM100_05d_SLDASM/Base",
            # Updated path for the new USD structure
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,  # Enable visualization
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # Original path in comments for reference
                    # prim_path="{ENV_REGEX_NS}/Robot/SO_100/SO_5DOF_ARM100_05d_SLDASM/Fixed_Gripper",
                    # Updated path for the new USD structure
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.1),
                        rot = (0.0, -0.7071, 0.7071, 0.0)
                    ),
                ),
            ],
        )

        # Configure cube marker with different color and path
        cube_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        cube_marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        }
        cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
        
        self.scene.marker = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            debug_vis=True,
            visualizer_cfg=cube_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="cube",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

@configclass
class SO100CubeLiftEnvCfg_PLAY(SO100CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False