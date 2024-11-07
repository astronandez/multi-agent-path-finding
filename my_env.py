from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

num_lanes = 4

class MyEnv(AbstractEnv):
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 8,  # [s]
                "destination": "o1",
                "controlled_vehicles": 2,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:

        lane_width = AbstractLane.DEFAULT_WIDTH     # DEFAULT_WIDTH = 4
        
        right_turn_radius = num_lanes * lane_width + 5      # radius of the left lane
        
        left_turn_radius = right_turn_radius + lane_width   # radius of the left lane
        
        outer_distance = right_turn_radius + lane_width / 2 
        access_length = 50 + 50

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        # TODO: if num_lanes = n > 1, consider whether we need to match each lane, so that the lanes in the intersection would be n*n?
        # TODO: By the way, it means how to define lanes in intersection?

        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            for lane_index in range(num_lanes):
                lane_offset = lane_width * lane_index  # Each lane offset

                # Incoming lanes
                start = rotation @ np.array(
                    [lane_offset + lane_width / 2, access_length + outer_distance]
                )
                end = rotation @ np.array([lane_offset + lane_width / 2, outer_distance])

                # Judge line type
                if num_lanes == 1:
                    line_type = [c, c]
                elif lane_index == 0:
                    line_type = [c, s]
                elif lane_index < num_lanes - 1:
                    line_type = [s, s]
                else:
                    line_type = [s, c]
                
                net.add_lane(
                    # TODO: maybe need to add str(lane_index) to start&end point name
                    "o" + str(corner) + "_" + str(lane_index),
                    "ir" + str(corner) + "_" + str(lane_index),
                    StraightLane(
                        # TODO: judge to determine line_types, not just left-s, right-c
                        start, end, line_types=line_type, priority=priority, speed_limit=10
                    ),
                )


                # Right turn
                r_center = rotation @ (np.array([outer_distance, outer_distance]))

                # Judge line type
                if num_lanes == 1:
                    line_type = [n, c]
                elif lane_index < num_lanes - 1:
                    line_type = [n, n]
                else:
                    line_type = [n, c]

                net.add_lane(
                    # TODO: maybe need to add str(lane_index) to start&end point name
                    "ir" + str(corner) + "_" + str(lane_index),
                    "il" + str((corner - 1) % 4) + "_" + str(lane_index),
                    CircularLane(
                        r_center,
                        # TODO: check the logic: + lane_offset or - lane_offset
                        right_turn_radius - lane_offset,  # Radius
                        angle + np.radians(180),
                        angle + np.radians(270),
                        # TODO: judge to determine line_types, not just left-n, right-c
                        line_types=line_type,
                        priority=priority,
                        speed_limit=10,
                    ),
                )

                # Left turn
                # TODO: check the center point
                l_center = rotation @ (
                    np.array(
                        [
                            -left_turn_radius + lane_offset + lane_width / 2,
                            left_turn_radius - lane_offset - lane_width / 2,
                        ]
                    )
                )
                net.add_lane(
                    # TODO: maybe need to add str(lane_index) to start&end point name
                    "ir" + str(corner) + "_" + str(lane_index),
                    "il" + str((corner + 1) % 4) + "_" + str(lane_index),
                    CircularLane(
                        l_center,
                        # TODO: check the logic: + lane_offset or - lane_offset
                        left_turn_radius + lane_offset,
                        angle + np.radians(0),
                        angle + np.radians(-90),
                        clockwise=False,
                        line_types=[n, n],
                        priority=priority - 1,
                        speed_limit=10,
                    ),
                )

                # Straight lane (for additional lanes in intersection)
                start = rotation @ np.array([lane_offset + lane_width / 2, outer_distance])
                end = rotation @ np.array([lane_offset + lane_width / 2, -outer_distance])
                net.add_lane(
                    # TODO: maybe need to add str(lane_index) to start&end point name
                    "ir" + str(corner) + "_" + str(lane_index),
                    "il" + str((corner + 2) % 4) + "_" + str(lane_index),
                    StraightLane(
                        # TODO: judge to determine line_types, not left-s, right-n
                        start, end, line_types=[n, n], priority=priority, speed_limit=10
                    ),
                )

                # Exit lanes(Right side of road)
                start = rotation @ np.flip(
                    [lane_offset + lane_width / 2, access_length + outer_distance], axis=0
                )
                end = rotation @ np.flip([lane_offset + lane_width / 2, outer_distance], axis=0)

                # Judge line type
                if num_lanes == 1:
                    line_type = [c, c]
                elif lane_index == 0:
                    line_type = [c, s]
                elif lane_index < num_lanes - 1:
                    line_type = [s, s]
                else:
                    line_type = [s, c]
                
                
                net.add_lane(
                    # TODO: maybe need to add str(lane_index) to start&end point name
                    "il" + str((corner - 1) % 4) + "_" + str(lane_index),
                    "o" + str((corner - 1) % 4) + "_" + str(lane_index),
                    StraightLane(
                        # TODO: judge to determine line_types, not left-n, right-c
                        end, start, line_types=line_type, priority=priority, speed_limit=10
                    ),
                )

        # 设置道路网络
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road


    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """

        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            lane = self.np_random.integers(1, num_lanes)
            ego_lane = self.road.network.get_lane(
                (f"o{ego_id % 4}_{lane}", f"ir{ego_id % 4}_{lane}", 0)
            )

            # destination = self.config["destination"] or "o" + str(self.np_random.integers(1, 4)) + "_" + str(self.np_random.integers(1, num_lanes))
            destination = self.config["destination"] + f"_{lane}"
            
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        lane = self.np_random.choice(range(num_lanes), size=2, replace=False)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            (f"o{route[0]}_{lane[0]}", f"ir{route[0]}_{lane[0]}", 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to(f"o{route[1]}_{lane[1]}")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )

