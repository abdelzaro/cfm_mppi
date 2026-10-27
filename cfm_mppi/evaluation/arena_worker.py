"""Persistent pretrained inference over newline-delimited JSON; no ROS imports.

Every request starts a fresh 80-step plan. No shrinking-horizon history or MPPI
cache crosses requests. Requests optionally supply map-frame obstacle positions
and velocities; omitted observations preserve the empty-scene baseline.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig, synthesize_control
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import stage_cost, terminal_cost, unicycle_dynamics


def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def rto_response_dynamics(state, action, delta_t=0.1):
    """Predict RTO's existing Kp=5, acceleration-limit=5 velocity response.

    State is [x,y,yaw,world_vx,world_vy,w]. This changes prediction only.
    Other worker request modes retain the repository's ideal unicycle model.
    """
    dt = delta_t / 5
    for _ in range(5):
        c, s = torch.cos(state[:, 2]), torch.sin(state[:, 2])
        body_x = c * state[:, 3] + s * state[:, 4]
        body_y = -s * state[:, 3] + c * state[:, 4]
        ax = torch.clamp(5 * (action[:, 0] - body_x), -5, 5)
        ay = torch.clamp(-5 * body_y, -5, 5)
        vx = state[:, 3] + (c * ax - s * ay) * dt
        vy = state[:, 4] + (s * ax + c * ay) * dt
        w = state[:, 5] + torch.clamp(5 * (action[:, 1] - state[:, 5]), -5, 5) * dt
        angle = state[:, 2] + w * dt
        state = torch.stack((state[:, 0] + vx * dt, state[:, 1] + vy * dt,
                             torch.atan2(torch.sin(angle), torch.cos(angle)), vx, vy, w), dim=1)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[2]
                        / "output_dir/cfm_transformer/checkpoint.pth")
    args = parser.parse_args()
    if args.samples < 10 or args.samples % 5 or args.threads < 1:
        parser.error("Need samples >= 10 divisible by 5, and positive threads")
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(name)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = TransformerModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    config = CFMConfig(
        ode_times=[0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
        dt=0.1, agent_radius=0.5, space_scale=10.0,
        safe_margin_coefs=[0.1, 0.3, 0.5, 0.7, 0.9], goal_margin_coef=0.1,
        device=name,
    )
    emit({"ready": True, "device": name})
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            state_values, goal_values = request["state"], request["goal"]
            if len(state_values) != 3 or len(goal_values) != 2:
                raise ValueError("Expected state [x,y,yaw] and goal [x,y]")
            if not all(math.isfinite(v) for v in state_values + goal_values):
                raise ValueError("Non-finite observation")
            state = torch.tensor([state_values], device=device, dtype=torch.float32)
            goal = torch.tensor([goal_values], device=device, dtype=torch.float32)
            sequence = request.get("sequence", False)
            bounds = torch.tensor([0.1, 0.3] if sequence else [2.0, 2.0])
            lower = torch.tensor([-0.1 if request.get("reverse", False) else 0.0, -0.3]) if sequence else -bounds
            if "limits" in request:
                limits = request["limits"]
                if (not sequence or len(limits) != 3 or not all(math.isfinite(v) for v in limits)
                        or not -0.75 <= limits[0] <= 0 < limits[1] <= 0.75
                        or not 0 < limits[2] <= 1.0):
                    raise ValueError("Invalid configured RTO velocity limits")
                lower = torch.tensor([limits[0], -limits[2]])
                bounds = torch.tensor(limits[1:])
            dynamics, dim_state = unicycle_dynamics, 3
            if request.get("dynamics") == "rto_response":
                velocity = request.get("velocity", [])
                if len(velocity) not in (2, 3) or not all(math.isfinite(v) for v in velocity):
                    raise ValueError("RTO prediction needs measured body [vx,vy,w]")
                if len(velocity) == 2:
                    velocity = [velocity[0], 0.0, velocity[1]]
                c, s = math.cos(state_values[2]), math.sin(state_values[2])
                world = [c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1], velocity[2]]
                state = torch.tensor([state_values + world], device=device, dtype=torch.float32)
                dynamics, dim_state = rto_response_dynamics, 6
            solver = FlowMPPI(
                num_samples=args.samples, dim_state=dim_state, dim_control=2,
                dynamics=dynamics, stage_cost=stage_cost, terminal_cost=terminal_cost,
                u_min=lower, u_max=bounds, sigmas=torch.tensor([0.3, 0.6]), lambda_=0.1,
                goal=goal.squeeze(0), horizon=80, dt=0.1, device=device,
                dynamics_type="unicycle",
            )
            noise = torch.randn(args.samples, 2, 80, device=device)
            observations = request.get("obstacles", [])
            if (len(observations) > 64 or any(len(o) != 4 or not all(math.isfinite(v) for v in o)
                                              for o in observations)):
                raise ValueError("Expected at most 64 finite [x,y,vx,vy] obstacle tracks")
            radius = float(request.get("collision_radius", 0.5))
            if not 0.225 <= radius <= 1.5:
                raise ValueError("Invalid collision radius")
            config.agent_radius = radius
            observations = torch.tensor(observations, device=device, dtype=torch.float32).reshape(1, -1, 4)
            if name == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            controls, _ = synthesize_control(
                model, solver, config, state, goal, noise, torch.zeros(1, device=device),
                observations[:, :, :2], observations[:, :, 2:], 80, d=0.1,
            )
            if name == "cuda":
                torch.cuda.synchronize()
            if controls.shape != (1, 2, 80) or not torch.isfinite(controls).all():
                raise ValueError("Invalid control sequence")
            action = controls[0, :, 0].detach().cpu().tolist()
            response = {"ok": True, "id": request["id"], "epoch": request["epoch"],
                        "action": action, "inference_seconds": time.perf_counter() - started,
                        "obstacle_count": observations.shape[1], "collision_radius": radius,
                        "moving_obstacle_count": int((torch.linalg.vector_norm(observations[:, :, 2:], dim=-1) > 0.02).sum())}
            if sequence:
                with torch.no_grad():
                    states = [state]
                    for i in range(80):
                        states.append(dynamics(states[-1], controls[:, :, i], 0.1))
                    response.update(controls=controls[0].transpose(0, 1).cpu().tolist(),
                                    states=torch.cat(states, dim=0)[:, :3].cpu().tolist(), dt=0.1,
                                    dynamics=request.get("dynamics", "unicycle"))
            emit(response)
        except Exception as exc:
            emit({"ok": False, "id": request.get("id"), "epoch": request.get("epoch"),
                  "error": str(exc)})


if __name__ == "__main__":
    main()
