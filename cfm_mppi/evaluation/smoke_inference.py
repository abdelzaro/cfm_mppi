"""Check pretrained CFM + MPPI inference without ROS or Arena.

From the repository root:
    python -m cfm_mppi.evaluation.smoke_inference --device cpu
"""
import argparse
import json
from pathlib import Path
import pickle
import time

import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig, synthesize_control
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import stage_cost, terminal_cost, unicycle_dynamics
from cfm_mppi.utils import AgentHistory


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dataset", choices=["ucy", "sdd"], default="ucy")
    parser.add_argument("--scenario", type=int, default=110)
    parser.add_argument("--samples", type=int, default=20,
                        help="20 for a smoke test; 200 matches the evaluator")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1,
                        help="Consecutive calls using the evaluator's shrinking horizon (1-80)")
    parser.add_argument("--empty-scene", action="store_true",
                        help="Use zero obstacle tracks instead of the recorded pedestrians")
    parser.add_argument("--check-reset", action="store_true",
                        help="Clear all planning state, replay, and compare control sequences")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "output_dir/cfm_transformer/checkpoint.pth")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.samples < 10 or args.samples % 5:
        parser.error("--samples must be at least 10 and divisible by 5")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not 1 <= args.steps <= 80:
        parser.error("--steps must be between 1 and 80")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or your working CUDA environment")
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(f"Loading pretrained model on {device}; {args.samples} candidates, horizon 80",
          flush=True)

    model = TransformerModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    ego = torch.load(root / f"dataset/eval80_ego_{args.dataset}.pt",
                     map_location="cpu", weights_only=False)
    with (root / f"dataset/eval80_obs_{args.dataset}.pkl").open("rb") as f:
        obstacles = pickle.load(f)
    if not 0 <= args.scenario < min(len(ego), len(obstacles)):
        parser.error("--scenario outside the available dataset")
    observation = obstacles[args.scenario]
    observation = observation[:, ~torch.isnan(observation).any(dim=(0, 2, 3))]
    if args.empty_scene:
        observation = observation[:, :0]
    positions = observation[:, :, :2, 0].to(device=device, dtype=torch.float32)
    velocities = observation[:, :, 2:4, 0].to(device=device, dtype=torch.float32)
    goal = ego[args.scenario, :2, -1].reshape(1, 2).to(device=device, dtype=torch.float32)
    state = torch.zeros(1, 3, device=device)
    if not all(torch.isfinite(x).all().item() for x in (state, goal, positions, velocities)):
        raise ValueError("Non-finite observation")

    horizon = 80
    config = CFMConfig(
        ode_times=[0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
        dt=0.1,
        agent_radius=0.5,
        space_scale=10.0,
        safe_margin_coefs=[0.1, 0.3, 0.5, 0.7, 0.9],
        goal_margin_coef=0.1,
        device=device_name,
    )
    bounds = torch.tensor([2.0, 2.0])
    print(f"Checkpoint loaded; planning against {positions.shape[1]} obstacle tracks",
          flush=True)

    def episode(label):
        # A reset creates fresh solver caches, history buffers, candidates, and
        # simulated state. The loaded pretrained model remains resident.
        torch.manual_seed(0)
        solver = FlowMPPI(
            num_samples=args.samples, dim_state=3, dim_control=2,
            dynamics=unicycle_dynamics, stage_cost=stage_cost, terminal_cost=terminal_cost,
            u_min=-bounds, u_max=bounds, sigmas=torch.tensor([0.3, 0.6]),
            lambda_=0.1, goal=goal.squeeze(0), horizon=horizon, dt=0.1,
            device=device, dynamics_type="unicycle",
        )
        histories = {key: AgentHistory(max_length=10) for key in (
            "ego_state", "ego_control_sin", "obs_state", "obs_control")}
        if solver.prev_optimal_action_seq is not None or any(len(h) for h in histories.values()):
            raise AssertionError("Episode did not start with clean planner state")
        state = torch.zeros(1, 3, device=device)
        noise = torch.randn(args.samples, 2, horizon, device=device)
        noise_level = torch.zeros(1, device=device)
        sequence_outputs, records = [], []
        for step in range(args.steps):
            pos = observation[:, :, :2, step].to(device=device, dtype=torch.float32)
            vel = observation[:, :, 2:4, step].to(device=device, dtype=torch.float32)
            history_len = len(histories["ego_state"])
            current_horizon = noise.shape[-1]
            config.ode_times = ([0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
                                if step == 0 else [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            # Guidance uses torch.func.grad; preserve functional differentiation.
            controls, planar_controls = synthesize_control(
                model, solver, config, state, goal, noise, noise_level,
                pos, vel, current_horizon, histories=histories, d=0.1,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            expected_shape = (1, 2, current_horizon - history_len)
            if controls.shape != expected_shape or planar_controls.shape != (1, 2, current_horizon):
                raise ValueError(f"Unexpected output shapes: {controls.shape}, {planar_controls.shape}")
            if not torch.isfinite(controls).all() or not torch.isfinite(planar_controls).all():
                raise ValueError("Inference returned non-finite controls")
            if (controls.abs() > 2.0 + 1e-5).any():
                raise ValueError("Refined controls exceed the configured bounds")
            sequence_outputs.append(controls.detach().cpu())
            first = controls[0, :, 0].detach().cpu().tolist()
            records.append({"step": step, "history_length": history_len,
                            "controls_shape": list(controls.shape),
                            "planar_controls_shape": list(planar_controls.shape),
                            "first_control": {"linear_x_m_s": first[0], "angular_z_rad_s": first[1]},
                            "inference_seconds": elapsed})
            print(f"{label}: step {step + 1}/{args.steps}, history={history_len}, "
                  f"future={controls.shape[-1]}, time={elapsed:.3f}s", flush=True)
            state = unicycle_dynamics(state, controls[:, :, 0], config.dt)
            if not torch.isfinite(state).all():
                raise ValueError("Non-finite numerical rollout state")
            # Preserve the reference evaluator's warm-start and history updates.
            fresh_noise = torch.randn(args.samples, 2, current_horizon, device=device)
            noise = 0.8 * planar_controls / config.space_scale + 0.2 * fresh_noise
            noise = noise[:, :, history_len + 1:]
            histories["ego_control_sin"].update(planar_controls[:, :, history_len])
            histories["ego_state"].update(state)
            histories["obs_state"].update(pos)
            histories["obs_control"].update(vel)
            past_controls = histories["ego_control_sin"].get()
            noise = torch.cat((past_controls.expand(args.samples, -1, -1) / config.space_scale,
                               noise), dim=-1)
            noise_level = torch.tensor([0.8], device=device)
        return records, sequence_outputs, state.detach().cpu().tolist()[0]

    records, outputs, final_state = episode("episode")
    reset_matches = None
    if args.check_reset:
        _, replay_outputs, _ = episode("reset replay")
        reset_matches = all(torch.allclose(a, b, atol=1e-5, rtol=1e-5)
                            for a, b in zip(outputs, replay_outputs))
        if not reset_matches:
            raise AssertionError("Reset replay differs from the clean episode")
    first_record = records[0]
    result = {
        "success": True, "checkpoint": str(args.checkpoint.resolve()),
        "device": device_name, "dataset": args.dataset, "scenario": args.scenario,
        "samples": args.samples, "horizon": horizon, "dt_s": config.dt,
        "obstacle_count": positions.shape[1], "goal_xy": goal.cpu().tolist()[0],
        "controls_shape": first_record["controls_shape"],
        "planar_controls_shape": first_record["planar_controls_shape"],
        "first_control": first_record["first_control"],
        "inference_seconds": first_record["inference_seconds"],
        "steps": args.steps, "empty_scene": args.empty_scene,
        "reset_replay_matches": reset_matches, "calls": records,
        "numerical_rollout_final_state": final_state,
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "calls"},
                     indent=2, allow_nan=False), flush=True)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
