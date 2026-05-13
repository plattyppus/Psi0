import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from psi.utils import parse_args_to_tyro_config, seed_everything
from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.models.psi0 import Psi0Model


def main():
    parser = argparse.ArgumentParser(description="Open-loop evaluation on a single random episode")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to the training run directory (e.g. runs/finetune/...)")
    parser.add_argument("--ckpt-step", type=int, default=30000,
                        help="Checkpoint step to load")
    parser.add_argument("--stride", type=int, default=4,
                        help="Frame stride for sampling within the episode")
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="Number of diffusion inference steps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # ---- Load config ----
    config_ = parse_args_to_tyro_config(run_dir / "argv.txt")
    conf = (run_dir / "run_config.json").read_text()
    launch_config = config_.model_validate_json(conf)

    seed_everything(launch_config.seed or 42)

    # ---- Load model ----
    # Point QWEN3VL_VARIANT to the local model path used during training,
    # so that config/processor are loaded from disk instead of HuggingFace.
    import psi.models.psi0 as psi0_module
    psi0_module.QWEN3VL_VARIANT = launch_config.model.model_name_or_path

    # Fall back to sdpa when flash_attn is not installed.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import \
        Qwen3VLForConditionalGeneration as _Qwen3VLCls
    _qwen3vl_init = _Qwen3VLCls.__init__
    def _patched_qwen3vl_init(self, config):
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                config._attn_implementation = "sdpa"
        _qwen3vl_init(self, config)
    _Qwen3VLCls.__init__ = _patched_qwen3vl_init

    psi0 = Psi0Model.from_pretrained(run_dir, args.ckpt_step, launch_config, device=args.device)
    psi0.to(args.device)
    psi0.eval()
    print(f"Model loaded from {run_dir}, checkpoint step {args.ckpt_step}")

    # ---- Load data config ----
    data_cfg: LerobotDataConfig = launch_config.data
    model_cfg: Psi0ModelConfig = launch_config.model
    maxmin = data_cfg.transform.field

    transform_kwargs = dict(vlm_processor=psi0.vlm_processor)
    dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)

    meta = dataset.raw_dataset.meta
    print(f"Dataset: {meta.repo_id}, episodes={meta.total_episodes}, frames={meta.total_frames}")

    # ---- Pick a random episode ----
    eps_idx = np.random.randint(0, meta.total_episodes)
    start_frame_idx = dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
    end_frame_idx = dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
    print(f"Episode {eps_idx}: frames [{start_frame_idx}, {end_frame_idx}), "
          f"total {end_frame_idx - start_frame_idx} frames")

    # ---- Run evaluation ----
    labels_denormed = [
        "hand_joints",
        "arm_joints",
        "torsor_roll",
        "torsor_pitch",
        "torsor_yaw",
        "height",
        "vx",
        "vy",
        "torso_vyaw",
        "target_yaw",
    ]
    group_slices = [14, 28, 29, 30, 31, 32, 33, 34, 35]

    avg_action_errors_denormed_list = []

    from tqdm import tqdm

    total_steps = len(range(start_frame_idx, end_frame_idx, args.stride))
    pbar = tqdm(total=total_steps, desc="Evaluating", unit="step")

    for idx, i in enumerate(range(start_frame_idx, end_frame_idx, args.stride)):
        frame = dataset[i]
        images = frame["raw_images"]
        instruction = frame["instruction"]
        states = torch.from_numpy(frame["states"]).unsqueeze(0).to(args.device)
        gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(args.device)

        pred_actions = psi0.predict_action(
            observations=[images],
            states=states,
            instructions=[instruction],
            num_inference_steps=args.num_inference_steps,
            traj2ds=None,
        )

        denormalized_pred_actions = maxmin.denormalize(pred_actions)
        error = (denormalized_pred_actions - gt_action).detach().abs().cpu().numpy()
        error = error.reshape(-1, gt_action.shape[-1])  # (Tp, Da)
        avg_action_errors_denormed_list.append(error.mean(0))

        pbar.update(1)
        pbar.set_postfix(frame=f"{i}/{end_frame_idx}")

    pbar.close()

    avg_action_errors_denormed_list = np.stack(avg_action_errors_denormed_list, axis=0)  # (N, Da)
    avg_action_errors_denormed = avg_action_errors_denormed_list.mean(axis=0)
    avg_action_errors_denormed_split = np.split(avg_action_errors_denormed, group_slices, axis=-1)

    # ---- Print results ----
    print("\nDenormalized L1 errors by dimension group:\n")
    for label, group in zip(labels_denormed, avg_action_errors_denormed_split):
        print(f"  {label:20s} {str(group.shape):10s}  {np.linalg.norm(group):.6f}")

    # ---- Plot ----
    if not args.no_plot:
        error_groups = np.split(avg_action_errors_denormed_list, group_slices, axis=-1)
        per_label_curves = [np.linalg.norm(g, axis=-1) for g in error_groups]
        curve_map = dict(zip(labels_denormed, per_label_curves))

        plot_groups = [
            ("hand_joints + arm_joints (rad)", ["hand_joints", "arm_joints"]),
            ("torso rpy (rad)", ["torsor_roll", "torsor_pitch", "torsor_yaw"]),
            ("height (m)", ["height"]),
            ("vx + vy (m/s)", ["vx", "vy"]),
            ("target_yaw (rad)", ["target_yaw"]),
        ]

        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        for ax, (title, keys) in zip(axes, plot_groups):
            for key in keys:
                if key in curve_map:
                    ax.plot(curve_map[key], label=key)
            ax.set_title(title)
            ax.set_ylabel("Error norm")
            ax.grid(True, alpha=0.3)
            if len(keys) > 1:
                ax.legend()
        axes[-1].set_xlabel(f"Sample step in episode (stride={args.stride})")
        plt.tight_layout()
        plot_path = run_dir / f"openloop_eval_eps{eps_idx}_step{args.ckpt_step}.png"
        plt.savefig(plot_path, dpi=100)
        print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
