#!/usr/bin/env python3
"""Run lm-eval evaluation with TitanWrapper"""

import argparse
import os
import sys

# REPO_ROOT is the benchmark directory
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
# Add torchtitan to path for imports
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run lm-eval evaluation with TorchTitan model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/data_hss/torchtitian/train_configs/llama2_pc_layer.toml",
        help="Path to torchtitan training TOML config."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=61100,
        help="Checkpoint step number to load."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for loading model."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="lambada_openai,hellaswag",
        help="Comma-separated list of lm-eval tasks to evaluate."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples for testing (None = all samples)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lm_eval_results",
        help="Directory to save evaluation results."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LM-Eval Evaluation with TorchTitan Model")
    print("=" * 60)

    # Load lm_eval from local harness
    harness_dir = os.path.join(REPO_ROOT, "lm-evaluation-harness")
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)

    # Import lm_eval
    from lm_eval import evaluator, tasks

    # Import torchtitan modules
    from torchtitan.lm_eval_wrapper import build_titan_model_and_tokenizer, TitanWrapper

    # Parse tasks
    task_names = args.tasks.split(",")
    print(f"[Selected Tasks]: {task_names}")

    # Load model and tokenizer
    print(f"\n[1/3] Loading model and tokenizer from step {args.step}...")
    model, tokenizer = build_titan_model_and_tokenizer(
        checkpoint_path=None,
        tokenizer_path=None,
        config_path=args.config_path,
        device=args.device,
        dtype="auto",
        strict=False,
        step=args.step,
    )
    print(f"  Model: {type(model).__name__}")
    print(f"  Tokenizer: {type(tokenizer).__name__}")

    # Wrap with TitanWrapper
    print("\n[2/3] Wrapping model with TitanWrapper...")
    wrapper = TitanWrapper(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_seq_len=8192,
        max_gen_toks=50,
        add_bos=True,
    )
    print("  TitanWrapper created")

    # Run evaluation
    print(f"\n[3/3] Running evaluation on: {args.tasks}...")

    results = evaluator.simple_evaluate(
        model=wrapper,
        model_args={},
        tasks=task_names,
        num_fewshot=0,
        batch_size=1,
        device=args.device,
        limit=args.num_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\nTask: {task}")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_step{args.step}.json")
    import json

    # Custom encoder to handle non-serializable objects
    class SafeEncoder(json.JSONEncoder):
        def default(self, o):
            if callable(o):
                return str(o)
            return super().default(o)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=SafeEncoder)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
