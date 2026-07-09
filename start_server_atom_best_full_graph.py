#!/usr/bin/env python3
"""
ATOM server wrapper that forces cudagraph_mode = FULL_AND_PIECEWISE.

ATOM's CLI does not expose --cudagraph-mode, so this patches
Config.__post_init__ to set FULL_AND_PIECEWISE after the normal level-3
piecewise setup is done.

Usage:
    python start_server_atom_best_full_graph.py \
        --model openai/gpt-oss-120b \
        --kv_cache_dtype fp8 \
        --gpu-memory-utilization 0.5 \
        [other EngineArgs / server args]
"""

import sys
from atom.config import Config, CUDAGraphMode, CompilationLevel

# Store the original __post_init__.
_orig_config_post_init = Config.__post_init__


def _patched_config_post_init(self: Config) -> None:
    """Run the normal post-init, then override cudagraph_mode for decode."""
    _orig_config_post_init(self)

    # Only touch level-3 (PIECEWISE) configs. If the backend cannot support
    # full CUDA graphs, ATOM may raise during model_runner.capture_cudagraph().
    if (
        self.compilation_config
        and self.compilation_config.level == CompilationLevel.PIECEWISE
    ):
        self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        print(
            "[wrapper] cudagraph_mode set to FULL_AND_PIECEWISE",
            flush=True,
        )


Config.__post_init__ = _patched_config_post_init

# Import and run the real OpenAI server entry point. The engine has not been
# created yet, so the patched Config.__post_init__ will be in effect.
from atom.entrypoints.openai_server import main

if __name__ == "__main__":
    main()
