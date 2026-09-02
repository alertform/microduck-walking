"""Run mjlab play with the sparse Jacobian solver.

The dense solver's tile_cholesky kernel fails to build on RTX 4060 (sm_89,
insufficient shared memory), so training uses --env.sim.mujoco.jacobian sparse.
play.py builds its env cfg from the task registry and never reads the training
run's env.yaml, so the override has to be reapplied here.
"""

import sys

import mjlab.scripts.play as play_mod

_original_load_env_cfg = play_mod.load_env_cfg


def _load_env_cfg_sparse(*args, **kwargs):
    cfg = _original_load_env_cfg(*args, **kwargs)
    cfg.sim.mujoco.jacobian = "sparse"
    print("[play_sparse] forced sim.mujoco.jacobian = sparse", flush=True)
    return cfg


play_mod.load_env_cfg = _load_env_cfg_sparse

if __name__ == "__main__":
    sys.exit(play_mod.main())
