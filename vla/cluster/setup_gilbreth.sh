#!/bin/bash
# One-time environment setup on Gilbreth (run on a login node).
# Bakes in every install gotcha we debugged locally: mujoco MUST be 2.3.x
# for robosuite 1.4.1, bddl needs `future`, LIBERO installs via a .pth to
# the cloned repo (its pip wheel is empty), SB3 needs tensorboard.
# Headless rendering on cluster GPUs uses MUJOCO_GL=egl.
set -e  # (no -u: conda's activate hooks use unbound vars)
WORK=${1:-$CLUSTER_SCRATCH/vla}
mkdir -p "$WORK" && cd "$WORK"

module load anaconda 2>/dev/null || module load conda
[ -d "$WORK/env" ] || conda create -y -p "$WORK/env" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "robosuite==1.4.1" "bddl==1.0.1" "mujoco==2.3.7" \
    easydict "gym==0.25.2" future matplotlib opencv-python h5py \
    gymnasium stable_baselines3 tensorboard huggingface_hub

git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$WORK/LIBERO" > "$SITE/libero_repo.pth"
echo "N" | python -c "import libero.libero" || true  # writes ~/.libero/config.yaml

cat > "$WORK/smoke_render.py" <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless EGL on cluster GPUs
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_object"]()
t = bm.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(
    get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
    camera_heights=128, camera_widths=128)
obs = env.reset()
print("render OK:", obs["agentview_image"].shape)
env.close()
PY
echo "setup done."
echo "NEXT (on a GPU node): srun -A joecamp --gpus=1 python $WORK/smoke_render.py"
