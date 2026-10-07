import h5py, numpy as np, torch, sys
sys.path.insert(0, "/scratch/gilbreth/nam120/vla/code")
from PIL import Image
from eval_openvla import load_teacher, UNNORM_KEY
lora = sys.argv[1] if sys.argv[1] != "base" else None
RV = "/scratch/gilbreth/nam120/vla/runs_vla"
model, processor = load_teacher(lora, f"{RV}/openvla_seen5/norm_stats.json")
f = h5py.File("/scratch/gilbreth/nam120/vla/LIBERO/libero/datasets/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5", "r")
g = f["data"]["demo_0"]
lang = "pick up the ketchup and place it in the basket"
prompt = f"In: What action should the robot take to {lang}?\nOut:"
preds, errs = [], []
for t in range(0, 120, 10):
    img = Image.fromarray(g["obs"]["agentview_rgb"][t]).resize((224, 224))
    inputs = processor(prompt, img).to("cuda", dtype=torch.bfloat16)
    a = np.asarray(model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False))
    preds.append(a); errs.append(np.abs(a - g["actions"][t]).mean())
preds = np.stack(preds)
print("VARIATION across frames (std per dim):", np.round(preds.std(0), 3))
print("mean |pred-demo|:", round(float(np.mean(errs)), 3))
