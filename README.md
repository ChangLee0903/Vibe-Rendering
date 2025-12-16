# Vibe-Rendering

### Folder layout

* Put all meshes under `inputs/` (e.g., `inputs/dolphin.obj`, `inputs/teapot.obj`, `inputs/Tree.obj`). 
* All rendered images are written to `outputs/`. Each script will create this folder if it does not exist. 

---

### What each script does

#### 1) Benchmark results (Ours vs CLIP-only)

* `run.sh` runs **our method** on the 3 benchmark objects (dolphin, teapot, tree) and saves:

  * `outputs/dolphin_ours.png`
  * `outputs/teapot_ours.png`
  * `outputs/tree_ours.png` 
* `run_bas.sh` runs the **CLIP-only baseline** on the same 3 objects and saves:

  * `outputs/dolphin_clip.png`
  * `outputs/teapot_clip.png`
  * `outputs/tree_clip.png` 

These two scripts together produce the benchmark comparison figure inputs (top row: ours, bottom row: CLIP-only) using the same target prompts per object.  

**Run:**

```bash
bash run.sh
bash run_bas.sh
```

---

#### 2) Ablation: Concrete vs Abstract prompt

* `run_com.sh` runs **two prompts on the teapot only**, saving:

  * `outputs/teapot_con.png` (concrete prompt)
  * `outputs/teapot_abs.png` (abstract prompt) 

**Run:**

```bash
bash run_com.sh
```

---

#### 3) Ablation: Prompt length extension

* `run_ext.sh` runs **three progressively longer prompts** on the teapot only, saving:

  * `outputs/teapot_ext1.png`
  * `outputs/teapot_ext2.png`
  * `outputs/teapot_ext3.png` 

**Run:**

```bash
bash run_ext.sh
```

---

### Output naming convention (for later comparisons)

* Our method outputs are named `*_ours.png`. 
* Baseline outputs are named `*_clip.png`. 
* Ablation outputs follow:

  * Concrete vs Abstract: `*_con.png`, `*_abs.png`. 
  * Length extension: `*_ext1.png` to `*_ext3.png`. 
