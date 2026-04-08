**EB-YOLO** is an oriented object detection framework for **industrial defect detection under data scarcity**, developed around the study:

> *IndusDet-OBB: An Industrial Benchmark for EB-PBF Oriented Defect Detection under Data Scarcity*

This repository organizes model definitions, dataset utilities, preprocessing scripts, and interpretability tools used to build and analyze EB-YOLO OBB detectors for EB-PBF defect inspection.

---

## 1. Project Scope

EB-YOLO focuses on:

- **Oriented bounding box (OBB) detection** for industrial defects.
- **Small-data regime** robustness (limited annotated samples).
- **Architecture-level ablations** over YOLO11 OBB variants.
- **Interpretability-oriented analysis** for model behavior inspection.

The codebase is built on top of an Ultralytics-style framework (`EB-YOLO-main`) and adds custom modules and YAML model definitions for controlled experimentation.

---

## 2. Repository Structure

```text
EB-YOLO/
├─ EB-YOLO-main/                  # Main training/inference framework (Ultralytics-style fork)
│  ├─ main/                       # Core package code (models, tasks, engine, utils)
│  ├─ results/runs/train/         # Saved experiment runs (args.yaml, results.csv, etc.)
│  ├─ info/                       # Packaging metadata and license
│  └─ Interpretability YOLO/      # Attention/ERF/TP-FP-FN visualization scripts
├─ yaml/                          # EB-YOLO model YAMLs (core contribution entry)
│  ├─ 11_based/
│  ├─ 11_obb_converted/
│  └─ 11_obb_ablation/
├─ dataset/
│  ├─ data.yaml                   # Dataset config (train/val/test + class names)
│  ├─ train/ val/ test/           # Dataset splits
│  └─ valhard/                    # Hard-case subset tools
├─ pre-processing/
│  └─ pre-yolo.py                 # Image enhancement pipeline for defect detection
└─ README.md
```

---

## 3. YAML-Centric Design (Core Contribution)

The `yaml/` directory is the central interface for EB-YOLO model definition and ablation.

## 3.1 `yaml/11_based/`

This folder contains baseline and extended YOLO11 family definitions (including OBB variants), e.g.:

- `yolo11-obb.yaml`
- `yolo11m-obb-LSCD-LQE.yaml`
- `yolo11m-obb-bifpn-GLSA-LSCD-LQE.yaml`
- `yolo11-AFPN-P2345.yaml`
- `yolo11-ReCalibrationFPN-P2345.yaml`

These YAMLs are suitable as primary training configs when the framework resolves standard module names.

## 3.2 `yaml/11_obb_converted/`

This folder stores OBB-oriented converted YAMLs for practical training compatibility and controlled experiments under the same OBB pipeline.

Typical examples:

- `yolo11-obb.yaml`
- `yolo11-bifpn-GLSA.yaml`
- `yolo11m-obb-LSCD-LQE.yaml`
- `yolo11m-obb-bifpn-GLSA-LSCD-LQE.yaml`

## 3.3 `yaml/11_obb_ablation/`

This folder contains structured ablation configurations (A/B/C groups), such as:

- `...-a0-vanilla.yaml`, `...-a1-lscd.yaml`, `...-a2-lqe.yaml`, `...-a3-lscd-lqe.yaml`
- `...-b0-bifpn-conv.yaml`, `...-b1-bifpn-conv-lscd.yaml`, ...
- `...-c0-bifpn-glsa.yaml`, `...-c3-bifpn-glsa-lscd-lqe.yaml`

This layout enables reproducible module-level comparison:

- baseline PAN-FPN OBB head
- neck upgrades (BiFPN / AFPN / ReCalibration variants)
- head upgrades (LSCD, LQE, and combinations)

## 3.4 How to Read a YAML in This Project

A typical YAML contains:

- `nc`: number of classes (set to 5 in this project setup)
- `scales`: compound depth/width scaling
- `backbone`: feature extraction stages
- `head`: feature fusion + OBB detection head (e.g., `OBB`, `OBB_LSCD_LQE`)
- neck/head custom modules (e.g., `GLSA`, `Fusion`, AFPN-related components)

The module registrations are implemented in:

- `EB-YOLO-main/main/nn/tasks.py`
- `EB-YOLO-main/main/nn/extra_modules/head.py`

---

## 4. Dataset and Hard-Case Evaluation

## 4.1 Dataset Configuration

`dataset/data.yaml` defines:

- train image path
- val image path
- test path
- class names (`normal`, `hole`, `wave`, `ballL`, `ballS`)

This file is the default data entry for training and validation.

However, it should be noted that due to the confidentiality restrictions imposed by some companies and schools on certain parts, some images cannot be displayed. Therefore, only a portion of the dataset can be made public. We apologize for any inconvenience this may cause.

## 4.2 Hard-Case Subset Utility

`dataset/valhard/hardcasedataset.py` provides a utility to build a **top-k hard-case subset** from a mined image list.

Main behavior:

- reads `hardcase_img_paths_top.txt`
- selects top-k samples (default `top_k=80`)
- copies matched image-label pairs into a compact subset (`images/`, `labels/`)
- writes `selected_images.txt` for reproducibility

This is useful for stress-testing detector robustness on difficult validation samples.

---

## 5. Preprocessing Pipeline

`pre-processing/pre-yolo.py` implements an enhanced preprocessing pipeline tailored to defect OBB detection, including:

- adaptive CLAHE
- bilateral denoising
- adaptive contrast enhancement
- multi-scale Retinex + 2D gamma correction
- optional edge enhancement / sharpening

It supports batch processing and is designed to improve weak-contrast defect visibility before training/inference.

---

## 6. Interpretability and Model Analysis

The `EB-YOLO-main/Interpretability YOLO/` folder contains scripts for post-hoc analysis:

- `get_model_erf.py`: effective receptive field analysis
- `visualize_attn.py`: attention/module activation visualization
- `visualize_tpfpfn.py`: TP/FP/FN-centric visual diagnostics
- `heatmap.py`, `heatmap_layer.py`: heatmap generation and layer-wise inspection

These tools support qualitative interpretation of why specific defects are detected or missed under scarce-data settings.

---

## 7. Installation

Use Python >= 3.8 with PyTorch installed for your CUDA version.

From the framework root:

```bash
cd EB-YOLO-main
pip install -e .
```

Optional:

```bash
pip install -e ".[dev]"
```

---

## 8. Training and Validation (OBB)

From `EB-YOLO-main/`, run:

```bash
yolo task=obb mode=train model=/path/to/yaml/11_obb_converted/yolo11m-obb-bifpn-GLSA-LSCD-LQE.yaml data=/path/to/dataset/data.yaml epochs=500 imgsz=1280 batch=8 project=runs/train name=exp_eb_yolo
```

Validation:

```bash
yolo task=obb mode=val model=/path/to/runs/train/exp_eb_yolo/weights/best.pt data=/path/to/dataset/data.yaml
```

For hard-case validation, prepare a dedicated data YAML pointing `val` to the hard subset directory.

---

## 9. Reproducibility Notes

- Experiment arguments(Part of, showed as examples) are logged under `EB-YOLO-main/results/runs/train/*/args.yaml`.
- Metrics curves/tables are saved as `results.csv` in each run folder.
- Keep YAML filename, random seed, and data split fixed across ablations.
- When reporting in papers, distinguish:
  - full validation performance
  - hard-case subset performance

---

## 10. Suggested Citation

If you use this repository, please cite your corresponding paper:


---

## 11. License

The framework part follows the license in `EB-YOLO-main/info/LICENSE` (Ultralytics-based AGPL-3.0 lineage).  
Please also check licenses in third-party submodules under `main/nn/extra_modules/`.

Other rotation box object detection algorithms can be referred to in the mmrotate library and the a4ir library.
