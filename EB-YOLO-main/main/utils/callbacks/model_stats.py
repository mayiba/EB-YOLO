# Ultralytics YOLO — optional callback: save Params / GFLOPs to run directory
"""训练开始时将参数量、FLOPs 等写入 save_dir（JSON + YAML）。需要 pip install thop 才能得到非零 GFLOPs。"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, get_flops, get_num_gradients, get_num_params


def on_train_start_save_model_stats(trainer) -> None:
    """在 on_train_start 调用：仅 rank 0 写入文件。异常仅打日志，不中断训练与其它回调。"""
    if RANK not in {-1, 0}:
        return

    try:
        m = de_parallel(trainer.model)
        imgsz = trainer.args.imgsz
        flops_g = get_flops(m, imgsz)
        n_p = get_num_params(m)
        n_g = get_num_gradients(m)
        n_l = len(list(m.modules()))

        stats = {
            "layers": n_l,
            "parameters": int(n_p),
            "parameters_M": round(n_p / 1e6, 4),
            "gradients": int(n_g),
            "GFLOPs": round(float(flops_g), 4) if flops_g else None,
            "imgsz": imgsz,
            "model_yaml": str(getattr(m, "yaml_file", "") or ""),
            "note": "GFLOPs 为 thop 估算值，与论文口径一致时需注明 imgsz；未安装 thop 时为 null",
        }

        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_dir / "model_stats.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        yaml_path = save_dir / "model_stats.yaml"
        try:
            from ultralytics.utils import yaml_save

            yaml_save(file=yaml_path, data=stats)
        except Exception:
            pass

        fs = f"{stats['GFLOPs']} GFLOPs" if stats["GFLOPs"] is not None else "GFLOPs=n/a (pip install thop)"
        LOGGER.info(f"{json_path.name}: {stats['parameters_M']}M params, {fs}")
    except Exception as e:
        LOGGER.warning(f"model_stats callback skipped (training continues): {e}")


callbacks = {
    "on_train_start": on_train_start_save_model_stats,
}
