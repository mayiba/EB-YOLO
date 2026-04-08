from pathlib import Path
import shutil

def collect_hardcase_subset(
    hardcase_txt: str,
    src_images_dir: str,
    src_labels_dir: str,
    dst_root_dir: str,
    top_k: int = 80,
):
    """
    ´Ó hardcase_img_paths_top.txt ÖÐÈ¡Ç° top_k ÕÅÍ¼£¬
    ÔÚ±¾µØÑéÖ¤¼¯Ä¿Â¼ÖÐ°´ÎÄ¼þÃûÆ¥ÅäÍ¼Æ¬ºÍÍ¬Ãû±êÇ©(txt)£¬
    ²¢¸´ÖÆµ½ÐÂµÄ×Ó¼¯Ä¿Â¼¡£

    Ä¿Â¼½á¹¹Êä³öÎª:
      dst_root_dir/
        images/
        labels/
    """
    hardcase_txt = Path(hardcase_txt)
    src_images_dir = Path(src_images_dir)
    src_labels_dir = Path(src_labels_dir)
    dst_root_dir = Path(dst_root_dir)

    dst_images_dir = dst_root_dir / "images"
    dst_labels_dir = dst_root_dir / "labels"
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    if not hardcase_txt.exists():
        raise FileNotFoundError(f"hardcase txt not found: {hardcase_txt}")
    if not src_images_dir.exists():
        raise FileNotFoundError(f"src images dir not found: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"src labels dir not found: {src_labels_dir}")

    lines = [x.strip() for x in hardcase_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
    selected = lines[:top_k]

    copied = 0
    missing_images = []
    missing_labels = []
    copied_names = []

    for p in selected:
        # hardcase ÎÄ¼þÀï¿ÉÄÜÊÇ linux ¾ø¶ÔÂ·¾¶£¬ÕâÀïÖ»È¡ÎÄ¼þÃûÆ¥Åä±¾µØÊý¾Ý
        img_name = Path(p).name                # e.g. 726.png
        stem = Path(img_name).stem             # e.g. 726

        src_img = src_images_dir / img_name
        src_lbl = src_labels_dir / f"{stem}.txt"

        if not src_img.exists():
            missing_images.append(str(src_img))
            continue
        if not src_lbl.exists():
            missing_labels.append(str(src_lbl))
            continue

        shutil.copy2(src_img, dst_images_dir / src_img.name)
        shutil.copy2(src_lbl, dst_labels_dir / src_lbl.name)
        copied += 1
        copied_names.append(src_img.name)

    # ¼ÇÂ¼Çåµ¥£¬·½±ã¸´ÏÖ
    (dst_root_dir / "selected_images.txt").write_text(
        "\n".join(copied_names), encoding="utf-8"
    )

    print(f"[DONE] requested top_k={top_k}")
    print(f"[DONE] copied pairs={copied}")
    print(f"[WARN] missing_images={len(missing_images)}")
    print(f"[WARN] missing_labels={len(missing_labels)}")

    if missing_images:
        print("\n[Missing image examples]")
        for x in missing_images[:10]:
            print(x)

    if missing_labels:
        print("\n[Missing label examples]")
        for x in missing_labels[:10]:
            print(x)


if __name__ == "__main__":
    # Äã°´Êµ¼ÊÂ·¾¶¸ÄÕâÀï
    collect_hardcase_subset(
        hardcase_txt=r"/home/chenhm/PycharmProjects/PythonProject1/ai4rs-main/work_dirs/hardcase_ref/hardcase_img_paths_top.txt",
        src_images_dir=r"/home/chenhm/PycharmProjects/PythonProject1/ultralytics/dataset_demo/val/images",
        src_labels_dir=r"/home/chenhm/PycharmProjects/PythonProject1/ultralytics/dataset_demo/labels/val/",
        dst_root_dir=r"D:\research\code\final_yolov11\final_yolo\ultralytics-yolo11-main\datasets\hardcase80",
        top_k=80,
    )