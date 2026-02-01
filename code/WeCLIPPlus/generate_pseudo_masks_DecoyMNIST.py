import argparse
import os
import re


def _default_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def _resolve_paths(repo_root):
    repo_root = os.path.abspath(repo_root)
    weclip_root = os.path.join(repo_root, "code", "WeCLIPPlus")
    voc_root = os.path.join(weclip_root, "VOCdevkit", "VOC2012")
    return {
        "weclip_root": weclip_root,
        "config": os.path.join(weclip_root, "configs", "voc_attn_reg.yaml"),
        "config_dir": os.path.join(weclip_root, "configs"),
        "voc_root": voc_root,
        "set_dir": os.path.join(voc_root, "ImageSets", "Main"),
        "dest_dir": os.path.join(voc_root, "JPEGImages"),
        "clip_pretrain_path": os.path.join(weclip_root, "pretrained", "ViT-B-16.pt"),
    }


def _write_runtime_config(
    base_config,
    output_dir,
    voc_root,
    clip_pretrain_path,
    dino_model=None,
    dino_fts_dim=None,
    dino_decoder_layers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    name_list_dir = os.path.join(voc_root, "ImageSets", "Main")

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_config)
        cfg.dataset.root_dir = voc_root
        cfg.dataset.name_list_dir = name_list_dir
        cfg.clip_init.clip_pretrain_path = clip_pretrain_path
        if dino_model:
            cfg.dino_init.dino_model = dino_model
        if dino_fts_dim is not None:
            cfg.dino_init.dino_fts_fuse_dim = int(dino_fts_dim)
        if dino_decoder_layers is not None:
            cfg.dino_init.decoder_layer = int(dino_decoder_layers)

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        OmegaConf.save(cfg, output_path)
        return output_path
    except Exception:
        # Fallback to text replacement if OmegaConf is unavailable.
        with open(base_config, "r") as f:
            content = f.read()

        content = re.sub(
            r"(root_dir:\s*')([^']*)(')",
            rf"\1{voc_root}\3",
            content,
        )
        content = re.sub(
            r"(name_list_dir:\s*')([^']*)(')",
            rf"\1{name_list_dir}\3",
            content,
        )
        content = re.sub(
            r"(clip_pretrain_path:\s*')([^']*)(')",
            rf"\1{clip_pretrain_path}\3",
            content,
        )

        if dino_model:
            content = re.sub(
                r"(dino_model:\s*')([^']*)(')",
                rf"\1{dino_model}\3",
                content,
            )
        if dino_fts_dim is not None:
            content = re.sub(
                r"(dino_fts_fuse_dim:\s*)([0-9]+)",
                rf"\1{int(dino_fts_dim)}",
                content,
            )
        if dino_decoder_layers is not None:
            content = re.sub(
                r"(decoder_layer:\s*)([0-9]+)",
                rf"\1{int(dino_decoder_layers)}",
                content,
            )

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        with open(output_path, "w") as f:
            f.write(content)
        return output_path


def main(
    repo_root,
    class_name,
    results_dir,
    dino_model,
    dino_fts_dim,
    dino_decoder_layers,
):
    if class_name:
        os.environ["CLIP_TEXT_VERSION"] = class_name

    from scripts import dist_clip_voc
    import test_msc_flip_voc

    paths = _resolve_paths(repo_root)

    # Verify that prepare_colored_mnist.py has already been run.
    train_txt = os.path.join(paths["set_dir"], "train.txt")
    if not os.path.isdir(paths["dest_dir"]) or not os.listdir(paths["dest_dir"]):
        raise FileNotFoundError(
            f"JPEGImages directory is missing or empty: {paths['dest_dir']}\n"
            "Run prepare_colored_mnist.py first."
        )
    if not os.path.isfile(train_txt):
        raise FileNotFoundError(
            f"ImageSets/Main/train.txt not found: {train_txt}\n"
            "Run prepare_colored_mnist.py first."
        )

    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        paths["clip_pretrain_path"],
        dino_model=dino_model,
        dino_fts_dim=dino_fts_dim,
        dino_decoder_layers=dino_decoder_layers,
    )

    final_path = dist_clip_voc.main(config)

    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir

    test_msc_flip_voc.outer_main(final_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pseudo masks for DecoyMNIST. "
        "Run prepare_colored_mnist.py first to populate JPEGImages/ and ImageSets/."
    )
    parser.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--class-name",
        default="digit",
        help="Foreground class name for DecoyMNIST (default: digit).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for prediction_cmap (default: results).",
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override DINO model name in config (e.g., xcit_medium_24_p16).",
    )
    parser.add_argument(
        "--dino-fts-dim",
        type=int,
        default=None,
        help="Override dino_fts_fuse_dim in config (e.g., 512 for XCiT-Medium).",
    )
    parser.add_argument(
        "--dino-decoder-layers",
        type=int,
        default=None,
        help="Override decoder_layer in config.",
    )
    args = parser.parse_args()

    main(
        args.repo_root,
        args.class_name,
        args.results_dir,
        args.dino_model,
        args.dino_fts_dim,
        args.dino_decoder_layers,
    )
