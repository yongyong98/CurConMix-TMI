"""Historical spatial training entry point; use an explicit stage configuration."""

import hydra


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    expected_split = "LLS48"
    if cfg.split_selector != expected_split:
        raise ValueError(
            f"This release entry point requires split_selector={expected_split}."
        )
    if cfg.method not in ("supcon", "curriculum_supcon"):
        raise ValueError("Only the released CurConMix training paths are supported.")
    from curconmix_release.spatial_protocol import cfg_split

    cfg_split(cfg, "lls48")
    from train import train_cross_val, train_cross_val_SSL

    if cfg.ssl:
        train_cross_val_SSL(cfg)
    else:
        train_cross_val(cfg)


if __name__ == "__main__":
    main()
