import hydra
from omegaconf import DictConfig
from main import run as baseline_run
from evaluation import run as eval_run


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(f"log is saved to {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/hub.log")

    if cfg["evaluation_mode"]:
        print(cfg)
        eval_run(**cfg)
    else:
        if cfg["model_name"] in ["baseline", "hrnet"]:
            baseline_run(**cfg)



if __name__ == "__main__":
    main()