from flux_controlnet_training import flux_controlnet_training
import nemo_run as run
from megatron.core.distributed import DistributedDataParallelConfig
from lightning.pytorch.loggers import WandbLogger
from flux_training import flux_training


def flux_cn(tp=4, devices=8):
    recipe = flux_controlnet_training()
    recipe.model.flux_params.t5_params = None
    recipe.model.flux_params.clip_params = None
    recipe.model.flux_params.vae_config = None
    recipe.model.flux_params.device = 'cuda'
    recipe.trainer.strategy.tensor_model_parallel_size = tp
    recipe.trainer.devices = devices
    recipe.data.global_batch_size = 8
    # recipe.trainer.callbacks.append(reciprun.Config(NsysCallback, start_step=10, end_step=11, gen_shape=True))
    recipe.model.flux_controlnet_config.num_single_layers = 38
    recipe.model.flux_controlnet_config.num_joint_layers = 19
    # recipe.trainer.strategy.ddp = run.Config(
    #     DistributedDataParallelConfig,
    #     check_for_nan_in_grad=True,
    #     grad_reduce_in_fp32=True,
    #     use_custom_fsdp=False,
    #     data_parallel_sharding_strategy='optim_grads_params',
    # )
    recipe.log.wandb = run.Config(
        WandbLogger,
        project = "flux-aot",
        name = f"flux_controlnet_tp{tp}_gbs{recipe.data.global_batch_size}_mbs{recipe.data.micro_batch_size}",
    )

    return recipe

if __name__ == '__main__':
    recipe = flux_cn(tp=2)

    run.run(recipe)