from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.models.flux.model import FluxModelParams, MegatronFluxModel


if __name__ == '__main__':
    params = FluxModelParams()
    model = MegatronFluxModel(flux_params=params)

    llm.import_ckpt(model, source = "hf://black-forest-labs/FLUX.1-dev", output_path="/workspace/weights/flux_dist/")
