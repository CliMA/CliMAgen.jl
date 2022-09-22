using Wandb

import CliMAgen: 
    log_config, 
    log_dict, 
    log_image, 
    log_artifact

logger = Wandb.WandbLogger(
    project=args.project,
    name="mnist_32x32-$(Dates.now())",
    config=Dict(),
)

function log_config(logger::Wandb.WandbLogger, config::Dict)
    Wandb.update_config!(logger, config)
end

function log_dict(logger::Wandb.WandbLogger, dict::Dict)
    Wandb.log(logger, dict)
end

function log_checkpoint(logger::Wandb.WandbLogger, path::String)
    artifact = Wandb.WandbArtifact("checkpoint", type="BSON-file")
    Wandb.add_file(artifact, path)
    Wandb.log(logger, artifact)
end
