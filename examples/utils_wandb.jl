using Wandb

import CliMAgen

function CliMAgen.log_config(logger::Wandb.WandbLogger, config::Dict)
    Wandb.update_config!(logger, config)
end

function CliMAgen.log_dict(logger::Wandb.WandbLogger, dict::Dict)
    Wandb.log(logger, dict)
end

function CliMAgen.log_checkpoint(logger::Wandb.WandbLogger, path::String; name="checkpoint", type="BSON-file")
    artifact = Wandb.WandbArtifact(name, type=type)
    Wandb.add_file(artifact, path)
    Wandb.log(logger, artifact)
end
