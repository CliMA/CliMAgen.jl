function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--outdir"
            help = "output directory for checkpoint files and artifacts"
            arg_type = String
            default = joinpath(pkgdir(Downscaling), "output/")
            required = false
        "--datadir"
            help = "directory containing the dataloader utils"
            arg_type = String
            default = joinpath(pkgdir(Downscaling), "data/")
            required = false
        "--restartfile"
            help = "restart from this checkpoint file"
            arg_type = String
            default = ""
            required = false
        "--logging"
            help = "logging to WandB"
            action = :store_true
    end

    return parse_args(s)
end
