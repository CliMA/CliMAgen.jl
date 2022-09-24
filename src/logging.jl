"""
    ClimaGen.log_config
"""
function log_config(::AbstractLogger, ::Dict) end

"""
    ClimaGen.log_dict
"""
function log_dict(::AbstractLogger, ::Dict) end

"""
    ClimaGen.log_image
"""
function log_image end

"""
    ClimaGen.log_checkpoint
"""
function log_checkpoint(::AbstractLogger, ::String) end
