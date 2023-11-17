
using DataFrames
using DelimitedFiles: readdlm, writedlm
using Statistics
using CairoMakie

channels = [1, 2]
wavenumbers = [2.0, 4.0, 8.0, 16.0]
function heaviside(x)
    if x >0
        return 1.0
    else
        return 0.0
    end
end

function cdf(x,samples)
    return mean( heaviside.(x.-samples))
end
output_ch1 = zeros((4,4))
output_ch2 = zeros((4,4))
for ch in channels
    @show ch
    fig = Figure(resolution=(1600, 400), fontsize=24)
    if ch == 1
        min_x, max_x = -25, 5
    else
        min_x, max_x = -20, 20
    end
    X = LinRange(min_x, max_x, 1000)
    for i in 1:4

        wn = wavenumbers[i]
        train_hr = readdlm("/groups/esm/kdeck/downscaling/stats/data/train/train_pixels_ch$(ch)_$(wn).csv", ',', Float32, '\n')[:]
        train_lr = readdlm("/groups/esm/kdeck/downscaling/stats/data/train/train_pixels_ch$(ch)_0.0.csv", ',', Float32, '\n')[:]
        gen_hr = readdlm("/groups/esm/kdeck/downscaling/stats/data/gen/downscale_gen_pixels_ch$(ch)_$(wn).csv", ',', Float32, '\n')[:]
        cdf_train_hr = cdf.(X, Ref(train_hr))
        cdf_gen_hr = cdf.(X, Ref(gen_hr))
        cdf_train_lr = cdf.(X, Ref(train_lr))
        ks_gen_hr_train_hr = maximum(abs.(cdf_train_hr .- cdf_gen_hr))
        ks_train_lr_train_hr = maximum(abs.(cdf_train_hr .- cdf_train_lr))
        if i == 1
            if ch ==1
                ax = Axis(fig[1,1], xlabel="Supersaturation", ylabel="CDF", title=L"k_x = k_y = 2")
            else
                ax = Axis(fig[1,1], xlabel="Vorticity", ylabel="CDF", title=L"k_x = k_y = 2")
            end
        elseif i == 2
            if ch ==1
                ax = Axis(fig[1,2], xlabel="Supersaturation", ylabel="CDF", title=L"k_x = k_y = 4")
            else
                ax = Axis(fig[1,2], xlabel="Vorticity", ylabel="CDF", title=L"k_x = k_y = 4")
            end
        elseif i ==3
            if ch ==1
                ax = Axis(fig[1,3], xlabel="Supersaturation", ylabel="CDF", title=L"k_x = k_y = 8")
            else
                ax = Axis(fig[1,3], xlabel="Vorticity", ylabel="CDF", title=L"k_x = k_y = 8")
            end
        elseif i==4
            if ch ==1
                ax = Axis(fig[1,4], xlabel="Supersaturation", ylabel="CDF", title=L"k_x = k_y = 16")
            else
                ax = Axis(fig[1,4], xlabel="Vorticity", ylabel="CDF", title=L"k_x = k_y = 16")
            end
        end

        lines!(X, cdf_train_hr, color=(:orange, 1.0), strokewidth = 1.5,label="real high res.")
        lines!(X, cdf_train_lr, color=(:green, 1.0), strokewidth = 1.5, label="generated high res.")
        lines!(X, cdf_gen_hr, color=(:purple, 1.0), strokewidth = 1.5, label="real low res.")
        xlims!(ax, min_x, max_x)
        ylims!(ax, 0, 1)
        @show wn
        @show ks_gen_hr_train_hr
        @show ks_train_lr_train_hr
        if ch==1
            output_ch1[i,:] .= [ch, wn, ks_gen_hr_train_hr, ks_train_lr_train_hr]
        elseif ch==2
            output_ch2[i,:] .= [ch, wn, ks_gen_hr_train_hr, ks_train_lr_train_hr]
        end

    end

    axislegend(; position= :lt, labelsize= 16)
    save("fig:cdf_ch$(ch).png", fig, px_per_unit = 2)
end

columns = ["Channel" "Wavenumber"  "KS-DGHR-THR" "KS-TLR-THR"]
output_data = vcat(columns,output_ch1, output_ch2)
open("ks_stats.txt", "w") do io
    writedlm(io, output_data,',')
end