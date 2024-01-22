function image_plot(gen, gen_bias, train, k, plotname)
    fig = Figure(size=(1000, 600), fontsize=24)
    nimages = size(train)[end]
    colorrange = extrema(train)
    for i in 1:5
        ylabel = i == 1 ? "Data" : ""
        ax = Axis(fig[1, i], xlabel="", ylabel = ylabel,xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
        heatmap!(ax, train[:,:,i], colorrange = colorrange)
    end
    for i in 1:5
        ylabel = i == 1 ? "Generated, k=0" : ""
        ax = Axis(fig[2, i], xlabel="", ylabel = ylabel,xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
        heatmap!(ax, gen[:,:,i], colorrange = colorrange)
    end 
    for i in 1:5
        ylabel = i == 1 ?  "Generated, k=$k" : ""
        ax = Axis(fig[3, i], xlabel="", ylabel = ylabel,xticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false, yticksvisible = false)
        heatmap!(ax, gen_bias[:,:,i], colorrange = colorrange)
    end
    save(plotname, fig, px_per_unit = 2)
    
end