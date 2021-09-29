using Flux, Random


function get_grad1(m, data)
    gradient(Flux.params(m)) do
        loss(m, data)
    end
end

function get_grad2(m, data)
    gradient(Flux.params(m)) do
        ps = Flux.params(m) # just creating params without using them
        loss(m, data)
    end
end

function get_grad3(m, data)
    gradient(Flux.params(m)) do
        ps = Flux.params(m)
        loss(m, data) + sum(sum(p) for p in ps)
    end
end

function get_grad4(m, data)
    ps = Flux.params(m)
    gradient(Flux.params(m)) do
        loss(m, data) + sum(sum(p) for p in ps)
    end
end

function get_grad5(m, data)
    gradient(Flux.params(m)) do
        sum(Flux.params(m)[1]) + sum(Flux.params(m)[2])
    end
end

function get_grad6(m, data)
    ps = Flux.params(m)
    gradient(Flux.params(m)) do
        sum(ps[1]) + sum(ps[2])
    end
end

function get_grad7(m, data)
    ps = Flux.params(m)
    gradient(Flux.params(m)) do
        sum(m.weight) + sum(m.bias)
    end
end

Random.seed!(17)
m = Dense(3, 2);
data = rand(Float32, 3, 5)
loss(m, x) = sum(m(x).^2)

g1 = get_grad1(m, data) 
g2 = get_grad2(m, data)
g3 = get_grad3(m, data) 
g4 = get_grad4(m, data) 
g5 = get_grad5(m, data) 
g6 = get_grad6(m, data) 
g7 = get_grad7(m, data) 

@show g1[m.weight] #            correct
@show g2[m.weight] # == g1 .+ 1  wrong, should be == g1
@show g3[m.weight] # == g1 .+ 2  wrong, should be == g1 .+ 1
@show g4[m.weight] # == g1 .+ 1  correct
@show g5[m.weight] # .== 2       wrong, should be .== 1
@show g6[m.weight] # == nothing  wrong,  should be .== 1
@show g7[m.weight] # == 1        correct
