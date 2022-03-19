using Flux
using GraphNeuralNetworks
using BenchmarkTools
using ProfileView

f(x) = 1

function test(g)
    loader = Flux.DataLoader(g, batchsize=100, shuffle=true)
    s = 0 
    for d in loader
        s += f(d)
    end
    return s
end

n = 5000
s = 10
data = [rand_graph(s, s, ndata = rand(1, s)) for i in 1:n] 
x1 = Flux.batch(data) 
x2 = Flux.batch([rand(s + s + s + s) for i in 1:n]) #source+target+data+extra

# @profview test(x1)
@btime test($x1);   #  1.295 s (2502 allocations: 6.17 MiB)
@btime test($x2);   #  357.595 μs (152 allocations: 1.61 MiB)
@btime test($data); #  65.288 ms (227002 allocations: 27.00 MiB) # this PR
