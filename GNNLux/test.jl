using Lux 
using Lux: AbstractExplicitContainerLayer, StatefulLuxLayer
using Random

struct A <: AbstractExplicitContainerLayer{(:x,)}
    x
    y 
end


a = A(Dense(3, 5), true)
rng = Random.default_rng()
ps = Lux.initialparameters(rng, a)
ps.x #ERROR, no field named x
ps.weight # OK

struct B <: AbstractExplicitContainerLayer{(:x,:y)}
    x
    y 
end

b = B(Dense(3, 5), Dense(5, 5))
rng = Random.default_rng()
ps = Lux.initialparameters(rng, b)
ps.x #OK
ps.y #OK

rng = Random.default_rng()
x = rand(rng, Float32, 2, 3)
model = Chain(Dense(2 => 5, relu), Dense(5 => 5))
ps = Lux.initialparameters(rng, model)
st = Lux.initialstates(rng, model)
y, _ = model(x, ps, st)

model2 = StatefulLuxLayer{true}(model, ps, st)
y2 = model2(x)


a = A(model, true)
ps = Lux.initialparameters(rng, a)
st = Lux.initialstates(rng, a)
model3 = StatefulLuxLayer(a.x, ps, st)
y3 = model3(x)
