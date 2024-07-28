@testsetup module SharedTestSetup

import Reexport: @reexport

@reexport using GNNLux
@reexport using Lux, Functors
@reexport using ComponentArrays, LuxCore, LuxTestUtils, Random, StableRNGs, Test,
                Zygote, Statistics
@reexport using LuxTestUtils: @jet, @test_gradients, check_approx
using MLDataDevices

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "cpu" ? CPUDevice() :
          mode == "cuda" ? CUDADevice() : mode == "amdgpu" ? AMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

export get_default_rng

# export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing

end