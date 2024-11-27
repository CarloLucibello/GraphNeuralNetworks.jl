using TestItemRunner

## See https://www.julia-vscode.org/docs/stable/userguide/testitems/
## for how to run the tests within VS Code.
## See test_module.jl for the test infrastructure.

## Uncomment below to change the default test settings
# ENV["GNN_TEST_CPU"] = "false"
# ENV["GNN_TEST_CUDA"] = "true"
# ENV["GNN_TEST_AMDGPU"] = "true"
# ENV["GNN_TEST_Metal"] = "true"

if get(ENV, "GNN_TEST_CPU", "true") == "true"
    @run_package_tests filter = ti -> :gpu ∉ ti.tags
end
if get(ENV, "GNN_TEST_CUDA", "false") == "true"
    @run_package_tests filter = ti -> :gpu ∈ ti.tags
end
if get(ENV, "GNN_TEST_AMDGPU", "false") == "true"
    @run_package_tests filter = ti -> :gpu ∈ ti.tags
end
if get(ENV, "GNN_TEST_Metal", "false") == "true"
    @run_package_tests filter = ti -> :gpu ∈ ti.tags
end

