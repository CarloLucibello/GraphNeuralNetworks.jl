using Literate

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

Literate.markdown("src_tutorials/introductory_tutorials/temporal_graph_classification.jl", 
                  "src/tutorials/"; execute = true)
