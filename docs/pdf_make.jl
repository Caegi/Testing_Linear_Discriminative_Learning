using JudiLing
using Documenter
using DocumenterLaTeX

makedocs(;
    modules=[JudiLing],
    authors="Xuefeng Luo",
    repo="https://github.com/MegamindHenry/JudiLing.jl/blob/{commit}{path}#L{line}",
    sitename="JudiLing.jl",
    format=LaTeX(platform = "none"),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "Make Cue Matrix" => "man/make_cue_matrix.md",
            "Make Semantic Matrix" => "man/make_semantic_matrix.md",
            "Cholesky" => "man/cholesky.md",
            "Make Adjacency Matrix" => "man/make_adjacency_matrix.md",
            "Make Yt Matrix" => "man/make_yt_matrix.md",
            "Find Paths" => "man/find_path.md",
            "Evaluation" => "man/eval.md",
            "Output" => "man/output.md",
            "Test Combo" => "man/test_combo.md",
            "Utils" => "man/utils.md"
        ],
        "All Manual index" => "man/all_manual.md"
    ],
)