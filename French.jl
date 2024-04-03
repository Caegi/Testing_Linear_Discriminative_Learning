using JudiLing # our package
using CSV # read csv files
using DataFrames # parse data into dataframes
using Statistics # mean function
using Word2Vec # load embeddings


mkpath(joinpath(@__DIR__, "data"))
# load french file
nouns = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "nlexique.csv")))
adjs = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "alexique.csv")))
verbs = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "vlexique.csv")))
ortho_adjs = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "alexique_ortho_final.csv")))
model = wordvectors(joinpath(@__DIR__, "assets","model_redo", "model.w2v"))
#display(vocabulary(model)[1:1000])
#display(get_vector(model, "manger_v"))
# french = vcat(nouns, adjs, verbs, cols = :union)
french = adjs
ortho_dataset = ortho_adjs
column = "lexeme"

"""# Open the file in read mode
file = open(joinpath(@__DIR__, "assets", "lemma-A-pos.txt"), "r")

# Read all lines from the file into an array
display("reading file")
lines = readlines(file)

# Close the file
close(file)

display("loading embeddings")
embedding_table = Dict(split(split(line)[1], "_")[1] => [parse(Float64, val) for val in split(line)[2:end]] for line in lines[2:1000])
lines = nothing
display("creating filter")
filters = [[in(word, keys(embedding_table)) for word in french[!, column]] for column in column_names]

"""
POS = "_adj"
column_names = vcat(["lexeme"], names(ortho_dataset)[3:end])
display(typeof(vocabulary(model)))
voca = Set(vocabulary(model))
display(column_names)
display(length(vocabulary(model)))
filters = [[in(word*POS, voca) for word in ortho_dataset[!, column_name]] for column_name in column_names]

display("filtering words without embeddings")
french_form = [word for (filter, column_name) in zip(filters, column_names) for word in french[!, column_name][filter] ]
french_embed = [get_vector(model, word*POS) for (filter, column_name) in zip(filters, column_names) for word in ortho_dataset[!, column_name][filter]]
#french_ortho = [word for (filter, column_name) in zip(filters, column_names) for word in ortho_dataset[!, column_name][filter]]

french_form = DataFrame(lexeme = french_form)
# french_embed = DataFrame(lexeme = french_embed)
#display(names(french_embed))
#display(french_embed[!, "lexeme"][1:10])
display(french_form[!, "lexeme"][1:10])
display("number of words after filtering:")
display([sum(filter) for filter in filters])


# Split the dataset into training and validation sets
function get_trigrams(word::AbstractString)
    trigrams = Set()
    word = "#" * word * "#" 
    
    # Split the word into a list of characters
    word_characters = collect(word)

    for i in 1:length(word_characters) - 2
            trigram = join(word_characters[i:i+2])
            push!(trigrams, trigram)
    end

    return trigrams
end

# Initialize empty lists for training and validation sets
train_set = Set()
val_set = Set()
all_trigrams = Set()

CORPUS_MAX_SIZE = 500000
# Iterate over words in the dataset
train_filter = []
val_filter = []
for column in names(french_form)
    display(column)
for word in french_form[!, column]
    if word ∈ (train_set ∪ val_set) || length(train_set) >= CORPUS_MAX_SIZE
        push!(train_filter, false)
        push!(val_filter, false)
        continue
    end
    word_trigrams = get_trigrams(word)
    if any(trigram ∉ all_trigrams for trigram in word_trigrams)  || length(val_set) >= round(nrow(french) * 0.2) ||
        push!(train_set, word)
        union!(all_trigrams, word_trigrams)
        push!(train_filter, true)
        push!(val_filter, false)
      #  push!(all_trigrams, word_trigrams)
    else
        push!(val_set, word)
        push!(train_filter, false)
        push!(val_filter, true)
    end
end
# train_filter = [i for i in 1:length(train_filter) if train_filter[i]]
# val_filter = [i for i in 1:length(val_filter) if val_filter[i]]

train_size = length(train_set)
val_size = length(val_set)
trigram_size = length(all_trigrams)
train_val_overlap = length(train_set ∩ val_set)
display("number of words in train: $train_size")
display("number of words in val: $val_size")
display("number of trigrams: $trigram_size")
display("number of words both in val and train (should be 0): $train_val_overlap")

# train_filter = [in(word, train_set) for word in french[!, column]]
# val_filter = [in(word, val_set) for word in french[!, column]]



display(typeof(train_filter))
train_filter = convert(Array{Bool}, train_filter)
val_filter = convert(Array{Bool}, val_filter)
display(typeof(train_filter))
display(length(train_filter))
display(length(val_filter))
display(nrow(french_form))
french_train = french_form[train_filter, :]
french_val = french_form[val_filter, :]

# create C matrices for both training and validation datasets
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
    french_train,
    french_val,
    grams = 3,
    target_col = :lexeme,
    tokenized = false,
    keep_sep = false,
)

# create S matrices
n_features = size(cue_obj_train.C, 2)

display("creating Semantic matrices")
S_train = french_embed[train_filter]
S_val = french_embed[val_filter]


display("transforming vector of vectors into matrix")
S_train = reduce(vcat, transpose.(S_train))
S_val = reduce(vcat, transpose.(S_val))



# here we learning mapping only from training dataset
display("learning transform matrices")
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
display("meaning to form done")
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)
display("form to meaning done")

# we predict S and C for both training and validation datasets
Chat_train = S_train * G_train
display("C_hat train done")
Chat_val = S_val * G_train
display("C_hat val done")
Shat_train = cue_obj_train.C * F_train
display("S_hat train done")
Shat_val = cue_obj_val.C * F_train
display("S_hat val done")

# we evaluate them


"""num_rows = size(Chat_train, 1)
split_point = num_rows ÷ 2

small_matrix = Chat_train[1:min(num_rows,15000) , :]
small_matrix_2 = cue_obj_train.C[1:min(num_rows,15000) , :]
Shat_train_small = Shat_train[1:min(num_rows,15000) , :]
S_train_small = S_train[1:min(num_rows,15000) , :]
display("evaluation 1")
@show JudiLing.eval_SC(small_matrix, small_matrix_2)
display("evaluation 2")
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
display("evaluation 3")
@show JudiLing.eval_SC(Shat_train_small, S_train_small)
display("evaluation 4")
@show JudiLing.eval_SC(Shat_val, S_val)
display("evaluation done")
"""
num_rows = size(Chat_train, 1)
batch_size = 15000

# Arrays to store evaluation results for averaging later
eval1_results = Float64[]
eval3_results = Float64[]

for start_idx in 1:batch_size:num_rows
    end_idx = min(start_idx + batch_size - 1, num_rows)
    
    # Extracting the batches
    small_matrix = Chat_train[start_idx:end_idx, :]
    small_matrix_2 = cue_obj_train.C[start_idx:end_idx, :]
    Shat_train_small = Shat_train[start_idx:end_idx, :]
    S_train_small = S_train[start_idx:end_idx, :]
    
    # Assuming eval_SC returns a scalar value that can be appended to an array
    push!(eval1_results, JudiLing.eval_SC(small_matrix, small_matrix_2))
    push!(eval3_results, JudiLing.eval_SC(Shat_train_small, S_train_small))
end

# Evaluation 2 and 4 are assumed to be computed on the full validation sets
eval2 = JudiLing.eval_SC(Chat_val, cue_obj_val.C)
eval4 = JudiLing.eval_SC(Shat_val, S_val)

# Averaging the batch results
avg_eval1 = mean(eval1_results)
avg_eval3 = mean(eval3_results)

# Printing the final averaged results
println("train form prediction: $avg_eval1")
println("Eval Form prediction : $eval2") # This is computed on the full set, not averaged over batches
println("train semantic prediction : $avg_eval3")
println("Eval Semantic prediction: $eval4") # Also computed on the full set


# we can use build path and learn path
A = cue_obj_train.A
max_t = JudiLing.cal_max_timestep(french_train, french_val, :lexeme)
display("there will be $max_t timesteps")

res_learn_train, gpi_learn_train = JudiLing.learn_paths(
    french_train,
    french_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    gold_ind = cue_obj_train.gold_ind,
    Shat_val = Shat_train,
    check_gold_path = true,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    sep_token = "_",
    keep_sep = false,
    target_col = :lexeme,
    issparse = :dense,
    verbose = true,
)

res_learn_val, gpi_learn_val = JudiLing.learn_paths(
    french_train,
    french_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    gold_ind = cue_obj_val.gold_ind,
    Shat_val = Shat_val,
    check_gold_path = true,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    is_tolerant = true,
    tolerance = -0.1,
    max_tolerance = 2,
    tokenized = false,
    sep_token = "-",
    keep_sep = false,
    target_col = :lexeme,
    issparse = :dense,
    verbose = true,
)

# you can save results into csv files or dfs
JudiLing.write2csv(
    res_learn_train,
    french,
    cue_obj_train,
    cue_obj_train,
    "french_learn_train_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :lexeme,
    root_dir = @__DIR__,
    output_dir = "french_out",
)

JudiLing.write2csv(
    res_learn_val,
    french,
    cue_obj_val,
    cue_obj_val,
    "french_learn_val_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :lexeme,
    root_dir = @__DIR__,
    output_dir = "french_out",
)


acc_learn_train =
    JudiLing.eval_acc(res_learn_train, cue_obj_train.gold_ind, verbose = false)
acc_learn_val = JudiLing.eval_acc(res_learn_val, cue_obj_val.gold_ind, verbose = false)

# with build: 
res_build_train = JudiLing.build_paths(
    french_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t = max_t,
    n_neighbors = 3,
    verbose = true,
)

res_build_val = JudiLing.build_paths(
    french_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t = max_t,
    n_neighbors = 20,
    verbose = true,
)


JudiLing.write2csv(
    res_build_train,
    french,
    cue_obj_train,
    cue_obj_train,
    "french_build_train_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :lexeme,
    root_dir = @__DIR__,
    output_dir = "french_out",
)

JudiLing.write2csv(
    res_build_val,
    french,
    cue_obj_val,
    cue_obj_val,
    "french_build_val_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :lexeme,
    root_dir = @__DIR__,
    output_dir = "french_out",
)

acc_build_train =
    JudiLing.eval_acc(res_build_train, cue_obj_train.gold_ind, verbose = false)
acc_build_val = JudiLing.eval_acc(res_build_val, cue_obj_val.gold_ind, verbose = false)

@show acc_learn_train
@show acc_learn_val
@show acc_build_train
@show acc_build_val

# Once you are done, you may want to clean up the workspace
#rm(joinpath(@__DIR__, "data"), force = true, recursive = true)
#rm(joinpath(@__DIR__, "french_out"), force = true, recursive = true)


