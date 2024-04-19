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
ortho_verbs = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "vlexique_ortho.csv")))
ortho_nouns = DataFrame(CSV.File(joinpath(@__DIR__, "assets", "distrib", "nlexique_ortho_final.csv")))
model = wordvectors(joinpath(@__DIR__, "assets","model_redo", "model.w2v"))
voca = Set(vocabulary(model))
VERBOSE = 0 # set verbosity to 1 to have more information about the process
CORPUS_MAX_SIZE = 500000 # Maximum number of words to be considered in the corpus. should be at least 2 times the column_size


# get rid of the gender column
select!(nouns, Not(:gen))
select!(ortho_nouns, Not(:gen))

# drop "variants" column for all datasets:
for df in [nouns, adjs, verbs, ortho_adjs, ortho_verbs, ortho_nouns]  
    select!(df, Not(:variants))
    
end
# choose which dataset to use can be either verbs or adjs or nouns, and their phonological form or their orthographic form
# if you want to use the orthographic form, french and ortho_dataset should be the same (and orthographic): (ex ortho_adj and ortho_adj, ortho_verbs and ortho_verbs)
# if you want to use the phonological form, french and ortho_dataset should be different (ex adj and ortho_adj or verbs and ortho_verbs)
# so at each CHANGE you MUST change 3 variable: french, ortho_dataset and POS so that they are consistent


POS = "_v" # _v for verbs _n for nouns _adj for adjectives



function run(french, ortho_dataset, POS, CORPUS_MAX_SIZE, PHONO)

    """french = verbs
    ortho_dataset = ortho_verbs"""
    
    
    MODE = PHONO ? "phoneme" : "grapheme" # variable used for logging
    
    column_names = names(ortho_dataset)# dont ignore it, idk vcat(["lexeme"], names(ortho_dataset)[2:end]) # ignore the variant column

    
    display("the column of the df are: $(column_names)")
    if VERBOSE == 1
        display("there are $(length(vocabulary(model))) words in the model")
    end
    filters = [[in(word*POS, voca) for word in ortho_dataset[!, column_name]] for column_name in column_names]


    display("filtering words without embeddings")
    french_forms = [[word for word in french[!, column_name][filter]] for (filter, column_name) in zip(filters, column_names)]
    french_embed = [get_vector(model, word*POS) for (filter, column_name) in zip(filters, column_names) for word in ortho_dataset[!, column_name][filter]]


    new_df = DataFrame()

    column_size = Int64(min((CORPUS_MAX_SIZE / length(column_names)), minimum([length(french_forms[i]) for i in 1:length(french_forms)])))


    # Populate the DataFrame with columns named as per `column_names`, each initialized with `french_form[1]`.
    for i in 1:length(column_names)
        new_df[!, column_names[i]] = french_forms[i][1:column_size] # Assuming `french_form[1]` is the value you want to replicate across the entire column. Adjust the replication as needed.
    end


    forms = [word for column in french_forms for word in column]
    french_form = DataFrame(lexeme = forms)


    display("number of words left after filtering those without embeddings:")
    display(sum([sum(filter) for filter in filters]))

    if VERBOSE == 1
        display("number of words in each column:")
        display([sum(filter) for filter in filters])
    end




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


    # Iterate over words in the dataset
    train_filter = []
    val_filter = []
    help = []

    column_size = 100000


    for column in french_forms
        i = 0
        col = length(help)+1
        
        push!(help, 1)
        for word in column # french_forms[!, column]
            i += 1
            if word ∈ (train_set ∪ val_set) || length((train_set ∪ val_set)) >= CORPUS_MAX_SIZE || i >= column_size
                push!(train_filter, false)
                push!(val_filter, false)
                continue
            end
            word_trigrams = get_trigrams(word)
            if any(trigram ∉ all_trigrams for trigram in word_trigrams) || length(help) <= 1 || length(val_set) >= round(length(train_set) * 0.2) || (i + col) % 2 == 0
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
        dataset_size = length((train_set ∪ val_set))
        train_size = length(train_set)
        val_size = length(val_set)
        
        display("processing column $col, dataset_size = $dataset_size , train_size = $train_size, val_size = $val_size")
    end


    train_size = length(train_set)
    val_size = length(val_set)
    trigram_size = length(all_trigrams)
    train_val_overlap = length(train_set ∩ val_set)
    dataset_size = length((train_set ∪ val_set))
    mkpath(joinpath(@__DIR__, "french_out", "$(POS)_$(MODE)_$dataset_size"))
    display("number of words in train: $train_size")
    display("number of words in val: $val_size")
    display("number of trigrams: $trigram_size")

    if VERBOSE == 1
        display("number of words both in val and train (should be 0): $train_val_overlap")
    end


    train_filter = convert(Array{Bool}, train_filter)
    val_filter = convert(Array{Bool}, val_filter)

    french_form = convert(Array{String}, french_form[!, "lexeme"])

    french_form = DataFrame(lexeme = french_form)

    french_train = french_form[train_filter, :]
    french_val = french_form[val_filter, :]


    # create C matrices for both training and validation datasets
    cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
        french_train,
        french_val,
        grams = 3,
        target_col = "lexeme",
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

    if VERBOSE == 1
        display("C_hat train done")
    end

    Chat_val = S_val * G_train
    if VERBOSE == 1   
        display("C_hat val done")
    end
    Shat_train = cue_obj_train.C * F_train
    if VERBOSE == 1
        display("S_hat train done")
    end
    Shat_val = cue_obj_val.C * F_train
    if VERBOSE == 1
        display("S_hat val done")
    end

    # we evaluate them

    num_rows = size(Chat_train, 1)
    batch_size = 15000

    # Arrays to store evaluation results for averaging later (because train_set is too big)
    train_form_results = Float64[]
    train_sem_results = Float64[]

    for start_idx in 1:batch_size:num_rows
        end_idx = min(start_idx + batch_size - 1, num_rows)
        
        # Extracting the batches
        small_form_matrix = Chat_train[start_idx:end_idx, :]
        small_form_matrix_2 = cue_obj_train.C[start_idx:end_idx, :]
        Shat_train_small = Shat_train[start_idx:end_idx, :]
        S_train_small = S_train[start_idx:end_idx, :]
        
        # Assuming eval_SC returns a scalar value that can be appended to an array
        push!(train_form_results, JudiLing.eval_SC(small_form_matrix, small_form_matrix_2))
        push!(train_sem_results, JudiLing.eval_SC(Shat_train_small, S_train_small))
    end

    # Evaluation 2 and 4 are assumed to be computed on the full validation sets
    eval_form_acc = JudiLing.eval_SC(Chat_val, cue_obj_val.C)
    eval_sem_acc = JudiLing.eval_SC(Shat_val, S_val)

    # Averaging the batch results for train because train was too big
    train_form_acc = mean(train_form_results)
    train_sem_acc = mean(train_sem_results)

    # Printing the final averaged results
    display("------RESULTS------")
    println("train semantic prediction : $(train_sem_acc*100)%")
    println("train form prediction: $(train_form_acc*100)%")
    println("Eval Semantic prediction: $(eval_sem_acc*100)%") # Also computed on the full set
    println("Eval form prediction : $(eval_form_acc*100)%") # This is computed on the full set, not averaged over batches
    display("-------------------")

    # write to CSV the prediction accuracies:

    CSV.write(joinpath(@__DIR__, "french_out", "$(POS)_$(MODE)_$dataset_size", "matrix_scores.csv"), DataFrame(
        model = ["train_semantic", "train_form", "eval_semantic", "eval_form"],
        accuracy = [train_sem_acc, train_form_acc, eval_sem_acc, eval_form_acc]
    ))


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

    """display(french)
    display(res_learn_train)"""
    # you can save results into csv files or dfs
    JudiLing.write2csv(
        res_learn_train,
        french_form,
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
        output_dir = joinpath("french_out", "$(POS)_$(MODE)_$dataset_size"),
    )

    JudiLing.write2csv(
        res_learn_val,
        french_form,
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
        output_dir = joinpath("french_out", "$(POS)_$(MODE)_$dataset_size"),
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
        french_form,
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
        output_dir = joinpath("french_out", "$(POS)_$(MODE)_$dataset_size"),
    )

    JudiLing.write2csv(
        res_build_val,
        french_form,
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
        output_dir = joinpath("french_out", "$(POS)_$(MODE)_$dataset_size"),
    )

    

    acc_build_train =
        JudiLing.eval_acc(res_build_train, cue_obj_train.gold_ind, verbose = false)
    acc_build_val = JudiLing.eval_acc(res_build_val, cue_obj_val.gold_ind, verbose = false)

    # write to a csv the scores:
    
    CSV.write(joinpath(@__DIR__, "french_out", "$(POS)_$(MODE)_$dataset_size", "form_scores.csv"), DataFrame(
        model = ["learn_train", "learn_val", "build_train", "build_val"],
        accuracy = [acc_learn_train, acc_learn_val, acc_build_train, acc_build_val]
    ))



    @show acc_learn_train
    @show acc_learn_val
    @show acc_build_train
    @show acc_build_val

    # Once you are done, you may want to clean up the workspace
    # rm(joinpath(@__DIR__, "data"), force = true, recursive = true)
    # rm(joinpath(@__DIR__, "french_out"), force = true, recursive = true)
end

for CORPUS_MAX_SIZE in [10000, 20000] # choose the upper bound of the corpus size and target (might not be reached)
    for PHONO in [true, false] # choose if you want to use the phonological or orthographic form or both
        for POS in ["_v", "_nc", "_adj"] # choose the part of speech you want to use, several will do a grid run
            MODE = PHONO ? "phoneme" : "grapheme" # variable used for logging
            display("--- STARTING RUN: CORPUS_MAX_SIZE = $CORPUS_MAX_SIZE, $MODE, POS = $POS ---")
            if POS == "_v"
                french = PHONO ? verbs : ortho_verbs
                ortho_dataset = ortho_verbs
            elseif POS == "_nc"
                french = PHONO ? nouns : ortho_nouns
                ortho_dataset = ortho_nouns
            elseif POS == "_adj"
                french = PHONO ? adjs : ortho_adjs
                ortho_dataset = ortho_adjs
            end
            n_row = length(french[!, "lexeme"])
            n_col = length(names(french))
            max_row_size = floor(Int, CORPUS_MAX_SIZE / n_col)
            
            if max_row_size < n_row
                display("reducing the size of the dataset to $max_row_size rows")
                french = french[1:max_row_size, :]
                ortho_dataset = ortho_dataset[1:max_row_size, :]
            end
            
            # make a dir to store the results:
            
            
            
          
            run(french, ortho_dataset, POS, CORPUS_MAX_SIZE, PHONO)
        end
    end
end