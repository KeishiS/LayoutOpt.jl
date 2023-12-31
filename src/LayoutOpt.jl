module LayoutOpt

using
    Random,
    Distributions,
    Statistics,
    LinearAlgebra,
    StatsBase

export
    Layout,
    swap_two_keys,
    crossover,
    calc_score,
    calc_score_from_file,
    one_generation,
    solve,
    all_chars,
    qwerty,
    dvorak

fingers = [
    [
        [4, 5, 10, 11, 16, 17, 21, 22, 23], # L2
        [3, 9, 15], # L3
        [2, 8, 14, 20], # L4
        [0, 1, 7, 13, 19] # L5
    ],
    [
        [24, 25, 32, 33, 40, 41, 48, 49], # R2
        [26, 34, 42, 50], # R3
        [27, 35, 43], # R4
        [28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 51, 52, 53, 54, 55], # R5
    ],
]
trip = Set{Tuple{Int,Int}}()
doub = Set{Tuple{Int,Int}}()
# doub
for k in 1:2
    for fing₁ in fingers[k]
        for fing₂ in fingers[k]
            if fing₁ == fing₂
                continue
            end
            for i in fing₁
                for j in fing₂
                    push!(doub, (i, j))
                end
            end
        end
    end
end

# trip
for k in 1:2
    for fing in fingers[k]
        for i in fing
            for j in fing
                if i == j
                    continue
                end
                push!(trip, (i, j))
            end
        end
    end
end

const low_alphabets = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z']
const numbers = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const symbols = [
    '!', '"', '#', '$', '&', '\'', '(', ')', '*', '+',
    ',', '-', '.', '/', '%', '^', '|', ':', ';', '<',
    '=', '>', '?', '@', '[', '\\', ']', '_', '{', '}',
    '~']
const chars = vcat(low_alphabets, uppercase.(low_alphabets), numbers, symbols)
const ng_coord = Set([
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),
    (2, 1, 8), (2, 2, 8), (2, 3, 8), (2, 4, 7), (2, 4, 8)])

function to_charmap(key)
    ret = Dict{Char,Any}()

    for i in 1:2
        for k in 1:2
            for row in 1:4
                cols = i == 1 ? range(1, 6) : range(1, 8)
                for col in cols
                    c = key[i][k, row, col]
                    if !(c in chars)
                        continue
                    end
                    ret[c] = (i, k, row, col)
                end
            end
        end
    end
    ret
end

function generate_random_layout(;
    seed=42,
    init=Dict(
        '1' => (1, 1, 1, 2),
        '2' => (1, 1, 1, 3),
        '3' => (1, 1, 1, 4),
        '4' => (1, 1, 1, 5),
        '5' => (1, 1, 1, 6),
        '6' => (2, 1, 1, 1),
        '7' => (2, 1, 1, 2),
        '8' => (2, 1, 1, 3),
        '9' => (2, 1, 1, 4),
        '0' => (2, 1, 1, 5)
    )
)
    Random.seed!(seed)
    key = Array{Array{Char}}(undef, 2)
    key[1] = fill('\0', (2, 4, 6))
    key[2] = fill('\0', (2, 4, 8))
    # 初期状態があれば先に埋める
    for (val, (i, k, row, col)) in init
        key[i][k, row, col] = val
    end

    alpha_remains = setdiff(low_alphabets, keys(init)) |> shuffle
    symb_remains = setdiff(symbols, keys(init)) |> shuffle
    num_remains = setdiff(numbers, keys(init)) |> shuffle
    cnt_alpha = 1
    cnt_symb = 1
    cnt_num = 1

    # shiftなし面を埋める
    for k in 1:2
        for i in 1:2
            for row in 1:4
                cols = i == 1 ? range(1, 6) : range(1, 8)
                for col in cols
                    if (i, row, col) in ng_coord || key[i][k, row, col] != '\0'
                        continue
                    end
                    if cnt_alpha <= length(alpha_remains)
                        if k != 1
                            throw(ErrorException("alphabet wrong"))
                        end
                        key[i][k, row, col] = alpha_remains[cnt_alpha]
                        key[i][k+1, row, col] = uppercase(key[i][k, row, col])
                        cnt_alpha += 1
                    elseif cnt_symb <= length(symb_remains)
                        key[i][k, row, col] = symb_remains[cnt_symb]
                        cnt_symb += 1
                    elseif cnt_num <= length(num_remains)
                        key[i][k, row, col] = num_remains[cnt_num]
                        cnt_num += 1
                    end
                end
            end
        end
    end
    key
end

mutable struct Layout
    key
    charmap
    scores
    function Layout(;
        seed=rand(1:typemax(Int)),
        init=Dict(
            '1' => (1, 1, 1, 2),
            '2' => (1, 1, 1, 3),
            '3' => (1, 1, 1, 4),
            '4' => (1, 1, 1, 5),
            '5' => (1, 1, 1, 6),
            '6' => (2, 1, 1, 1),
            '7' => (2, 1, 1, 2),
            '8' => (2, 1, 1, 3),
            '9' => (2, 1, 1, 4),
            '0' => (2, 1, 1, 5))
    )
        layout = new()
        layout.key = generate_random_layout(; seed=seed, init=init)
        layout.charmap = to_charmap(layout.key)
        layout.scores = Inf
        return layout
    end
    function Layout(left, right)
        layout = new()
        layout.key = Array{Array{Char}}(undef, 2)
        layout.key[1] = copy(left)
        layout.key[2] = copy(right)
        layout.charmap = to_charmap(layout.key)
        layout.scores = Inf
        layout
    end
    function Layout(left, right, charmap::Dict{Char,Any})
        layout = new()
        layout.key = Array{Array{Char}}(undef, 2)
        layout.key[1] = copy(left)
        layout.key[2] = copy(right)
        layout.charmap = charmap
        layout.scores = Inf
        layout
    end
end

function Base.show(io::IO, layout::Layout)
    println(io, "------------------------------------------------------------------------------------------")
    println(io, "    |   1 |   2 |   3 |   4 |   5 |   6 ||   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |")
    println(io, "    |------------------------------------------------------------------------------------|")
    for row in 1:4
        print(io, "  $(row) |")
        for col in 1:6
            c₁ = layout.key[1][1, row, col]
            c₂ = layout.key[1][2, row, col]
            print(io, " $(c₁ in chars ? c₁ : ' ') $(c₂ in chars ? c₂ : ' ') |")
        end
        print(io, "|")
        for col in 1:8
            c₁ = layout.key[2][1, row, col]
            c₂ = layout.key[2][2, row, col]
            print(io, " $(c₁ in chars ? c₁ : ' ') $(c₂ in chars ? c₂ : ' ') |")
        end
        println(io, "")
    end
    println(io, "------------------------------------------------------------------------------------------")
    println(io, "")
end

Base.isless(lay1::Layout, lay2::Layout) = lay1.scores < lay2.scores

function all_chars(layout::Layout)
    vec(layout.key[1]) ∪ vec(layout.key[2]) |> unique |> sort
end



#---[QWERTY]--------------------------------
qwerty_lft = fill('\0', (2, 4, 6))
qwerty_lft[1, :, :] = [
    ' ' '1' '2' '3' '4' '5'
    ' ' 'q' 'w' 'e' 'r' 't'
    ' ' 'a' 's' 'd' 'f' 'g'
    ' ' 'z' 'x' 'c' 'v' 'b'
]
qwerty_lft[2, :, :] = [
    ' ' '!' '"' '#' '$' '%'
    ' ' 'Q' 'W' 'E' 'R' 'T'
    ' ' 'A' 'S' 'D' 'F' 'G'
    ' ' 'Z' 'X' 'C' 'V' 'B'
]
qwerty_rht = fill('\0', (2, 4, 8))
qwerty_rht[1, :, :] = [
    '6' '7' '8' '9' '0' '-' '^' ' '
    'y' 'u' 'i' 'o' 'p' '@' '[' ' '
    'h' 'j' 'k' 'l' ';' ':' ']' ' '
    'n' 'm' ',' '.' '/' '\\' ' ' ' '
]
qwerty_rht[2, :, :] = [
    '&' '\'' '(' ')' ' ' '=' '~' '|'
    'Y' 'U' 'I' 'O' 'P' '`' '{' ' '
    'H' 'J' 'K' 'L' '+' '*' '}' ' '
    'N' 'M' '<' '>' '?' '_' ' ' ' '
]
qwerty = Layout(qwerty_lft, qwerty_rht)

#---[DVORAK]--------------------------------
dvorak_lft = fill('\0', (2, 4, 6))
dvorak_lft[1, :, :] = [
    '$' '&' '[' '{' '}' '('
    ' ' ';' ',' '.' 'p' 'y'
    ' ' 'a' 'o' 'e' 'u' 'i'
    ' ' '\'' 'q' 'j' 'k' 'x'
]
dvorak_lft[2, :, :] = [
    '~' '%' '7' '5' '3' '1'
    ' ' ':' '<' '>' 'P' 'Y'
    ' ' 'A' 'O' 'E' 'U' 'I'
    ' ' '\"' 'Q' 'J' 'K' 'X'
]

dvorak_rht = fill('\0', (2, 4, 8))
dvorak_rht[1, :, :] = [
    '=' '*' ')' '+' ']' '!' '#' ' '
    'f' 'g' 'c' 'r' 'l' '/' '@' '\\'
    'd' 'h' 't' 'n' 's' '-' ' ' ' '
    'b' 'm' 'w' 'v' 'z' ' ' ' ' ' '
]
dvorak_rht[2, :, :] = [
    '9' '0' '2' '4' '6' '8' '`' ' '
    'F' 'G' 'C' 'R' 'L' '?' '^' '|'
    'D' 'H' 'T' 'N' 'S' '_' ' ' ' '
    'B' 'M' 'W' 'V' 'Z' ' ' ' ' ' '
]
dvorak = Layout(dvorak_lft, dvorak_rht)

const lft_costs = [
    2.0 1.80 1.75 1.70 1.65 1.60
    Inf 1.50 1.50 0.80 0.85 1.20
    Inf 0.25 0.50 0.25 0.25 0.30
    Inf 1.20 1.50 1.00 1.20 1.30
]

const rht_costs = [
    1.60 1.65 1.70 1.75 1.80 1.85 1.90 2.00
    0.80 0.75 0.50 0.30 1.50 1.55 1.60 1.65
    0.30 0.25 0.30 0.35 0.25 0.30 1.50 1.60
    0.35 0.30 0.40 1.25 0.80 1.00 1.20 1.25
]

function calc_score(layout::Layout, text::String)
    N = length(text)
    conv((i, _, row, col)) = i == 1 ? 6 * (row - 1) + (col - 1) : 8 * (row - 1) + (col - 1) + 24
    cost((i, _, row, col)) = i == 1 ? lft_costs[row, col] : rht_costs[row, col]
    extract((_, k, _, _)) = k
    f(c) = getindex(layout.charmap, c)

    fₛ = text |> collect .|> f .|> conv
    gₛ = text |> collect .|> f .|> extract
    gₛ = (gₛ .- 1) .* 0.2 .+ 1
    fs = hcat(fₛ[1:end-1], fₛ[2:end])

    muls = in.(Tuple.(eachrow(fs)), Ref(trip)) .* 3.0 + in.(Tuple.(eachrow(fs)), Ref(doub)) .* 2.0
    muls[muls.==0] .= 1.0

    costₛ = ((text |> collect .|> f .|> cost) .+ 1) .* gₛ
    costs = hcat(costₛ[1:end-1], costₛ[2:end])

    dot(vec(prod(costs, dims=2)), muls) / N
end

function calc_score_old(layout::Layout, text::String)
    N = length(text)
    score = 0.0
    for i in 1:N-1
        cᵢ = text[i]
        cⱼ = text[i+1]
        (i_hand, i_sft, i_row, i_col) = layout.charmap[cᵢ]
        (j_hand, j_sft, j_row, j_col) = layout.charmap[cⱼ]
        fᵢ = i_hand == 1 ? 6 * (i_row - 1) + (i_col - 1) : 8 * (i_row - 1) + (i_col - 1) + 24
        fⱼ = j_hand == 1 ? 6 * (j_row - 1) + (j_col - 1) : 8 * (j_row - 1) + (j_col - 1) + 24
        mul = (fᵢ, fⱼ) in trip ? 3.0 : (fᵢ, fⱼ) in doub ? 2.0 : 1.0
        i_cost = i_hand == 1 ? lft_costs[i_row, i_col] : rht_costs[i_row, i_col]
        j_cost = j_hand == 1 ? lft_costs[j_row, j_col] : rht_costs[j_row, j_col]
        score += ((i_cost + 1) * (1 + 0.2 * (i_sft - 1)) * (j_cost + 1) * (1 + 0.2 * (j_sft - 1))) * mul
    end

    score / N
end

function calc_score_from_file(layout::Layout, filepath::String)
    io = open(filepath, "r")
    text = readline(io)
    close(io)

    calc_score(layout, text)
end

function swap_two_symb(layout::Layout; seed=rand(1:typemax(Int)), fixed=[])
    Random.seed!(seed)
    key = Array{Array{Char}}(undef, 2)
    key[1] = copy(layout.key[1])
    key[2] = copy(layout.key[2])

    cs = sample(setdiff(symbols, fixed), 2; replace=false)
    (i, i_sft, i_row, i_col) = layout.charmap[cs[1]]
    (j, j_sft, j_row, j_col) = layout.charmap[cs[2]]
    key[i][i_sft, i_row, i_col] = cs[2]
    key[j][j_sft, j_row, j_col] = cs[1]

    Layout(key[1], key[2])
end

function swap_two_keys(layout::Layout; seed=rand(1:typemax(Int)), fixed=[])
    Random.seed!(seed)
    key = Array{Array{Char}}(undef, 2)
    key[1] = copy(layout.key[1])
    key[2] = copy(layout.key[2])

    i, i_row, i_col = 1, 1, 1
    j, j_row, j_col = 1, 1, 1
    cᵢ₁, cᵢ₂, cⱼ₁, cⱼ₂ = nothing, nothing, nothing, nothing
    while true
        i, j = rand(1:2), rand(1:2)
        i_row, j_row = rand(1:4), rand(1:4)
        i_col = i == 1 ? rand(1:6) : rand(1:8)
        j_col = j == 1 ? rand(1:6) : rand(1:8)
        if (i, i_row, i_col) ∈ ng_coord || (j, j_row, j_col) ∈ ng_coord || (i, i_row, i_col) == (j, j_row, j_col)
            continue
        end

        cᵢ₁, cᵢ₂ = key[i][1, i_row, i_col], key[i][2, i_row, i_col]
        cⱼ₁, cⱼ₂ = key[j][1, j_row, j_col], key[j][2, j_row, j_col]
        if cᵢ₁ ∈ fixed || cᵢ₂ ∈ fixed || cⱼ₁ ∈ fixed || cⱼ₂ ∈ fixed
            continue
        elseif cᵢ₁ ∈ chars && cᵢ₁ ∈ chars && cⱼ₁ ∈ chars && cⱼ₂ ∈ chars
            break
        end
    end

    # 単純にswap
    key[i][1, i_row, i_col] = cⱼ₁
    key[i][2, i_row, i_col] = cⱼ₂
    key[j][1, j_row, j_col] = cᵢ₁
    key[j][2, j_row, j_col] = cᵢ₂

    if !(cᵢ₁ in vcat('A':'Z', 'a':'z')) && !(cⱼ₁ in vcat('A':'Z', 'a':'z'))
        if Bool(rand(0:1)) # shiftでswap
            key[i][1, i_row, i_col], key[i][2, i_row, i_col] =
                key[i][2, i_row, i_col], key[i][1, i_row, i_col]
        end
        if Bool(rand(0:1)) # shiftでswap
            key[j][1, j_row, j_col], key[j][2, j_row, j_col] =
                key[j][2, j_row, j_col], key[j][1, j_row, j_col]
        end
    end

    Layout(key[1], key[2])
end

function crossover(lay1::Layout, lay2::Layout; seed=rand(1:typemax(Int)))
    Random.seed!(seed)
    key = Array{Array{Char}}(undef, 2)
    key[1] = copy(lay1.key[1])
    key[2] = copy(lay2.key[2])
    uniq = setdiff(
        setdiff(
            lowercase.(vec(key[1]) ∪ vec(key[2])),
            lowercase.(vec(key[1]) ∩ vec(key[2]))
        ),
        ['\0', ' ']
    ) |> sort
    remains_nonalpha = setdiff(vcat(numbers, symbols), uniq) |> shuffle
    remains_alpha = setdiff(low_alphabets, uniq) |> shuffle


    # 残す場所以外 '\0' で初期化
    for i in 1:2
        for k in 1:2
            for row in 1:4
                cols = i == 1 ? range(1, 6) : range(1, 8)
                for col in cols
                    c = key[i][k, row, col]
                    c = lowercase(c)
                    if !(c in uniq)
                        key[i][k, row, col] = '\0'
                    end
                end
            end
        end
    end
    coord_alpha = []
    for i in 1:2
        for row in 1:4
            cols = i == 1 ? range(1, 6) : range(1, 8)
            for col in cols
                if (i, row, col) in ng_coord
                    continue
                end
                if key[i][1, row, col] == key[i][2, row, col] && key[i][1, row, col] == '\0'
                    push!(coord_alpha, (i, row, col))
                end
            end
        end
    end
    if length(coord_alpha) < length(remains_alpha)
        throw(ErrorException("something wrong"))
    end
    shuffle!(coord_alpha)

    for (id, c) in enumerate(remains_alpha)
        (i, row, col) = coord_alpha[id]
        key[i][1, row, col] = lowercase(c)
        key[i][2, row, col] = uppercase(c)
    end

    cnt_nonalpha = 1
    for k in 1:2
        for i in 1:2
            for row in 1:4
                cols = i == 1 ? range(1, 6) : range(1, 8)
                for col in cols
                    c = key[i][k, row, col]
                    if (i, row, col) in ng_coord ||
                       c in chars ||
                       cnt_nonalpha > length(remains_nonalpha)
                        continue
                    end
                    key[i][k, row, col] = remains_nonalpha[cnt_nonalpha]
                    cnt_nonalpha += 1
                end
            end
        end
    end

    Layout(key[1], key[2])
end

function fit2probs(vals)
    fmax = maximum(vals)
    vals = fmax * 1.2 .- vals
    vals ./ sum(vals)
end

function one_generation(layouts, texts; fixed=[])
    N = length(layouts)
    ret = Layout[]

    # 増やして
    ## cld(N,20)個を単体選別
    sort!(layouts)
    ret = vcat(ret, layouts[1:cld(N, 10)])

    ## crossoverで+N個
    probs = fit2probs([layout.scores for layout in layouts])
    ret2 = [
        Layout[]
        for _ in 1:Threads.nthreads()
    ]
    Threads.@threads for i in 1:N
        tid = Threads.threadid()
        id₁, id₂ = nothing, nothing
        id₁ = rand(Categorical(probs))
        while true
            id₂ = rand(Categorical(probs))
            if id₁ != id₂
                try
                    child = crossover(layouts[id₁], layouts[id₂])
                    push!(ret2[tid], child)
                catch err
                    break
                end
                break
            end
        end
    end
    ret2 = foldl(vcat, ret2)
    ret = vcat(ret, ret2)

    ## cld(N,10)+N個に対して，swapしたものを追加
    ret2 = [
        Layout[]
        for _ in 1:Threads.nthreads()
    ]
    Threads.@threads for layout in ret
        tid = Threads.threadid()
        K = rand(0:10)
        lay = swap_two_keys(layout; fixed=fixed)
        for _ in 1:K
            lay = swap_two_keys(lay; fixed=fixed)
        end
        push!(ret2[tid], lay)
    end
    ret2 = foldl(vcat, ret2)
    ret = vcat(ret, ret2)

    ## cld(N,10)+N個に対して，symbolをswapしたものを追加
    ret2 = [
        Layout[]
        for _ in 1:Threads.nthreads()
    ]
    Threads.@threads for layout in ret
        tid = Threads.threadid()
        K = rand(0:10)
        lay = swap_two_symb(layout; fixed=fixed)
        for _ in 1:K
            lay = swap_two_symb(lay; fixed=fixed)
        end
        push!(ret2[tid], lay)
    end
    ret2 = foldl(vcat, ret2)
    ret = vcat(ret, ret2)

    # 評価して
    Threads.@threads for i in eachindex(ret)
        ret[i].scores = mean([calc_score(ret[i], text) for text in texts])
    end

    # 選定
    sort!(ret)

    ret[1:N]
end

function solve(N::Int;
    n_samples=50, datadir="data/",
    init=Dict(
        '1' => (1, 1, 1, 2),
        '2' => (1, 1, 1, 3),
        '3' => (1, 1, 1, 4),
        '4' => (1, 1, 1, 5),
        '5' => (1, 1, 1, 6),
        '6' => (2, 1, 1, 1),
        '7' => (2, 1, 1, 2),
        '8' => (2, 1, 1, 3),
        '9' => (2, 1, 1, 4),
        '0' => (2, 1, 1, 5)
    ))
    files = [file for file in readdir(datadir, join=true) if isfile(file)]
    if length(files) == 0
        throw(ErrorException("no file found for evaluation"))
    end
    texts = String[]
    for file in files
        io = open(file, "r")
        push!(texts, readline(io))
        close(io)
    end

    layouts = Layout[Layout() for _ in 1:n_samples]
    Threads.@threads for i in eachindex(layouts)
        layouts[i].scores = mean([calc_score(layouts[i], text) for text in texts])
    end

    means = []
    mins = []
    maxes = []
    stds = []
    for loop in 1:N
        println("[START] gen $(loop)")
        layouts = one_generation(layouts, texts; fixed=keys(init))
        push!(means, mean([lay.scores for lay in layouts]))
        push!(mins, minimum([lay.scores for lay in layouts]))
        push!(maxes, maximum([lay.scores for lay in layouts]))
        push!(stds, std([lay.scores for lay in layouts]))

        println(layouts[1])
        println("[ END ] gen $(loop) min: $(mins[end]), max: $(maxes[end]), mean: $(means[end]), std: $(stds[end])")
        println("-------------------------------------------------------------------")
    end

    return layouts
end

end # module LayoutOpt
