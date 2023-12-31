# README

## Prerequisites

1. 評価に使いたい英文，日本語文のtxtファイルを `static` ディレクトリ内に格納してください．例えば [青空文庫の'吾輩は猫である'](https://www.aozora.gr.jp/cards/000148/files/789_14547.html)の本文のみを保存したtxtファイルなどを想定しています．
2. 評価に使えるよう，アルファベット+記号のみの文章へ変換

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
```

3. Julia環境の設定

```julia
julia --project=.
pkg> instantiate
```

## How to use

```julia
julia --project=.
julia> using LayoutOpt
julia> layouts = solve(100; n_samples=100);
julia> layouts[1] # スコアが最も良い配列
```

## Output

自分の手元で実行した際の結果が以下の通り

```
------------------------------------------------------------------------------------------
    |   1 |   2 |   3 |   4 |   5 |   6 ||   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |
    |------------------------------------------------------------------------------------|
  1 |     | 1 * | 2 % | 3 ! | 4 ] | 5 } || 6 ^ | 7 @ | 8 ~ | 9 + | 0 \ | j J | q Q |     |
  2 |     | f F | v V | d D | y Y | x X || - > | : " | , ; | h H | [ | | # $ | ' ? |     |
  3 |     | r R | l L | t T | s S | n N || o O | i I | u U | p P | e E | a A | & < |     |
  4 |     | k K | w W | c C | m M | b B || _ ( | ) . | g G | z Z | = { | /   |     |     |
------------------------------------------------------------------------------------------
```