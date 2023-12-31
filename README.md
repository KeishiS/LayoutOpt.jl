# README

## Prerequisites

1. 評価に使いたい英文，日本語文のtxtファイルを `static` ディレクトリ内に格納してください．例えば [青空文庫の'吾輩は猫である'](https://www.aozora.gr.jp/cards/000148/files/789_14547.html)の本文のみを保存したtxtファイルなどを想定しています．
1. 評価に使えるよう，アルファベット+記号のみの文章へ変換
    1. `python -m venv .venv`
    1. `source .venv/bin/activate`
    1. `pip install -r requirements.txt`
    1. `python src/preprocess.py`
1. Julia環境の設定
    1. `julia --project=.`
    1. `pkg> instantiate`

## 実行

```julia
julia --project=.

julia> using LayoutOpt
julia> layouts = solve(200; n_samples=100);
julia> layouts[1] # スコアが最も良い配列
```