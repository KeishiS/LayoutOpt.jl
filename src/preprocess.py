import pykakasi
from functools import reduce
import os

# ------------------
datadir = "static"
outputdir = "data"

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

replace_chars = {
    "？": "?",
    "＜": "<",
    "・": None,
    "○": None,
    "△": None,
    "※": None,
    "…": "...",
    "”": '"',
    "“": '"',
    "`": None,
    " ": None,
    "’": "'",
    "\t": None,
}

files = [
    file for file in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, file))
]

for file in files:
    f = open(os.path.join(datadir, file))
    lines = f.readlines()
    f.close()
    lines = [line.strip() for line in lines if line.strip() != ""]
    if "。" in reduce(lambda a, b: a + b, lines):
        lines = [line + "。" for line in reduce(lambda a, b: a + b, lines).split("。")]
        kks = pykakasi.kakasi()
        lines = [
            reduce(lambda a, b: a + b, [elem["hepburn"] for elem in kks.convert(line)])
            for line in lines
        ]
    text = reduce(lambda a, b: a + b, lines)
    text = text.translate(str.maketrans(replace_chars))
    with open(os.path.join(outputdir, file), "w") as f:
        f.write(text)

texts = []
for file in files:
    with open(os.path.join(outputdir, file)) as f:
        texts.append(f.readline())
text = reduce(lambda a, b: a + b, texts)
print(sorted(list(set(text))))
