#!/usr/bin/python3

import json
import os

f = open("cycles_quasicycles_LSE.ipynb", "r")
data = json.load(f)
f.close()

text = ""
for cell in (data['cells']):
    if cell['cell_type'] == "markdown":
        for line in cell['source']:
            text += line
        text += "\n\n"

md = open("notebookMarkdown.md", "w")
md.write(text)
md.close()

os.system("markdown-toc --no-firsth1 notebookMarkdown.md")
os.system("rm -f notebookMarkdown.md")
