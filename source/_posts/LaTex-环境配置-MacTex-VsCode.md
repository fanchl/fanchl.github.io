---
title: LaTex 环境配置 (MacTex & VsCode)
date: 2020-10-21
tags:
    - LaTex
categories: 配置
---

## 环境配置 MacTex + VS Code

1. 下载安装MacTex和VSCode

   ![%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%20MacTex%20+%20VS%20Code%20fb5f49e21c274303880bfca168d097cf/Untitled.png](https://i.loli.net/2021/05/16/lyGJSORTYKa6zbx.png)
   

<!-- more -->

2. 在VSCode内安装 LaTeX Workshop 插件

   ![%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%20MacTex%20+%20VS%20Code%20fb5f49e21c274303880bfca168d097cf/Untitled%201.png](https://i.loli.net/2021/05/16/Is4HgWi7j1CBDoS.png)

3. 修改VSCode中的 settings.json

   ```json
   {
       "workbench.colorTheme": "One Dark Pro",
       "terminal.integrated.shell.osx": "/bin/zsh",
       "terminal.integrated.fontFamily": "Menlo for Powerline",
   
       "latex-workshop.latex.autoBuild.run": "onSave",
       "latex-workshop.showContextMenu": true,
       "latex-workshop.intellisense.package.enabled": true,
       "latex-workshop.message.error.show": true,
       "latex-workshop.message.warning.show": false,
       "latex-workshop.latex.tools": [
           {
               "name": "xelatex",
               "command": "xelatex",
               "args": [
                   "-synctex=1",
                   "-interaction=nonstopmode",
                   "-file-line-error",
                   "%DOCFILE%"
               ]
           },
           {
               "name": "pdflatex",
               "command": "pdflatex",
               "args": [
                   "-synctex=1",
                   "-interaction=nonstopmode",
                   "-file-line-error",
                   "%DOCFILE%"
               ]
           },
           {
               "name": "latexmk",
               "command": "latexmk",
               "args": [
                   "-synctex=1",
                   "-interaction=nonstopmode",
                   "-file-line-error",
                   "-pdf",
                   "-outdir=%OUTDIR%",
                   "%DOCFILE%"
               ]
           },
           {
               "name": "bibtex",
               "command": "bibtex",
               "args": [
                   "%DOCFILE%"
               ]
           }
       ],
       "latex-workshop.latex.recipes": [
           {
               "name": "XeLaTeX",
               "tools": [
                   "xelatex"
               ]
           },
           {
               "name": "PDFLaTeX",
               "tools": [
                   "pdflatex"
               ]
           },
           {
               "name": "BibTeX",
               "tools": [
                   "bibtex"
               ]
           },
           {
               "name": "LaTeXmk",
               "tools": [
                   "latexmk"
               ]
           },
           {
               "name": "xelatex -> bibtex -> xelatex*2",
               "tools": [
                   "xelatex",
                   "bibtex",
                   "xelatex",
                   "xelatex"
               ]
           },
           {
               "name": "pdflatex -> bibtex -> pdflatex*2",
               "tools": [
                   "pdflatex",
                   "bibtex",
                   "pdflatex",
                   "pdflatex"
               ]
           },
       ],
       "latex-workshop.latex.clean.fileTypes": [
           "*.aux",
           "*.bbl",
           "*.blg",
           "*.idx",
           "*.ind",
           "*.lof",
           "*.lot",
           "*.out",
           "*.toc",
           "*.acn",
           "*.acr",
           "*.alg",
           "*.glg",
           "*.glo",
           "*.gls",
           "*.ist",
           "*.fls",
           "*.log",
           "*.fdb_latexmk"
       ],
       "latex-workshop.latex.autoClean.run": "onFailed",
       "latex-workshop.latex.recipe.default": "lastUsed",
       "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
       "workbench.iconTheme": "material-icon-theme",
       "python.defaultInterpreterPath": "/Users/fan/anaconda3/envs/ML/bin/python",
       "workbench.startupEditor": "none",
       "git.confirmSync": false,
       "workbench.editorAssociations": [
           {
               "viewType": "jupyter.notebook.ipynb",
               "filenamePattern": "*.ipynb"
           }
       ],
   
   }
   ```