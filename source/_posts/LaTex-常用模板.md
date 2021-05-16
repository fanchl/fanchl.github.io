---
title: LaTex 常用模板
date: 2020-10-22
tags:
    - LaTex
---

## 开篇模板

```latex
\documentclass{ctexart}
\title{}
\author{}
\date{}
\usepackage{geometry}
\usepackage[breaklinks=true,bookmarks=false]{hyperref}
\geometry{a4paper,scale=0.7}
\begin{document}
    \maketitle
    \large

\end{document}
```

<!-- more -->

# 创建参考文献

添加 `name.bib`文件

```latex
\documentclass{ctexart}
\title{}
\author{}
\date{}
\usepackage{geometry}
\usepackage[breaklinks=true,bookmarks=false]{hyperref}
\geometry{a4paper,scale=0.7}
\begin{document}
    \maketitle
    \large

\bibliographystyle{unsrt}
\bibliography{name.bib}
\end{document}

```

在正文中通过 `\cite{}`来引用

同时需要更改编译方式

