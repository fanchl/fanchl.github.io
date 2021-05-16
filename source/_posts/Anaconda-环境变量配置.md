---
title: Anaconda 环境变量配置
date: 2020-09-15
tags:
    - Anaconda
    - Linux
categories: 配置
---

## 环境变量

如果需要初始化虚拟环境则添加

```bash
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/Users/fan/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/Users/fan/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/fan/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/Users/fan/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<
```

<!-- more -->

```bash
export PATH="/Users/fan/anaconda3/bin:$PATH"
```

