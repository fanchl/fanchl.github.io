---
title: Jupyter Notebook 选择Kernel
date: 2021-03-20 09:43:59
tags: 
    - Jupyter Notebook
    - 远程连接
---

# 选择 Kernel

## 一、创建对应 conda 环境的 kernel

### 激活 conda 环境

```bash
source activate 环境名称
```

### 安装 `ipykernel`

```bash
conda install ipykernel
```

<!-- more -->

### 将环境写入 notebook 的 kernel 中

```bash
python -m ipykernel install --user --name 环境名称 --display-name "conda (环境名称)"

# python -m ipykernel install --user --name DeepLog --display-name "conda (DeepLog)"
```

会在 `~/.local/share/jupyter/kernels/` 生成 `deeplog` 文件夹，其中有 `kernel.json` 文件 

```json
{
 "argv": [
  "/mnt/sda1/fanchl/anaconda3/envs/Deeplog/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "conda(DeepLog)",
 "language": "python"
}
```

## 二、删除指定 kernel

### 1、查看所有核心

```bash
jupyter kernelspec list
```

### 2、卸载指定核心

```bash
jupyter kernelspec remove kernel_name
```