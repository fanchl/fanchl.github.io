---
title: Jupyter Notebook 服务器
date: 2021-03-20 09:47:17
tags: 
    - Jupyter Notebook
    - 远程连接
categories: 配置
---

## 1、 生成配置文件

```bash
python
>> from notebook.auth import passwd
>> passwd()
Enter password: 
Verify password: 
'sha1:673a8456a8e8:4377bd9ee8dc33d4cb5a2011f7a89643de15c11c'
```
<!-- more -->

## 2、设置配置文件

可以直接修改配置文件 `~/.jupyter/jupyter_notebook_config.py` ，但需要改动到默认的配置文件中。

也可以自行创建一个配置文件 `~/.jupyter/jupyter_config.py`，然后在运行 Jupyter Notebook 的时候动态加载配置信息。

配置内容如下：

```python
c.NotebookApp.ip = 'localhost' # 指定
c.NotebookApp.open_browser = False # 关闭自动打开浏览器
c.NotebookApp.port = 8899 # 端口随意指定
c.NotebookApp.password = u'sha1:37e84432f5a9:c373b2bcbb673b6ffc0fb9593251db76872500d9' # 复制前一步生成的密钥
c.NotebookApp.notebook_dir = u'/mnt/sda1/fanchl/' # 配置默认目录
```

## 3、启动 Jupyter

```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8899 --allow-root --config=~/.jupyter/jupyter_config.py
```

这样还存在一个问题，就是一旦关闭终端，Jupyter 程序也就终止了运行。这是因为该 Jupyter 程序作为当前终端的子进程，在用户终端关闭的时候将收到一个 hangup 信号，从而被关闭。

所以为了让程序能忽视 hangup 信号，可以使用 `nohup` 命令。同时还要配合 `&` 来将程序放入后台运行。

```bash
nohup jupyter notebook --no-browser --ip=0.0.0.0 --port=8899 --allow-root --config=~/.jupyter/jupyter_config.py &
```

**查看运行的后台进程**

`jobs -l`  命令查看当前终端中后台运行的进程，如果关闭终端后不能再显示了，需要使用ps命令。

`ps -aux | grep jupyter`  查看运行的 `jupyter` 进程

a:显示所有程序

u:以用户为主的格式来显示

x:显示所有程序，不以终端机来区分

`kill -9 pid`  关闭运行中的 jupyter notebook。

**查看端口占用情况**

```json
lsof -i:8899
```