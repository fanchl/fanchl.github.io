---
title: Vim 配置
date: 2021-01-18
tags: Vim
categories: 配置
---

# Vim 配置

Vim编辑器相关的所有功能开关都可以通过 **`.vimrc`** 文件进行设置。 

**`.vimrc`**配置文件分系统配置和用户配置两种。

- 系统vimrc配置文件存放在Vim的安装目录，默认路径为`/usr/share/vim/.vimrc`。可以使用命令`echo $VIM`来确定Vim的安装目录。
- 用户vimrc文件，存放在用户主目录下`~/.vimrc`。可以使用命令`echo $HOME`确定用户主目录。

*注意*：用户配置文件优先于系统配置文件，Vim启动时会优先读取当前用户根目录下的**.vimrc**文件。所以与个人用户相关的个性化配置一般都放在`~/.vimrc`中。

<!-- more -->

## 使更改生效

要让.vimrc变更内容生效，一般的做法是先保存 .vimrc 再重启vim，增加如下设置，可以实现保存 .vimrc 时自动重启加载。

```bash
'让vimrc配置变更立即生效'
autocmd BufWritePost $MYVIMRC source $MYVIMRC
```

## 一、基础设置

### 1、设置中文不乱码

```bash
'设置编码'
set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
```

### 2、显示行号

```bash
'显示行号'
set nu
set number
```

### 3、突出显示当前行

```bash
set cursorline
set cul          'cursorline的缩写形式'
```

### 4、启用鼠标

```bash
set mouse=a
set selection=exclusive
set selectmode=mouse,key
```

Vim编辑器里默认是不启用鼠标的，也就是说不管你鼠标点击哪个位置，光标都不会移动。通过以上设置就可以启动鼠标。

### 5、设置缩进

```bash
'设置Tab长度为4空格'
set tabstop=4
'设置自动缩进长度为4空格'
set shiftwidth=4
'继承前一行的缩进方式，适用于多行注释'
set autoindent
```

## 二、主题配置

```bash
mkdir ~/.vim
git clone https://gitclone.com/github.com/flazz/vim-colorschemes.git ~/.vim
```

```bash
# .vimrc
colorscheme janah
```

## 三、整体配置

```bash
set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
set nu
set number
set mouse=a
set selection=exclusive
set selectmode=mouse,key
set tabstop=4
set shiftwidth=4
set autoindent

colorscheme janah
```

## 四、推荐配置

配置可以直接使用网友共享的优化配置：

[https://github.com/amix/vimrc](https://github.com/amix/vimrc)

```bash
git clone --depth=1 https://gitclone.com/github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh
```

Vim Plugins - NERDTree

Vim插件，装了这个插件，便可以显示树形目录结构。

```bash
git clone https://gitclone.com/github.com/scrooloose/nerdtree.git ~/.vim/bundle/nerdtree
```

重启Vim，在命令模式下输入NERDTree即可开启目录展示，默认是当前路径。

```bash
:NERDTree
```