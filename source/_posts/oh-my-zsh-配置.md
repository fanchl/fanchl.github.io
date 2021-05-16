---
title: oh-my-zsh 配置
date: 2020-08-11
tags:
    - macOs
    - 终端
categories: 配置
---

# oh-my-zsh 配置

## 一、clone 项目

```bash
git clone https://github.com.cnpmjs.org/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
```

## 二、复制 .zshrc

```bash
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
```

<!-- more -->

## 三、更改默认 Shell

```bash
chsh -s /bin/zsh
```

查看当前shell

```bash
echo $SHELL
# "如果切换之后还是bash，重启终端"
```

激活配置文件 `.zshrc`

```bash
source .zshrc
```

得到结果如下：

![oh-my-zsh%20%E9%85%8D%E7%BD%AE%20ecf592a1f6604e968d0fc07981e66d1c/Untitled.png](https://i.loli.net/2021/05/16/XPYRoOyv6Bh81eM.png)

## 四、配置主题

在 `.zshrc`配置文件中修改

```bash
ZSH_THEME="agnoster"
```

激活配置文件`.zshrc`

```bash
source .zshrc
```

得到结果如下：

![oh-my-zsh%20%E9%85%8D%E7%BD%AE%20ecf592a1f6604e968d0fc07981e66d1c/Untitled%201.png](https://i.loli.net/2021/05/16/iuhd8XHFnJfPjDG.png)

## 五、配置插件

### （1）zsh-syntax-highlighting(命令语法高亮)

1. Clone项目到$ZSH_CUSTOM/plugins文件夹下 (默认为 ~/.oh-my-zsh/custom/plugins)

   ```bash
   git clone https://github.com.cnpmjs.org/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
   ```

2. 在 Oh My Zsh 的配置文件 (~/.zshrc)中设置:

   ```bash
   plugins=(其他插件 zsh-syntax-highlighting)
   ```

3. 运行 `source ~/.zshrc` 更新配置

### （2）zsh-autosuggestions(命令自动补全)

1. Clone项目到$ZSH_CUSTOM/plugins文件夹下 (默认为 ~/.oh-my-zsh/custom/plugins)

   ```bash
   git clone https://github.com.cnpmjs.org/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
   ```

2. 在 Oh My Zsh 的配置文件 (~/.zshrc)中设置:

   ```bash
   plugins=(其他插件 zsh-autosuggestions)
   ```

3. 运行 `source ~/.zshrc` 更新配置

   > 当重新打开终端的时候可能看不到变化，可能你的字体颜色太淡了，我们把其改亮一些：

   ```bash
   cd ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
   vim zsh-autosuggestions.zsh
   vim ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
   # 修改 ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=10'
   ```

   > 修改成功后需要运行 source ~/.zshrc 更新配置

4. 自定义快捷键

   如果感觉 → 补全不方便，还可以自定义补全的快捷键，比如我设置的逗号补全，只需要在 `.zshrc` 文件下方添加这句话

   ```bash
   bindkey ',' autosuggest-accept
   ```

   运行 `source ~/.zshrc` 更新配置


## 六、.zshrc 样例

```bash
# If you come from bash you might have to change your $PATH.
export PATH=$PATH:~/anaconda3/bin

export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_281.jdk/Contents/Home

# Path to your oh-my-zsh installation.
export ZSH="/Users/fan/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="agnoster"

# Set list of themes to pick from when loading at random
# Setting this variable when ZSH_THEME=random will cause zsh to load
# a theme from this variable instead of looking in $ZSH/themes/
# If set to an empty array, this variable will have no effect.
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment the following line to disable bi-weekly auto-update checks.
# DISABLE_AUTO_UPDATE="true"

# Uncomment the following line to automatically update without prompting.
# DISABLE_UPDATE_PROMPT="true"

# Uncomment the following line to change how often to auto-update (in days).
# export UPDATE_ZSH_DAYS=13

# Uncomment the following line if pasting URLs and other text is messed up.
# DISABLE_MAGIC_FUNCTIONS="true"

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
# ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
# COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# You can set one of the optional three formats:
# "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
# or set a custom format using the strftime function format specifications,
# see 'man strftime' for details.
# HIST_STAMPS="mm/dd/yyyy"

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(git zsh-syntax-highlighting zsh-autosuggestions)

source $ZSH/oh-my-zsh.sh
bindkey ',' autosuggest-accept

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='mvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
```