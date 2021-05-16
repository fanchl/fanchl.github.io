---
title: Bash 技巧
date: 2020-12-27
tags:
    - Linux
---

# Bash 技巧

# 模式扩展

## 子命令扩展

1. $(...)可以扩展成另一个命令的运行结果，该命令的所有输出都会作为返回值。

   $(...)可以嵌套，比如$(ls $(pwd))

   ```bash
   $ echo $(date)
   Tue Jan 28 00:01:13 CST 2020
   
   ```

<!-- more -->

2. 还有另一种较老的语法，子命令放在反引号之中，也可以扩展成命令的运行结果。

   ```bash
   $ echo `date`
   Tue Jan 28 00:01:13 CST 2020
   ```

# 变量

1. 环境变量

   ```bash
   env
   printenv
   ```

   常见的环境变量：

   - `BASHPID`：Bash 进程的进程 ID。
   - `BASHOPTS`：当前 Shell 的参数，可以用`shopt`命令修改。
   - `DISPLAY`：图形环境的显示器名字，通常是`:0`，表示 X Server 的第一个显示器。
   - `EDITOR`：默认的文本编辑器。
   - `HOME`：用户的主目录。
   - `HOST`：当前主机的名称。
   - `IFS`：词与词之间的分隔符，默认为空格。
   - `LANG`：字符集以及语言编码，比如`zh_CN.UTF-8`。
   - `PATH`：由冒号分开的目录列表，当输入可执行程序名后，会搜索这个目录列表。
   - `PS1`：Shell 提示符。
   - `PS2`： 输入多行命令时，次要的 Shell 提示符。
   - `PWD`：当前工作目录。
   - `RANDOM`：返回一个0到32767之间的随机数。
   - `SHELL`：Shell 的名字。
   - `SHELLOPTS`：启动当前 Shell 的`set`命令的参数，参见《set 命令》一章。
   - `TERM`：终端类型名，即终端仿真器所用的协议。
   - `UID`：当前用户的 ID 编号。
   - `USER`：当前用户的用户名。

   查看单个环境变量的值

   ```bash
   printenv PATH
   echo $PATH
   ```

2. 自定义变量

   自定义变量是用户在当前 Shell 里面自己定义的变量，必须先定义后使用，而且仅在当前 Shell 可用。一旦退出当前 Shell，该变量就不存在了。

   `set`命令可以显示所有变量（包括环境变量和自定义变量），以及所有的 Bash 函数。

   ```bash
   $ set
   ```

## 创建变量

用户创建变量的时候，变量名必须遵守下面的规则。

- 字母、数字和下划线字符组成。
- 第一个字符必须是一个字母或一个下划线，不能是数字。
- 不允许出现空格和标点符号。

变量声明的语法如下。

```bash
variable=value
```

上面命令中，等号左边是变量名，右边是变量。注意，**等号两边不能有空格**。

如果变量的值包含空格，则必须将值放在引号中。

```bash
myvar="hello world"
```

Bash 没有数据类型的概念，所有的变量值都是字符串。

## 读取变量

读取变量的时候，直接在变量名前加上`$`就可以了。

```bash
$ foo=bar
$ echo $foo
bar
```

每当 Shell 看到以`$`开头的单词时，就会尝试读取这个变量名对应的值。

如果变量不存在，Bash 不会报错，而会输出空字符。

## 删除变量

`unset`命令用来删除一个变量。

```bash
unset NAME
```

这个命令不是很有用。因为不存在的 Bash 变量一律等于空字符串，所以即使`unset`命令删除了变量，还是可以读取这个变量，值为空字符串。

所以，删除一个变量，也可以将这个变量设成空字符串。

```bash
$ foo=''
$ foo=
```

上面两种写法，都是删除了变量`foo`。由于不存在的值默认为空字符串，所以后一种写法可以在等号右边不写任何值。

## 输出变量 export 命令

用户创建的变量仅可用于当前 Shell，子 Shell 默认读取不到父 Shell 定义的变量。为了把变量传递给子 Shell，需要使用`export`命令。**这样输出的变量，对于子 Shell 来说就是环境变量。**

`export`命令用来向子 Shell 输出变量。

```bash
NAME=foo
export NAME
```

上面命令输出了变量`NAME`。变量的赋值和输出也可以在一个步骤中完成。

```bash
export NAME=value
```

上面命令执行后，当前 Shell 及随后新建的子 Shell，都可以读取变量`$NAME`。

子 Shell 如果修改继承的变量，不会影响父 Shell。

```bash
# 输出变量 $foo
$ export foo=bar

# 新建子 Shell
$ bash

# 读取 $foo
$ echo $foo
bar

# 修改继承的变量
$ foo=baz

# 退出子 Shell
$ exit

# 读取 $foo
$ echo $foo
bar
```

上面例子中，子 Shell 修改了继承的变量`$foo`，对父 Shell 没有影响。

## 特殊变量

1. `$?`

   `$?`为上一个命令的退出码，用来判断上一个命令是否执行成功。返回值是`0`，表示上一个命令执行成功；如果是非零，上一个命令执行失败。

2. `$$`

   `$$`为当前 Shell 的进程 ID。

   ```bash
   $ echo $$
   10662
   
   ```

   这个特殊变量可以用来命名临时文件。

   ```bash
   LOGFILE=/tmp/output_log.$$
   ```

# 字符串操作

## 字符串长度

获取字符串长度的语法如下。

```
${#varname}
```

下面是一个例子。

```
$ myPath=/home/cam/book/long.file.name
$ echo ${#myPath}
29
```

## 子字符串

字符串提取子串的语法如下。

```
${varname:offset:length}
```

上面语法的含义是返回变量`$varname`的子字符串，从位置`offset`开始（从`0`开始计算），长度为`length`。

```
$ count=frogfootman
$ echo ${count:4:4}
foot
```

如果省略`length`，则从位置`offset`开始，一直返回到字符串的结尾。

```
$ count=frogfootman
$ echo ${count:4}
footman
```

上面例子是返回变量`count`从4号位置一直到结尾的子字符串。

如果`offset`为负值，表示从字符串的末尾开始算起。注意，负数前面必须有一个空格， 以防止与`${variable:-word}`的变量的设置默认值语法混淆。这时还可以指定`length`，`length`可以是正值，也可以是负值（负值不能超过`offset`的长度）。

```
$ foo="This string is long."
$ echo ${foo: -5}
long.
$ echo ${foo: -5:2}
lo
$ echo ${foo: -5:-2}
lon
```

上面例子中，`offset`为`-5`，表示从倒数第5个字符开始截取，所以返回`long.`。如果指定长度`length`为`2`，则返回`lo`；如果`length`为`-2`，表示要排除从字符串末尾开始的2个字符，所以返回`lon`。

# Bash 启动环境

- `login shell` 登陆时走完整的会话构建流程, 比如 `tty1`~`tty6` 控制终端, 或者 ssh 远程登陆.
- `no login shell` 登陆时不需要走完整的会话构建流程, 比如 在 X11 图形环境下, 打开的终端窗口, 或者是在 Shell 下进入子 Shell 进程.

**两者最大的区别**:

1.  `login shell` 会执行 系统范围 `/etc/profile` 一直到用户环境的 `~/.bash_profile` 等等环境信息.
2.  而`no login shell`并不会执行系统范围的环境初始化流程,仅执行用户环境 `~/.bashrc` 初始化流程. `no login shell` 的系统环境信息是从父进程中集成过来的.

注：比如在`/etc/profile.d`下添加了环境信息, Bash Shell 父进程如果没刷新, 直接进入 Bash Shell 子进程,那子进程也感知到最新环境信息, 确实要刷新的话, 需要手动初始化系统范围的环境信息, 比如执行 `source /etc/profile`或者 `. /etc/profile`.

`source` 和 `.` 符号是等价的.

- `/etc/profile` 系统范围的环境信息初始化, 在新的 `login shell` 构建过程中会激活该环境配置信息
- `/etc/bash.bashrc` 每个交互 Shell 初始化文件
- `/etc/bash.bash.logout` 系统范围`login shell`退出时的环境清理文件
- `~/.bash_profile` 每个 `login shell` 初始化过程,用户环境初始化配置文件.
- `~/.bashrc` 用户环境下交互 Shell 的环境初始化配置文件.
- `~/.bash_logout` `login shell` 退出时执行用户环境清理配置文件
- `~/.inputrc` 用户环境交互原信息配置信息, 比如定义一些交互快捷键

## 登录Session

登录 Session 一般进行整个系统环境的初始化，启动的初始化脚本依次如下。

1. `/etc/profile`：所有用户的全局配置脚本，脚本中会执行`/etc/profile.d`目录里面所有`.sh`文件。

   ```bash
   for i in /etc/profile.d/*.sh /etc/profile.d/sh.local ; do
       if [ -r "$i" ]; then
           if [ "${-#*i}" != "$-" ]; then
               . "$i"
           else
               . "$i" >/dev/null
           fi
       fi
   done
   ```

2. `~/.bash_profile`：用户的个人配置脚本，这个脚本定义了一些最基本的环境变量，然后执行了`~/.bashrc`。(如果`~/.bash_profile`存在，则执行完就不再往下执行。）

   ```bash
   if [ -f ~/.bashrc ]; then
   . ~/.bashrc
   fi
   ```

   - `~/.bash_login`：如果`~/.bash_profile`没找到，则尝试执行这个脚本（C shell 的初始化脚本）。如果该脚本存在，则执行完就不再往下执行。
   - `~/.profile`：如果`~/.bash_profile`和`~/.bash_login`都没找到，则尝试读取这个脚本（Bourne shell 和 Korn shell 的初始化脚本）。

Linux 发行版更新的时候，会更新`/etc`里面的文件，比如`/etc/profile`，因此不要直接修改这个文件。如果想修改所有用户的登陆环境，就在`/etc/profile.d`目录里面新建`.sh`脚本。

## 非登录Session

非登录 Session 的初始化脚本依次如下。

1. `~/.bashrc`：定义当前用户下的环境变量，其中会执行`/etc/bash.bashrc`。

   ```bash
   # .bashrc
   
   # User specific aliases and functions
   
   alias rm='rm -i'
   alias cp='cp -i'
   alias mv='mv -i'
   # Sodded by Anaconda3 4.4.0 installer
   export PATH="/root/anaconda3/bin:$PATH"
   source /root/.bashrcurce global definitions
   
   **if [ -f /etc/bashrc ]; then
   	. /etc/bashrc
   fi**
   ```