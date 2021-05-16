---
title: SSH 互信机制
date: 2021-04-21
tags: 
    - SSH
    - Linux
    - 远程连接
---

## SSH 互信机制

1. 客户端

   ```bash
   $ cd
   $ ssh-keygen -t rsa
   $ cd .ssh/
   $ ls
   ## id_rsa  id_rsa.pub
   ```

2. 服务器

   ```bash
   $ cd .ssh/
   $ vim authorized_keys
   ## 粘贴客户端的id_rsa.pub内容
   ```

<!-- more -->

## 原理

### 1、中间人攻击

SSH之所以能够保证安全，原因在于它采用了公钥加密。

过程：

1. 远程主机收到用户的登录请求，把自己的公钥发给用户。
2. 用户使用这个公钥，将登录密码加密后，发送回来。
3. 远程主机用自己的私钥，解密登录密码，如果密码正确，就同意用户登录。

这个过程本身是安全的，但是实施的时候存在一个风险：如果有人截获了登录请求，然后冒充远程主机，将伪造的公钥发给用户，那么用户很难辨别真伪。因为不像https协议，SSH协议的公钥是没有证书中心（CA）公证的，也就是说，都是自己签发的。

### 2、口令登陆

如果第一次登陆服务器，系统会出现如下提示：

```bash
The authenticity of host 'host (12.18.429.21)' can't be established.
RSA key fingerprint is 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d.
Are you sure you want to continue connecting (yes/no)?
## 无法确认host主机的真实性，只知道它的公钥指纹，还想继续连接吗？
```

所谓"公钥指纹"，是指公钥长度较长（这里采用RSA算法，长达1024位），很难比对，所以对其进行MD5计算，将它变成一个128位的指纹。上例中是98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d，再进行比较，就容易多了。

很自然的一个问题就是，用户怎么知道远程主机的公钥指纹应该是多少？回答是没有好办法，远程主机必须在自己的网站上贴出公钥指纹，以便用户自行核对。

假定经过风险衡量以后，用户决定接受这个远程主机的公钥。系统会出现一句提示，表示host主机已经得到认可。

```bash
Warning: Permanently added 'host,12.18.429.21' (RSA) to the list of known hosts.
```

然后，会要求输入密码。如果密码正确，就可以登录了。

当远程主机的公钥被接受以后，它就会被保存在文件 `$HOME/.ssh/known_hosts` 之中。下次再连接这台主机，系统就会认出它的公钥已经保存在本地了，从而跳过警告部分，直接提示输入密码。

每个SSH用户都有自己的known_hosts文件，此外系统也有一个这样的文件，通常是 `/etc/ssh/ssh_known_hosts`，保存一些对所有用户都可信赖的远程主机的公钥。

### 3、公钥登陆

SSH还提供了公钥登录，可以省去输入密码的步骤。

所谓"公钥登录"，就是用户将自己的公钥储存在远程主机上。登录的时候，远程主机会向用户发送一段随机字符串，用户用自己的私钥加密后，再发回来。远程主机用事先储存的公钥进行解密，如果成功，就证明用户是可信的，直接允许登录shell，不再要求密码。

这种方法要求用户必须提供自己的公钥。如果没有现成的，可以直接用ssh-keygen生成一个：

```bash
$ ssh-keygen
```

运行上面的命令以后，系统会出现一系列提示，可以一路回车。其中有一个问题是，要不要对私钥设置口令（passphrase），如果担心私钥的安全，这里可以设置一个。

运行结束以后，在 `$HOME/.ssh/` 目录下，会新生成两个文件： `id_rsa.pub`和 `id_rsa`。

前者是公钥，后者是私钥。

这时再输入下面的命令，将公钥传送到远程主机host上面：

```bash
$ ssh-copy-id user@host
```

从此再登录，就不需要输入密码了。

如果还是不行，就打开远程主机的/etc/ssh/sshd_config这个文件，检查下面几行前面"#"注释是否取掉。

```bash
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

然后，重启远程主机的ssh服务。

```bash
// ubuntu系统
service ssh restart
　　
// debian系统
/etc/init.d/ssh restart
```

### 4、authorized_keys文件

远程主机将用户的公钥，保存在登录后的用户主目录的 `$HOME/.ssh/authorized_keys`文件中。公钥就是一段字符串，只要把它追加在 `authorized_keys`文件的末尾就行了。

```bash
$ ssh user@host 'mkdir -p .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub
```

这条命令由多个语句组成，依次分解开来看：

（1） `$ ssh user@host`，表示登录远程主机；

（2）单引号中的 `mkdir .ssh && cat >> .ssh/authorized_keys`，表示登录后在远程shell上执行的命令；

（3） `$ mkdir -p .ssh` 的作用是，如果用户主目录中的.ssh目录不存在，就创建一个；

（4） `cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub`的作用是，将本地的公钥文件 `~/.ssh/id_rsa.pub`，重定向追加到远程文件 `authorized_keys`的末尾。

写入 `authorized_keys`文件后，公钥登录的设置就完成了。