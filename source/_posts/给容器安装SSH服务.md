---
title: 给容器安装 SSH 服务
date: 2021-01-16
tags: 
    - Docker
    - SSH
    - 远程连接
categories: 配置
---

# 给容器安装 SSH 服务

## 启动 Container

```bash
docker exec -it [containerID] /bin/bash
```

## 更新源

```bash
apt-get update
```
<!-- more -->

## 安装SSH服务

```bash
apt-get install openssh-server
```

## 启动SSH服务

```bash
service ssh satrt
```

## 安装Vim

```bash
apt-get install vim
```

## 更改SSH服务配置

`vim /etc/ssh/sshd_config`

将PermitRootLogin的值从withoutPassword改为yes，允许root用户进行登录

```bash
PermitRootLogin yes
```

## 设置root用户登陆密码

```bash
passwd root
```

## 重启SSH服务

```bash
service ssh restart
```

## 退出当前容器

```bash
exit
```

## 保存镜像到本地

```bash
docker commit [containerID] [imagename]:[version]
```

## 重启镜像

```bash
docker run -it -d -p 8400:20 -p 8855:22 --name [containername] [imagename]:[tag]
```

## 让容器保持运行

```bash
docker update --restart=always <containerID>
```