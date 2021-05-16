---
title: Java 安装配置
date: 2021-03-24
tags: 
    - Java
    - macOS
categories: 配置
---

# Java 安装配置

## 一、Mac 自带JRE

终端输入

<!-- more -->

```python
/usr/libexec/java_home -V

'''
Matching Java Virtual Machines (2):
    1.8.281.09 (x86_64) "Oracle Corporation" - "Java" /Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home
    1.8.0_281 (x86_64) "Oracle Corporation" - "Java SE 8" /Library/Java/JavaVirtualMachines/jdk1.8.0_281.jdk/Contents/Home
/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home
'''
```

 macOS 自带 JRE，而不是 JDK

选择目录 `/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home`，显示

![Java%20%E5%AE%89%E8%A3%85%E9%85%8D%E7%BD%AE%2057424c6814d946eaa916ccddd8bcdb9a/Untitled.png](https://i.loli.net/2021/05/16/49kSeP3BtL8qKVj.png)

## 二、安装 JDK 8

1. 下载安装包安装

2. 添加环境变量 

   在 `.zshrc` 文件中

   ```bash
   export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_281.jdk/Contents/Home
   ```

   ```python
   fan@MacBook-Pro  ~  java -version
   java version "1.8.0_281"
   Java(TM) SE Runtime Environment (build 1.8.0_281-b09)
   Java HotSpot(TM) 64-Bit Server VM (build 25.281-b09, mixed mode)
   ```