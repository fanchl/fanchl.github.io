---
title: Maven 安装配置
date: 2021-03-24
tags: 
    - Java
    - macOS
categories: 配置
---



## 一、下载 Maven

首先从 Maven 官方地址：[http://maven.apache.org/download.cgi](http://maven.apache.org/download.cgi) 下载最新版本apache-maven-xxx-bin.tar.gz

加下来将下载的文件解压到 /usr/local/maven 下。

<!-- more -->

## 二、配置环境变量

编辑 `.zshrc`

添加如下的 maven 配置：

```bash
export M3_HOME=/usr/local/maven/apache-maven-3.6.3
export PATH=$M3_HOME/bin:$PATH
```

执行命令 `source .zshrc`

## 三、测试是否安装成功

```bash
mvn -version
```

## 四、配置本地仓库与镜像

在安装的 maven 文件夹中，找到 `conf/settings.xml` 

**添加本地仓库位置**

```xml
<localRepository>/usr/local/maven/apache-maven-3.8.1/repository</localRepository>
```

**添加镜像**

```xml
<mirror>  
   <id>aliyun-maven</id>  
   <name>aliyun maven</name>  
   <url>http://maven.aliyun.com/nexus/content/groups/public/</url>  
   <mirrorOf>*</mirrorOf>          
</mirror>

```