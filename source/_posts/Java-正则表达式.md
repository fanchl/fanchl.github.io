---
title: Java 正则表达式
date: 2021-05-14 10:50:33
tags:
---

## 一、捕获组的概念

捕获组可以通过从左到右计算其开括号来编号，**编号从 1 开始**。例如，在表达式 ((A)(B(C))) 中，存在四个组：

```java
((A)(B(C)))
(A)
(B(C))
(C)
```

组 0 代表整个表达式

<!-- more -->

## 二、详解 Pattern 类与 Matcher 类

Java 正则表达式通过 `java.util.regex` 包下的 Pattern 类与 Matcher 类实现。

Pattern 类用于创建一个正则表达式，创建一个 pattern，构造方法是私有的，不能够直接创建，需要通过 `Pattern.compile(String regex)` 创建一个正则表达式。

```java
Pattern p=Pattern.compile("\\w+"); 
p.pattern();//返回 \w+
```

pattern() 返回正则表达式的字符串形式，其实就是返回 Pattern.complile(String regex) 的 regex 参数。

### `Pattern.matcher(CharSequence input)`

Pattern.matcher(CharSequence input)返回一个Matcher对象。

Matcher 类的构造方法是私有的，不能够随意创建，只能通过该方法获取该类的实例。

```java
Pattern p=Pattern.compile("\\d+"); 
Matcher m=p.matcher("22bb23"); 
m.pattern();//返回p 也就是返回该Matcher对象是由哪个Pattern对象的创建的
```

### Mathcer.start()/ Matcher.end()/ Matcher.group()

```java
Pattern p=Pattern.compile("([a-z]+)(\\d+)"); 
Matcher m=p.matcher("aaa2223bb"); 
**m.find();   //匹配aaa2223** 
m.groupCount();   //返回2,因为有2组 
m.start(1);   //返回0 返回第一组匹配到的子字符串在字符串中的索引号 
m.start(2);   //返回3 
m.end(1);   //返回3 返回第一组匹配到的子字符串的最后一个字符在字符串中的索引位置. 
m.end(2);   //返回7 
m.group(1);   //返回aaa,返回第一组匹配到的子字符串 
m.group(2);   //返回2223,返回第二组匹配到的子字符串
```