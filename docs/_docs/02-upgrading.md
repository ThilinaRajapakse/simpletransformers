---
title: Upgrading
permalink: /docs/upgrading/
excerpt: "Instructions for upgrading the library."
last_modified_at: 2020-05-02 17:57:14
toc: true
---

Simple Transformers is updated regularly and using the latest version is highly recommended. This will ensure that you have access to the latest features, improvements, and bug fixes.

## Check current version

To check your current version with pip, you can do;

```shell
pip show simpletransformers
```

As Simple Transformers is built on top of the Hugging Face Transformers library, make sure that you are using the latest Transformers release.

```shell
pip show transformers
```

## Update to latest version

You can update a pip package with the following command.

```shell
pip install --upgrade simpletransformers
```

This should upgrade the Transformers package to the required version as well. However, you can also update Transformers manually via;

```shell
pip install --upgrade transformers
```
