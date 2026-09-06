# GitHub 上传说明

建议上传两个私有仓库，避免提交版和学习版互相污染。

## 版本一：提交版

用途：交作业和实验复核。只包含可复现代码、部署脚本、评测脚本、prompt、正式结果 log、提交版报告和 README。

本机准备目录：

```text
/root/alfworld-agent-submission
```

上传命令：

```bash
cd /root/alfworld-agent-submission
git init
git add -A
git commit -m "Prepare ALFWorld agent submission"
git branch -M main
git remote add origin git@github.com:<your-name>/alfworld-agent.git
git push -u origin main
```

如果已经在 GitHub 网页端创建了仓库，把 `<your-name>` 换成自己的 GitHub 用户名即可。

如果已经存在 origin，使用：

```bash
git remote set-url origin git@github.com:<your-name>/alfworld-agent.git
git push -u origin main
```

## 版本二：学习全量版

用途：自己复盘和继续开发。保留当前完整工作区，包括学习版文档、测试、探索材料和额外记录。

上传命令：

```bash
cd /root/alfworld-agent
git init
git add -A
git commit -m "Complete ALFWorld agent workspace"
git branch -M main
git remote add origin git@github.com:<your-name>/alfworld-agent-learning.git
git push -u origin main
```

如果已经存在 origin，使用：

```bash
git remote set-url origin git@github.com:<your-name>/alfworld-agent-learning.git
git push -u origin main
```

## 使用 GitHub CLI 的替代命令

如果服务器已经登录 GitHub CLI：

```bash
cd /root/alfworld-agent-submission
gh repo create alfworld-agent --private --source=. --remote=origin --push

cd /root/alfworld-agent
gh repo create alfworld-agent-learning --private --source=. --remote=origin --push
```

## 上传前检查

提交版检查：

```bash
cd /root/alfworld-agent-submission
git status --short
git ls-files | grep -E '(__pycache__|\.pyc$|task3_smoke|task4_smoke|reference_baseline_probe|learning_notes)' || true
```

学习全量版检查：

```bash
cd /root/alfworld-agent
git status --short
find . -type d -name __pycache__ -print
find . -name '*.pyc' -print
```

不需要上传的内容：模型权重、conda 环境目录、ALFWorld 数据缓存、smoke test 输出、`__pycache__`、`.pyc`、无意义临时日志。
