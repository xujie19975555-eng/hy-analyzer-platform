# GitHub Branch Protection 设置指南

## 目的
确保 main 分支只能通过 PR 合并，且 CI 必须通过。

## 设置步骤

1. 打开 GitHub 仓库: https://github.com/xujie19975555-eng/hy-analyzer-platform

2. 点击 **Settings** (设置)

3. 左侧菜单选择 **Branches**

4. 在 "Branch protection rules" 下点击 **Add rule**

5. 配置如下:

### Branch name pattern
```
main
```

### 勾选以下选项:

- [x] **Require a pull request before merging**
  - [x] Require approvals (设为 0 或 1)

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - 搜索并添加: `test` (这是 CI workflow 的 job 名)

- [ ] **Do not require reviews** (个人项目可以不要求 review)

- [x] **Do not allow bypassing the above settings**

6. 点击 **Create** 保存

## 验证

设置完成后:
- 直接 push 到 main 会被拒绝
- 必须通过 PR
- PR 必须等 CI 通过才能合并

## 注意

如果你是唯一的开发者，可以不勾选 "Require approvals"，这样自己可以合并自己的 PR。
