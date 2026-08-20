# .claude/skills/ 内容来源说明

本目录下所有 skill 均已**直接提交进本仓库**（非 submodule、非插件安装），
clone 后无需任何安装步骤即可使用。

---

## 1. education-agent-skills（165 个）

教育学 / 教学法 skill，扁平铺在本目录下。

- 来源: https://github.com/GarethManning/education-agent-skills
- 版本: v2.1.0
- 同步自 commit: 4be2795b574e91bdcbb6bda01ab235a05cfadbcc (2026-08-10)
- 许可: CC BY-SA 4.0（作者 Gareth Manning）

**布局说明**：上游按 `skills/<分类>/<技能名>/SKILL.md` 两层组织；本仓库为符合
`.claude/skills/` 的「每个直接子目录一个 skill」约定，已扁平化为
`.claude/skills/<技能名>/SKILL.md`。165 个技能名两两不重复，且目录名与
SKILL.md frontmatter 的 `name` 字段一致，扁平化未修改任何文件内容。
原分类信息保留在每个 SKILL.md 的 `domain:` / `skill_id:` 字段中。

**更新方式**：

    git clone --depth 1 https://github.com/GarethManning/education-agent-skills /tmp/eas
    for d in /tmp/eas/skills/*/*/; do rm -rf ".claude/skills/$(basename $d)"; done
    for d in /tmp/eas/skills/*/*/; do cp -r "$d" ".claude/skills/$(basename $d)"; done
    claude plugin validate .claude

---

## 2. KernelWiki

Blackwell / Hopper GPU kernel 优化知识库（3520 个文件，约 22.5 MB）。

- 来源: https://github.com/mit-han-lab/KernelWiki
- 同步自 commit: 2777d18ffb3a3d682d8f25a3e3b8864d925a5ff1 (2026-06-09)
- 许可: 上游仓库未提供 LICENSE 文件

## 3. ncu-report-skill

Nsight Compute 性能分析 skill（22 个文件，约 0.21 MB）。

- 来源: https://github.com/mit-han-lab/ncu-report-skill
- 同步自 commit: 1cf238d6b41c79bd35041192506c4d45e765a3f1 (2026-05-24)
- 许可: 上游仓库未提供 LICENSE 文件

### 关于 2 和 3 的历史说明

这两个目录原先是嵌套 git 仓库，在父仓库索引里表现为 gitlink（mode 160000），
但 `.gitmodules` 中并无对应条目——`git submodule status` 会直接报
`fatal: no submodule mapping found`。后果是**别人 clone 本仓库时这两个目录是空的**。
现已改为普通文件直接提交，问题消除。

被排除的唯一文件是 `KernelWiki/scripts/__pycache__/*.pyc`（编译产物，
上游 .gitignore 本就排除）。其余 3520 / 22 个文件与上游 tracked 列表逐一核对一致。

**更新方式**（注意：目录内已无 .git，需重新拉取后覆盖）：

    git clone --depth 1 https://github.com/mit-han-lab/KernelWiki /tmp/kw
    rm -rf /tmp/kw/.git && rm -rf .claude/skills/KernelWiki
    cp -r /tmp/kw .claude/skills/KernelWiki

ncu-report-skill 同理。更新后请回来修改本文件记录的 commit hash。
