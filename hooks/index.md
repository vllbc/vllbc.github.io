# Claude Code Hooks 笔记


> 整理自 [Claude Code 官方 Hooks 参考](https://code.claude.com/docs/zh-CN/hooks)。原文是逐字段罗列的字典式文档，这里压缩成"什么时候用什么、怎么决定"的速查笔记。第一次配置 hook 前推荐先看 [使用 hooks 自动化工作流](/docs/zh-CN/hooks-guide) 指南。

# Claude Code Hooks 笔记

Hooks 是挂在 Claude Code 生命周期事件上的回调：可以跑 shell 命令、HTTP 请求、调用 MCP 工具、或让 LLM 帮判断。本质是"事件 → 过滤 → 副作用/决定"的三段式管道。

按触发频率分三档：

- **每个会话一次**：`SessionStart`、`SessionEnd`、`Setup`、`InstructionsLoaded`
- **每轮对话一次**：`UserPromptSubmit`、`UserPromptExpansion`、`Stop`、`StopFailure`、`PreCompact`、`PostCompact`
- **代理循环内每个工具调用**：`PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`PostToolBatch`、`PermissionRequest`、`PermissionDenied`、`SubagentStart`、`SubagentStop`、`TaskCreated`、`TaskCompleted`、`Notification`、`CwdChanged`、`FileChanged`、`ConfigChange`、`WorktreeCreate`、`WorktreeRemove`、`MessageDisplay`、`Elicitation`、`ElicitationResult`

另外有些事件是异步独立触发：`TeammateIdle`（agent team 队友闲置）、`ConfigChange`、`CwdChanged`、`FileChanged`、`WorktreeCreate/Remove`、`Notification`。

## 生命周期全景

![Hook 生命周期](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/hooks-lifecycle.svg)

上图里几个要点：

- `Setup`（可选）→ `SessionStart` → 每轮循环（`UserPromptSubmit` → Agentic Loop → `Stop` / `StopFailure`）→ `SessionEnd`
- Agentic Loop 里嵌套 `PreToolUse` → `PermissionRequest` → 工具执行 → `PostToolUse` / `PostToolUseFailure` → `PostToolBatch`
- Subagent 循环是平行的：`SubagentStart` / `SubagentStop`、`TaskCreated` / `TaskCompleted`
- 灰色虚线的事件（`Notification`、`ConfigChange`、`WorktreeCreate` 等）是异步副作用，不在主路径上
- `Elicitation` / `ElicitationResult` 嵌在 MCP 工具执行里
- `MessageDisplay` 是只读显示事件，不影响 Claude 行为

## 解析流程

![Hook 解析流程](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/hook-resolution.svg)

一个 hook 要真正跑起来，三层过滤缺一不可：

1. **事件触发**：Claude Code 决定发事件（如要执行 Bash）
2. **匹配器检查**：`matcher` 字段过滤事件（如只对 `Bash` 工具）
3. **if 条件检查**：`if` 用权限规则语法再过滤一次（如 `Bash(rm *)`）
4. **执行 handler**：从 stdin 收 JSON，决定如何回应

下面这个例子同时演示了三层结构 + 通过 JSON `permissionDecision: "deny"` 阻止调用：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh"
          }
        ]
      }
    ]
  }
}
```

handler 通过退出码和 stdout 表达决定（详见 [决定控制](#决定控制)）。

## 配置位置

| 位置 | 范围 | 可共享 |
|---|---|---|
| `~/.claude/settings.json` | 所有项目 | 否（用户级） |
| `.claude/settings.json` | 单个项目 | 是（提交到仓库） |
| `.claude/settings.local.json` | 单个项目 | 否（gitignored） |
| 托管策略设置 | 组织 | 是（管理员控制） |
| Plugin `hooks/hooks.json` | 启用插件时 | 是 |
| Skill / Subagent frontmatter | 组件活跃时 | 是 |

临时禁用所有 hook：在任意 settings 文件里设 `"disableAllHooks": true`。托管策略里的 hook 不受用户/项目设置里的 `disableAllHooks` 影响——只有托管级才能禁托管 hook。`/hooks` 菜单可读不能改。

## Handler 类型

配置里 `hooks` 数组每个对象是一个 handler，有五种类型：

| `type` | 行为 | 怎么收输入 | 怎么传结果 |
|---|---|---|---|
| `command` | 跑 shell 命令 | stdin 收 JSON | 退出码 + stdout/stderr |
| `http` | POST 到 URL | 请求体收 JSON | 状态码 + 响应体 |
| `mcp_tool` | 调已连接 MCP 服务器的工具 | 按工具 schema | 工具文本输出 |
| `prompt` | 送 LLM 单轮评估 | 提示里 `$ARGUMENTS` 占位 | LLM 返 `{ok, reason}` |
| `agent` | 起 subagent 验证（实验性） | 提示里 `$ARGUMENTS` 占位 | LLM 返 `{ok, reason}` |

并非所有事件都支持所有类型。`SessionStart` / `Setup` 只支持 `command` 和 `mcp_tool`。`prompt` 和 `agent` 只能在能"做决定"的事件上用（`PreToolUse`、`PostToolUse`、`Stop`、`UserPromptSubmit` 等），不能在纯副作用事件上用。

### 通用字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `type` | 是 | 上面五种之一 |
| `if` | 否 | 权限规则语法过滤，如 `Bash(git *)`、`Edit(*.ts)`。只在工具事件上评估 |
| `timeout` | 否 | 秒数。默认 `command`/`http`/`mcp_tool`=600，prompt=30，agent=60 |
| `statusMessage` | 否 | hook 跑时显示的加载提示 |
| `once` | 否 | 每会话只跑一次（仅在 skill frontmatter 的 hook 里生效） |

### 命令 hook 的 exec vs shell 形式

设了 `args` 走 **exec 形式**：`command` 解析为可执行文件，`args` 是参数向量，无 shell 介入。引号、空格、`$` 都原样传递。**路径占位符必须用 exec 形式**（`${CLAUDE_PROJECT_DIR}` 等不会被 shell 破坏）。

```json
{ "type": "command", "command": "node",
  "args": ["${CLAUDE_PROJECT_DIR}/scripts/format.js", "--fix"] }
```

没设 `args` 走 **shell 形式**：`command` 整体交给 `sh -c` 解释，能用管道、`&&`、变量展开。

```json
{ "type": "command",
  "command": "node \"${CLAUDE_PROJECT_DIR}\"/scripts/format.js --fix" }
```

Windows 上：exec 形式需要真实 `.exe`，npm 垫片（`.cmd`/`.bat`）必须走 shell 形式或 `node` 直调。设 `"shell": "powershell"` 可在 PowerShell 里跑单条 hook。

### 路径占位符

- `${CLAUDE_PROJECT_DIR}`：项目根
- `${CLAUDE_PLUGIN_ROOT}`：插件安装目录（每次更新会变）
- `${CLAUDE_PLUGIN_DATA}`：插件持久数据目录（更新后保留）

## 匹配器（matcher）

不同事件匹配不同字段：

- **工具事件**（`PreToolUse`/`PostToolUse`/`PostToolUseFailure`/`PermissionRequest`/`PermissionDenied`）：匹配 `tool_name`
- **`SessionStart`**：匹配 `source`（`startup`/`resume`/`clear`/`compact`）
- **`SessionEnd`**：匹配退出原因（`clear`/`resume`/`logout`/`prompt_input_exit`/`bypass_permissions_disabled`/`other`）
- **`Notification`**：匹配通知类型（`permission_prompt`/`idle_prompt`/`auth_success`/`elicitation_*`）
- **`SubagentStart`/`SubagentStop`**：匹配代理类型（`general-purpose`/`Explore`/`Plan`/自定义名）
- **`PreCompact`/`PostCompact`**：匹配 `manual`/`auto`
- **`ConfigChange`**：匹配 `user_settings`/`project_settings`/`local_settings`/`policy_settings`/`skills`
- **`FileChanged`**：字面文件名（用 `|` 分隔，如 `.envrc|.env`），不遵循下面的通用规则
- **`StopFailure`**：匹配错误类型（`rate_limit`/`authentication_failed`/`billing_error` 等）
- **`InstructionsLoaded`**：匹配加载原因（`session_start`/`nested_traversal`/`compact` 等）
- **`UserPromptExpansion`**：匹配命令名
- **`Elicitation`/`ElicitationResult`**：匹配 MCP 服务器名

**不支持 matcher、永远全触发**：`UserPromptSubmit`、`PostToolBatch`、`Stop`、`TeammateIdle`、`TaskCreated`、`TaskCompleted`、`WorktreeCreate`、`WorktreeRemove`、`MessageDisplay`、`CwdChanged`。给这些事件加 `matcher` 字段会被静默忽略。

匹配器语法：

- `"*"`、`""` 或省略：匹配所有
- 只有字母/数字/`_`/`|`：精确字符串或 `|` 列表（`Bash`、`Edit|Write`）
- 含其他字符：JS 正则（`^Notebook`、`mcp__memory__.*`）

MCP 工具命名是 `mcp__<server>__<tool>`，匹配整个 server 用 `mcp__<server>__.*`（末尾的 `.*` 必须有，否则按字面精确匹配）。

`if` 字段在同一 handler 上做更窄的二次过滤，权限规则语法：`Bash(rm *)`、`Edit(*.ts)`。注意只支持单条规则，不支持 `&&`/`||`，要多个条件就写多个 handler。

## 事件清单（按生命周期）

| 事件 | 何时触发 | 典型用途 |
|---|---|---|
| `SessionStart` | 会话开始/恢复/clear/compact | 加载上下文、设环境变量、注入项目状态 |
| `Setup` | `--init-only` / `--init` / `--maintenance` | CI 一次性准备 |
| `InstructionsLoaded` | CLAUDE.md 或 `.claude/rules/*.md` 加载 | 跟踪记忆文件被加载 |
| `UserPromptSubmit` | 用户提交 prompt，Claude 处理前 | 注入上下文、过滤敏感词 |
| `UserPromptExpansion` | slash command 展开为 prompt 前 | 拦截 / 改写 / 阻止命令 |
| `PreToolUse` | 工具执行前 | 校验输入、阻止危险命令、改写参数 |
| `PermissionRequest` | 权限弹窗出现时 | 自动批准 / 拒绝 |
| `PermissionDenied` | 自动模式分类器拒绝 | 记录、告诉模型可重试 |
| `PostToolUse` | 工具成功后 | 检查结果、改写输出、加上下文 |
| `PostToolUseFailure` | 工具失败后 | 告警、向 Claude 反馈错误上下文 |
| `PostToolBatch` | 一批并行工具全部完成后、下次模型调用前 | 基于整批结果注入上下文 |
| `SubagentStart` / `SubagentStop` | subagent 启动 / 结束 | 计时、注入子代理上下文 |
| `TaskCreated` / `TaskCompleted` | 任务创建 / 标记完成 | 跟踪任务状态 |
| `Stop` | Claude 自然完成回复 | 强制继续、追加任务 |
| `StopFailure` | API 错误导致轮次结束 | 告警 |
| `TeammateIdle` | agent team 队友要闲置 | 阻止闲置、让其继续 |
| `Notification` | 通知发送 | 桌面通知、提醒 |
| `MessageDisplay` | 助手消息文本流式显示 | 改写展示文本（不改变 Claude 看到的内容） |
| `CwdChanged` | 工作目录变化 | direnv 类的响应式环境 |
| `FileChanged` | 被监视的文件磁盘上变化 | 文件变更触发后续动作 |
| `ConfigChange` | 配置文件变化 | 跟踪配置 |
| `WorktreeCreate` | `--worktree` 或 `isolation: "worktree"` | 自定义 worktree 路径 |
| `WorktreeRemove` | worktree 移除 | 清理 |
| `PreCompact` | 压缩前 | 阻止压缩 |
| `PostCompact` | 压缩后 | 注入压缩后上下文 |
| `SessionEnd` | 会话终止 | 清理 |
| `Elicitation` | MCP 服务器在工具调用中请求用户输入 | 编程方式响应、跳过对话框 |
| `ElicitationResult` | 用户回应 MCP 询问后 | 观察 / 改写 / 阻止响应 |

## 输入输出

### 通用输入字段

每条 hook 通过 stdin（命令）/ POST body（HTTP）/ 工具参数（MCP）收到 JSON：

```json
{
  "session_id": "abc123",
  "transcript_path": "/.../.claude/projects/.../transcript.jsonl",
  "cwd": "/home/user/project",
  "permission_mode": "default",
  "effort": { "level": "high" },
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": { "command": "npm test" }
}
```

`--agent` 启动或 subagent 内触发还会多 `agent_id`、`agent_type`。`SessionStart` 还会多 `model` 字段（不是环境变量）。

`tool_name` / `tool_input` 是事件特定字段，每个事件能拿到什么见 [原文事件章节](https://code.claude.com/docs/zh-CN/hooks#hook-events)。

### 退出码语义

- **0**：成功。stdout 被解析为 JSON（可选），除 `UserPromptSubmit` / `UserPromptExpansion` / `SessionStart` 外，stdout 不进对话
- **2**：阻止错误。stderr 进 Claude 上下文作为错误信息，stdout 被忽略
- **其他**：非阻止错误。`WorktreeCreate` 例外——任何非零都会中止

### 退出 2 能阻止什么

| 事件 | 退出 2 效果 |
|---|---|
| `PreToolUse` | 阻止工具调用 |
| `PermissionRequest` | 拒绝权限 |
| `UserPromptSubmit` | 阻止并从上下文删 prompt |
| `UserPromptExpansion` | 阻止命令展开 |
| `Stop` / `SubagentStop` | 阻止停止，继续对话 |
| `TeammateIdle` | 阻止队友闲置 |
| `TaskCreated` | 回滚任务创建 |
| `TaskCompleted` | 阻止标完成 |
| `ConfigChange` | 阻止配置生效（policy_settings 除外） |
| `PostToolBatch` | 在下次模型调用前停代理循环 |
| `PreCompact` | 阻止压缩 |
| `Elicitation` | 拒绝询问 |
| `ElicitationResult` | 操作变为 decline |

**已发生的事件不能阻止**：`PostToolUse` / `PostToolUseFailure` / `Notification` / `SessionStart` / `Setup` / `SessionEnd` / `CwdChanged` / `FileChanged` / `PostCompact` / `InstructionsLoaded` / `MessageDisplay` / `WorktreeRemove` / `StopFailure` —— 退出 2 只会让 stderr 展示给用户。`PermissionDenied` 不能用退出码阻止，但能用 JSON `retry: true` 告诉模型"可以再试一次"。

### JSON 输出（精细控制）

退出 0 时往 stdout 打 JSON，得到比退出码更细的语义。**只支持一种模式**：要么只用退出码，要么用 JSON——不能混。stdout 必须只含 JSON，否则 shell 启动输出会破坏解析。

通用字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `continue` | `true` | `false` 强制停止 Claude |
| `stopReason` | - | `continue: false` 时给用户看的消息 |
| `suppressOutput` | `false` | 隐藏 stdout |
| `systemMessage` | - | 给用户看的警告 |
| `terminalSequence` | - | 让 Claude Code 发出 OSC 转义（桌面通知、窗口标题、BEL）。受限白名单（OSC 0/1/2/9/99/777），不能写光标控制或颜色 |

为 Claude 注入上下文（在 `hookSpecificOutput` 里）：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "这个文件是生成的，请改 src/schema.ts 然后跑 bun generate"
  }
}
```

`additionalContext` 出现在 `SessionStart`/`UserPromptSubmit`/`PreToolUse` 等不同位置——本质是"Claude 看到的下一个模型请求时多了一条系统提醒"。多个 hook 都返回时 Claude 收到全部。超 10000 字符会被截断到文件并附带预览。

CLAUDE.md 比 `additionalContext` 更适合放不变的项目说明——它加载时不需要跑脚本。

## 决定控制

每个事件表达"决定"的方式不同，下面是速查表：

| 事件 | 决定方式 | 关键字段 |
|---|---|---|
| `UserPromptSubmit` / `UserPromptExpansion` / `PostToolUse` / `PostToolUseFailure` / `PostToolBatch` / `Stop` / `SubagentStop` / `ConfigChange` / `PreCompact` | 顶级 `decision: "block"` | `reason` |
| `TeammateIdle` / `TaskCreated` / `TaskCompleted` | 退出 2 或 `continue: false` | - |
| `PreToolUse` | `hookSpecificOutput.permissionDecision` | `allow` / `deny` / `ask` / `defer` + `permissionDecisionReason` |
| `PermissionRequest` | `hookSpecificOutput.decision.behavior` | `allow` / `deny`；allow 时还能 `updatedInput` 改工具参数 |
| `PermissionDenied` | `hookSpecificOutput.retry: true` | 让模型重试被拒的工具 |
| `WorktreeCreate` | 命令 hook stdout 打印路径 / HTTP `hookSpecificOutput.worktreePath` | - |
| `Elicitation` / `ElicitationResult` | `hookSpecificOutput.action` | `accept` / `decline` / `cancel` + `content` |
| `MessageDisplay` | `hookSpecificOutput.displayContent` | 改屏幕上显示的文本（Claude 看到的还是原文） |
| `SessionStart` / `Setup` / `SubagentStart` | 仅 `additionalContext` | 无阻止能力（`SessionStart` 还接受 `initialUserMessage` / `watchPaths` / `sessionTitle` / `reloadSkills`） |
| `WorktreeRemove` / `Notification` / `SessionEnd` / `PostCompact` / `InstructionsLoaded` / `StopFailure` / `CwdChanged` / `FileChanged` | 无 | 只能做副作用（日志、清理） |

PreToolUse 的 deferred 模式：返回 `permissionDecision: "defer"` 配合 `toolUseID`，可以把工具调用延迟到下一轮——典型场景是用户审批 UI 弹窗没准备好时。

PreToolUse 改写参数：

```json
{ "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "updatedInput": { "command": "npm run lint" } } }
```

PostToolUse 改写输出：

```json
{ "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "updatedToolOutput": { "stdout": "[redacted]", "stderr": "",
                            "interrupted": false, "isImage": false } } }
```

注意：`updatedToolOutput` 只能改 Claude 看到的内容，工具已执行的副作用（写文件、跑命令、HTTP 请求）都已经发生。遥测也在 hook 前抓完。

## 异步 Hook

命令 hook 默认阻塞。加 `"async": true` 后台跑，Claude 不等结果。适合"测试、部署、外部 API 回调"这类长操作。

```json
{ "type": "command", "command": "/path/to/run-tests.sh",
  "async": true, "timeout": 120 }
```

异步 hook **不能阻止**、**不能返回决定**（操作已经走完了）。它能做的：返回 `additionalContext` 在下一轮传给 Claude。`systemMessage` 给用户看。`asyncRewake: true` 让它在退出 2 时即使会话闲置也立刻唤醒 Claude。

限制：

- 只 `type: "command"` 支持
- 不支持 `prompt` / `agent` 异步
- 每次执行独立后台进程，不去重
- 输出默认被抑制（`Ctrl+O` 开详细模式）

## 基于提示的 Hook

`type: "prompt"`，不跑命令，送一个 LLM 单轮评估决定：

```json
{ "type": "prompt",
  "prompt": "评估 Claude 是否应停止: $ARGUMENTS。检查任务是否完成。" }
```

LLM 返回 `{ "ok": true }` 或 `{ "ok": false, "reason": "..." }`。`$ARGUMENTS` 会被 hook 输入 JSON 替换。

- `ok: true` 允许
- `ok: false`：
  - `Stop` / `SubagentStop` / `PreToolUse` 等：原因作为下一条指令或工具错误回给 Claude
  - `PostToolUse`：默认结束轮次；设 `continueOnBlock: true` 把原因作为上下文
  - `PostToolBatch` / `UserPromptSubmit` / `UserPromptExpansion`：结束轮次

`PermissionRequest` 和 `PermissionDenied` 上 `ok: false` 无效，要用命令 hook 返回 `decision.behavior: "deny"` 或 `retry: true`。

## 基于代理的 Hook

`type: "agent"`（实验性，行为可能变）。和 prompt hook 类似，但代理可以调 Read/Grep/Glob 工具来验证条件，50 轮内返回。配置字段和 prompt 一样，只是默认 `timeout=60`。**生产环境建议用命令 hook**。

## 安全

命令 hook 以你的系统用户完整权限跑——能改/删任何文件。

最佳实践：

- 永远不信任 hook 输入数据
- shell 变量要引号 (`"$VAR"` 不是 `$VAR`)
- 检查 `..` 路径遍历
- 用绝对路径（`${CLAUDE_PROJECT_DIR}` + exec 形式最稳）
- 跳过 `.env`、`.git/`、密钥目录

企业管理员可以设 `allowManagedHooksOnly` 禁止用户/项目/插件 hook——只允许托管设置里的 hook 跑。托管 `enabledPlugins` 强制启用的插件里的 hook 是豁免的。

## 调试

- `/hooks` 菜单：只读浏览所有配置的 hook
- `claude --debug`：把 hook 匹配、退出码、stdout/stderr 写到 `~/.claude/debug/<session-id>.txt`，不打印终端
- `claude --debug-file <path>`：日志写到指定位置
- `CLAUDE_CODE_DEBUG_LOG_LEVEL=verbose`：额外显示匹配器计数和查询细节

典型调试场景：

- hook 不触发 → 检查 `matcher` 字段拼写和事件支持列表
- Stop hook 持续阻止 → 确认退出 0 / JSON 模式二选一
- JSON 解析失败 → 检查 shell rc 文件（`.bashrc` 等）是否在 hook stdout 里打印东西

## 速查

```text
事件 → matcher → [if] → handler
                 ↓
         stdin/POST 收 JSON
                 ↓
         退出 0 (stdout JSON) | 退出 2 (阻止) | 其他 (非阻止)
                 ↓
         stdout JSON 含 decision / permissionDecision / additionalContext / continue
```

参考资料：

- [Hooks 官方参考](https://code.claude.com/docs/zh-CN/hooks)
- [使用 hooks 自动化工作流（指南）](https://code.claude.com/docs/zh-CN/hooks-guide)
- [设置文件](https://code.claude.com/docs/zh-CN/settings)
- [权限规则](https://code.claude.com/docs/zh-CN/permissions)
- [Bash 命令验证器示例](https://github.com/anthropics/claude-code/blob/main/examples/hooks/bash_command_validator_example.py)

