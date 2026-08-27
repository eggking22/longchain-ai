# MCQ Step 2 — 中文翻译层 + Review 审核前端

在 Step 1（Statement Draft）之上新增两层：确定性中文翻译层（无 LLM）与人工审核
Web 前端。英文草稿 `question_drafts.json` 保持只读，翻译结果写入独立产物
`mcq_drafts_zh.json`。

## 1. 中文翻译层（app/services/question_translation/）

原则：**英文原文逐字保留**，`statement_zh` 是新增字段；数值、基因/试剂名、
TRUE/FALSE 标记、evidence_ids 一律不翻译。

| 模块 | 职责 |
| ---- | ---- |
| `terminology.py` | 术语表 `TERMINOLOGY`（increases→提高、inhibitor→抑制剂、body weight→体重…），最长优先 + 忽略大小写替换；`_DROP_WORDS` 仅匹配全小写冠词（保留 "Treatment A" 中的 "A"） |
| `translator.py` | 有序模板 `_TEMPLATES`（According-to 前置优先）+ 动词映射 + `_glue()` 拉丁/中文边界补空格；返回 `(中文, method)` |
| `pipeline.py` | `translate_drafts` / `translate_document` / `persist_mcq_zh`（读 `question_drafts.json` → 写 `mcq_drafts_zh.json`） |

翻译方法两级：`template`（模板命中，paper_1 上 94 条）/ `term_fallback`
（仅术语替换，11 条）。产物零时间戳、字节可复现。

```bash
python scripts/generate_mcq_zh.py --doc-id paper_1
# data/paper_semantics/paper_1/mcq_drafts_zh.json
```

## 2. Review 审核前端（static/review.html，vanilla JS）

入口：启动服务后访问 `/review`（无前端框架，复用 FastAPI static 挂载）。

- 左侧 Figure 导航（状态徽标 SUFFICIENT/PARTIAL + 题组数）
- 题组卡片：实验摘要（组别/指标/方向）→ A–E 陈述（中文主显，点击切换英文原文）
- 陈述标记：TRUE / FALSE + 扰动类型标签（DIRECTION_FLIP 等）
- 证据回链：点击证据 chip（如 `ev_f02_001`）→ 侧栏显示证据全文（角色/类型/页码）
- 审核操作：通过 / 拒绝 / 需要修改（localStorage 持久化，按 draft id）

## 3. API（app/routers/paper_questions.py）

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| GET | `/api/v1/paper-questions/{doc_id}?figure_id=` | 总览：按 Figure 分组，含实验摘要、蓝图上下文、陈述（A–E 标签）与证据预览 |
| GET | `/api/v1/paper-questions/{doc_id}/{set_id}` | 集合详情：完整中英陈述 + 证据原文 |

缺产物（未翻译/无草稿/无语义产物）返回 404 并带中文提示；蓝图产物缺失时
上下文优雅降级为空。证据 id 全局唯一（`figure_key` 前缀 `ev_f02_001`），
跨图不再碰撞。

## 4. 测试（22 项新增）

- `tests/paper_semantics/test_mcq_translation.py`（16）：模板断言（方向动词、
  数值保留、基因名不译、TRUE/FALSE 不变）、报告不变量（英文逐字保留、
  evidence 不变、中文含数值）、产物隔离 + 字节可复现
- `tests/test_paper_questions_api.py`（6）：TestClient 全链路夹具
  （reconstruct → blueprints → drafts → translate）、figure 过滤、集合详情
  含证据原文、404、`/review` 页面可达

```bash
pytest tests/paper_semantics/test_mcq_translation.py tests/test_paper_questions_api.py -q
pytest -q     # 349 passed, 1 skipped
```

## 5. paper_1 实测

24 题组 / 105 陈述全部翻译（94 template + 11 term_fallback）；10 个 Figure
分组展示；Figure 2 示例：

```text
A ✔ TRUE   cPLA2 抑制剂可提高 GFP 表达。
B ✘ DIRECTION_FLIP   cPLA2 抑制剂可降低 GFP 表达。
C ✘ CONCLUSION_FLIP  cPLA2 抑制剂不提高 GFP 表达。
D ✘ DV_SWAP          cPLA2 抑制剂可提高 CCR7 上调。
E ✘ PANEL_MISATTRIBUTION  根据 Figure 3，cPLA2 抑制剂可提高 GFP 表达。
```

## 6. 后续（不在本阶段）

L2 LLM 润色中文、A/B/C/D 终排与乱序、审核结果落库（当前 localStorage）、
Reviewer 自动预审。
