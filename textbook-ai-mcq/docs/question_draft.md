# Question Draft（MCQ Generator Step 1）— 设计与用法

> **定位**：Question Blueprint → Statement Draft。为"下列哪些说法正确/错误？"题型
> 生成结构化题目逻辑：每个集合恰 1 条真陈述 + ≤4 条受控错误假陈述。
> 本阶段不做：中文翻译、A/B/C/D 排版、MCQ Reviewer、L2 LLM。

## 1. 架构

```text
Question Blueprint（每图每题型取首条，固定优先序 RI > PRED > DATA > ED，每图 ≤3 set）
        ↓ 真陈述（仅由蓝图绑定内容构造：expected_answer / 显著性措辞模板 / 字面数值）
受控扰动（十类，最小编辑，替换素材全部来自论文证据池）
        ↓
StatementDraftSet（恰 1 真 + ≤4 假，文本去重；无可用扰动则整个 set 跳过）
        ↓
data/paper_semantics/{doc_id}/question_drafts.json（零时间戳、字节可复现）
```

## 2. 十类受控扰动

| 类型 | 门控（不满足即跳过并计数） | 最小编辑 |
|---|---|---|
| DIRECTION_FLIP | 陈述含方向动词 | 单趟词级翻转 increases↔decreases / higher↔lower |
| SIGNIFICANCE_FLIP | 真陈述含 "significantly"（RI 显著性模板提供） | `X significantly increases Y.` → `X does not significantly increase Y.` |
| GROUP_SWAP | 实验组与对照组名都在陈述中出现 | 双名占位交换 |
| CAUSALITY_UPGRADE | **仅** association/correlation 蓝图 | "is associated with" → "causes"（只作假陈述出现） |
| CONCLUSION_FLIP | 陈述含动词/关联措辞 | 否定插入：increases→does not increase；is associated with→is not associated with |
| DV_SWAP | 同一实验有 ≥2 个因变量（池卫生过滤后） | 指标名替换为同实验另一指标 |
| VARIABLE_SWAP | 同图证据存在其它干净处理名 | 干预名替换（过滤 "treated with…"/"versus…" 噪声跨名） |
| PANEL_MISATTRIBUTION | 有兄弟面板或其它图 | `According to {另一面板/图}, {真陈述}` |
| NUMERIC_MUTATION | 仅 DATA 蓝图；同 kind 全文字面池有其它值 | 数值替换为池中另一原文值（如 30%→50%，两值都为论文原文） |
| CONDITION_MUTATION | 图证据句含浓度/时间字面值且池有同类替代 | 条件值替换（如 (25 µM)→(10 µM)），额外绑定来源证据 |

## 3. 安全约束

- **Evidence-first**：真陈述与所有替换素材仅来自论文证据；假陈述 = 真内容的最小错误化
- **association 永不作为真陈述升级为因果**（测试断言）；CAUSALITY_UPGRADE 只产生假陈述
- **数值零伪造**：NUMERIC_MUTATION 的替换值必须来自全文字面池（测试断言 ∈ {论文原文值}）
- **扰动失败即跳过**（文本未变/重复/池空）；set 需要 ≥1 假陈述，否则不生成
- 所有 draft 保留 `evidence_ids`（数值/条件变异额外绑定替换值来源证据）

## 4. Schema（app/schemas/question_draft.py）

```text
StatementDraft     draft_id / blueprint_id / figure_id / panel_ids / statement /
                   is_correct / perturbation_type(NONE+十类) / evidence_ids /
                   confidence / status / detail{base=真陈述}
StatementDraftSet  draft_set_id(qd_f02_001) / blueprint_id / question_type / statements
QuestionDraftReport doc_id / summary{sets, statements, true/false, by_perturbation, skipped} / draft_sets
```

## 5. CLI 与产物

```bash
python scripts/generate_question_drafts.py --doc-id paper_1 [--figure 2] [--json]
```

`data/paper_semantics/{doc_id}/question_drafts.json`；不改动任何既有产物文件。

## 6. paper_1 实测

```text
24 sets / 105 statements（24 真 / 81 假）
by_perturbation: CONCLUSION_FLIP 14 · DIRECTION_FLIP 14 · DV_SWAP 14 ·
                 PANEL_MISATTRIBUTION 17 · NUMERIC_MUTATION 10 · VARIABLE_SWAP 6 ·
                 CONDITION_MUTATION 5 · CAUSALITY_UPGRADE 1
验证：产物字节可复现 ✓；六个既有产物文件字节不变 ✓；105 条陈述 evidence_ids 全回链 ✓
```

示例（qd_f02_001，RESULT_INTERPRETATION）：
```text
✔ TRUE  cPLA2 inhibitor increases GFP expression.
✘ DIRECTION_FLIP   cPLA2 inhibitor decreases GFP expression.
✘ CONCLUSION_FLIP  cPLA2 inhibitor does not increase GFP expression.
✘ DV_SWAP          cPLA2 inhibitor increases CCR7 upregulation.
✘ PANEL_MISATTRIBUTION  According to Figure 3, cPLA2 inhibitor increases GFP expression.
```

已知噪声：SIGNIFICANCE_FLIP/GROUP_SWAP 依赖真陈述含相应措辞，paper_1 上激活率低
（模板覆盖所限）；假陈述英文语法在上游槽位噪声时偶有瑕疵，最终中文渲染阶段统一规范化。

## 7. 测试（tests/paper_semantics/test_question_draft.py，15 项）

集合结构（恰 1 真 / ≥1 假 / 证据非空 / 文本去重）/ 六类扰动确定性断言 /
CAUSALITY_UPGRADE 仅作用 association 且真陈述永不含因果措辞 / NUMERIC_MUTATION
值 ∈ 字面池（30%/50% 注入语料）/ 产物隔离与字节可复现。

```bash
pytest tests/paper_semantics/test_question_draft.py -q
pytest -q     # 327 passed, 1 skipped
```

## 8. 后续（不在本阶段）

中文渲染已由 MCQ Step 2 完成（见 `docs/mcq_review.md`）。剩余：A/B/C/D 终排与
乱序、干扰项难度分级、审核结果落库、L2 LLM refinement。
