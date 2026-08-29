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

## 2. 真陈述构造

- RI：detail.significance=="significant" 时 `{intervention} significantly {increases/decreases} {dv}.`（否则 expected_answer）
- ED / PREDICTION：expected_answer 原样
- **DATA：图锚点 + 证据句原文引用**（对象与定位来自句子本身，零猜测）：
  `According to Figure 1, DCs spent ∼35% of their time displaying diameters between ∼2 and 4 µm.`
  - 仅排版级清洗：细空格/不换行空格归一（Nature 的 "25 µM" U+2009 ↔ 抽取值的 ASCII 空格）、
    剪句首面板标号（"a, GFP intensity…"）、合并断行连字符（"dis- playing"→"displaying"、
    "Color- coded"→"Color-coded"）、剪掉句末悬空 `(Fig.` 碎片与纯图引用尾 `(Figure 6)`
  - 安全回退链：值不在句中 / 句子是分句碎片 / 引用超 240 字符且值在窗口外 →
    `According to Figure 2, the reported concentration is 25 µM.`（类型标签，不猜对象）
  - **可选 LLM 对象抽取**（`--no-llm` 可关；LLM_API_KEY+LLM_MODEL 配置时自动启用）：
    仅作用于上述回退尾部，输入证据句+数值，输出对象短语；**必须逐字是证据文本的连续
    span（≤80 字符）否则整体拒绝**，回退形态保持——LLM 永不发明措辞。升级后形态：
    `According to Figure 6, the reported percentage for migration distance is 30%.`
    （zh：`根据 Figure 6，论文报告的迁移距离的百分比为 30%。`）；
    summary 记 `object_extraction{extracted,rejected,errors}`，任一成功则 method=`deterministic+llm`
  - **跳过**：值本身是面板引用误读（"2d"）且句子为碎片 → 不生成该集（宁缺毋滥）
  - NUMERIC_MUTATION 在引用句内仅替换数值本身，句身逐字不变

## 3. 十类受控扰动

| 类型 | 门控（不满足即跳过并计数） | 最小编辑 |
|---|---|---|
| DIRECTION_FLIP | 陈述含方向动词 | 单趟词级翻转 increases↔decreases / higher↔lower |
| SIGNIFICANCE_FLIP | 真陈述含 "significantly"（RI 显著性模板提供） | `X significantly increases Y.` → `X does not significantly increase Y.` |
| GROUP_SWAP | 实验组与对照组名都在陈述中出现 | 双名占位交换 |
| CAUSALITY_UPGRADE | **仅** association/correlation 蓝图 | "is associated with" → "causes"（只作假陈述出现） |
| CONCLUSION_FLIP | 陈述含动词/关联措辞 | 否定插入：increases→does not increase；is associated with→is not associated with |
| DV_SWAP | 同一实验有 ≥2 个因变量（池卫生过滤后） | 指标名替换为同实验另一指标 |
| VARIABLE_SWAP | 同图证据存在其它干净处理名 | 干预名替换（过滤 "treated with…"/"versus…" 噪声跨名） |
| PANEL_MISATTRIBUTION | 有兄弟面板或其它图 | 非 DATA：前缀 `According to {另一面板/图}, `；DATA 引用句已带锚点 → **替换首锚点** |
| NUMERIC_MUTATION | 仅 DATA 蓝图；同 kind 全文字面池有其它值 | 在引用句内把数值替换为池中另一原文值（如 35%→50%，句身逐字不变） |
| CONDITION_MUTATION | **非 DATA 蓝图**；图证据句含浓度/时间字面值且池有同类替代 | 条件值替换（如 (25 µM)→(10 µM)），额外绑定来源证据 |

## 4. 安全约束

- **Evidence-first**：真陈述与所有替换素材仅来自论文证据；假陈述 = 真内容的最小错误化
- **association 永不作为真陈述升级为因果**（测试断言）；CAUSALITY_UPGRADE 只产生假陈述
- **数值零伪造**：NUMERIC_MUTATION 的替换值必须来自全文字面池（测试断言 ∈ {论文原文值}）
- **扰动失败即跳过**（文本未变/重复/池空）；set 需要 ≥1 假陈述，否则不生成
- 所有 draft 保留 `evidence_ids`（数值/条件变异额外绑定替换值来源证据）

## 5. Schema（app/schemas/question_draft.py）

```text
StatementDraft     draft_id / blueprint_id / figure_id / panel_ids / statement /
                   is_correct / perturbation_type(NONE+十类) / evidence_ids /
                   confidence / status / detail{base=真陈述}
StatementDraftSet  draft_set_id(qd_f02_001) / blueprint_id / question_type / statements
QuestionDraftReport doc_id / summary{sets, statements, true/false, by_perturbation, skipped} / draft_sets
```

## 6. CLI 与产物

```bash
python scripts/generate_question_drafts.py --doc-id paper_1 [--figure 2] [--json]
```

`data/paper_semantics/{doc_id}/question_drafts.json`；不改动任何既有产物文件。

## 7. paper_1 实测

```text
22 sets / 96 statements（22 真 / 74 假；2 个集因面板引用误读值跳过）
by_perturbation: CONCLUSION_FLIP 14 · DIRECTION_FLIP 14 · DV_SWAP 14 ·
                 PANEL_MISATTRIBUTION 15 · NUMERIC_MUTATION 8 · VARIABLE_SWAP 6 ·
                 GROUP_SWAP 2 · CAUSALITY_UPGRADE 1（CONDITION_MUTATION 已不对 DATA 集生效，碎片形态归零）
DATA 覆盖：8/8 集全部为"图锚点+证据句引用"形态（细空格归一修复后无类型标签回退；
LLM 对象抽取层在 paper_1 上 0 候选，为其它论文兜底）
验证：产物字节可复现 ✓；question_blueprints.json 等既有产物字节不变 ✓；96 条陈述 evidence_ids 全回链 ✓
```

示例（qd_f02_001，RESULT_INTERPRETATION）：
```text
✔ TRUE  cPLA2 inhibitor increases GFP expression.
✘ DIRECTION_FLIP   cPLA2 inhibitor decreases GFP expression.
✘ CONCLUSION_FLIP  cPLA2 inhibitor does not increase GFP expression.
✘ DV_SWAP          cPLA2 inhibitor increases CCR7 upregulation.
✘ PANEL_MISATTRIBUTION  According to Figure 3, cPLA2 inhibitor increases GFP expression.
```

示例（DATA，qd_f01_001，证据句原文引用）：
```text
✔ TRUE  According to Figure 1, We further observed that DCs spent ∼35% of their time
        displaying diameters between ∼2 and 4 µm and spent ∼50% of their time at diameters of >4 µm.
✘ PANEL_MISATTRIBUTION  同句，锚点换成 Figure 2
✘ NUMERIC_MUTATION      同句，仅 35% → 50%（句身逐字不变）
```

已知噪声：SIGNIFICANCE_FLIP/GROUP_SWAP 依赖真陈述含相应措辞，paper_1 上激活率低
（模板覆盖所限）；假陈述英文语法在上游槽位噪声时偶有瑕疵，最终中文渲染阶段统一规范化。

## 8. 测试（test_question_draft.py 35 项 + test_llm_object_extractor.py 15 项）

集合结构（恰 1 真 / ≥1 假 / 证据非空 / 文本去重）/ 六类扰动确定性断言 /
CAUSALITY_UPGRADE 仅作用 association 且真陈述永不含因果措辞 / NUMERIC_MUTATION
句内替换（骨架不变 + 值 ∈ 字面池）/ DATA 锚点引用（断行连字符合并、悬空引用剪除、
碎片句回退、面板引用值跳过、超长截断、细空格归一、CONDITION_MUTATION 排除）/
LLM 对象抽取（MockTransport 全离线：逐字 span 验证、拒绝/错误/重问、pipeline
升级与回退、无 extractor 行为不变）/ 产物隔离与字节可复现。

```bash
pytest tests/paper_semantics/test_question_draft.py tests/paper_semantics/test_llm_object_extractor.py -q
pytest -q     # 388 passed, 1 skipped
```

## 9. 后续（不在本阶段）

中文渲染已由 MCQ Step 2 完成（见 `docs/mcq_review.md`）。剩余：A/B/C/D 终排与
乱序、干扰项难度分级、审核结果落库、L2 LLM refinement。
