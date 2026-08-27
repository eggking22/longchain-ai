# Question Blueprint（Phase 5）— 设计与用法

> **定位**：在 Paper Semantic Reconstruction → Experiment Model 之上新增确定性出题蓝图层。
> 本阶段**只生成蓝图**：无 MCQ 题干、无选项、无答案键、无 Reviewer、无 LLM。

## 1. 架构

```text
reconstruct_figures(persist=False)          （语义层只读重建，不重写任何产物）
        ↓ 图级 + 面板级，严格门控
Experiment Model（observations / groups / intervention / DV / relationship）
        ↓ 规则 + 模板（deterministic baseline）
Question Blueprint
 ├── RESULT_INTERPRETATION   结果理解（比较对象/指标/方向）
 ├── EXPERIMENTAL_DESIGN     实验设计（对照组/处理组/测量指标的作用）
 ├── SIMPLE_PREDICTION       简单预测（唯一模板，答案锚定已观测组）
 └── DATA_STATEMENT          数据陈述（仅字面数值）
        ↓
data/paper_semantics/{doc_id}/question_blueprints.json
```

Evidence-first 链条：`Evidence → Experiment → Blueprint`；每条蓝图的 `evidence_ids`
全部可回链 `evidence.jsonl`（未来出题约束如"结论型题必须有 direct_observation
证据"直接可查）。

## 2. 四种题型与门控矩阵

| 题型 | reasoning_operation | 门控（不满足即不生成，计入 summary.skipped） | 上限/图 |
|---|---|---|---|
| RESULT_INTERPRETATION | comparison / result_interpretation | ≥1 条有向 observation + intervention + DV（图级与面板级均可；INSUFFICIENT 图不出） | 3 |
| EXPERIMENTAL_DESIGN | experimental_design_reasoning | 双组齐备 + DV + 存在 experimental_design 证据；expected_answer 仅复述已记录槽位，不发明实验目的 | 3 |
| SIMPLE_PREDICTION | local_prediction | 图级 SUFFICIENT + 双组 + 有向 observation + **关系 ∈ {causal, inhibition, activation, knockout, overexpression, dose_response}**；association/correlation 一律不出 | 1 |
| DATA_STATEMENT | quantitative_reading | 证据句中存在**字面数值**（百分比 / fold change / p 值 / 浓度 / 时间），且证据为 direct_observation 或 statistical_result（Methods 数值与 continuation 段不算"报告结果"） | 4 |

### 2.1 关键安全设计

- **预测题的答案锚定已观测组**：唯一模板 "If {intervention} were applied to the
  {control} group…" 的答案 "as already observed in the {experimental} group" 是
  已经观测到的事实，不是自由外推。
- **association ≠ causation**：关系类型从 Observation 原样带入
  `detail.relationship_type`；association 实验的 focus/answer 使用
  "relationship / associated with" 措辞，且永不生成预测题。
- **绝不伪造数值**：DATA_STATEMENT 只取正文字面值并绑定所在句
  （`detail.sentence`）；"significantly increased" 不产生任何数值蓝图；图片视觉
  信息（曲线高度等）根本不存在于文本管线中。
- **观察/解释/结论分层不混**：expected_answer 优先复用 Conclusion 层陈述；
  Interpretation 层不进入蓝图答案。

## 3. Schema（app/schemas/question_blueprint.py）

```text
QuestionBlueprint
├── blueprint_id      # qb_{实验键}_{题型缩写}_{序号}，如 qb_f02_ri_001 / qb_f02a_ds_001
├── question_type     # 四题型 Literal
├── experiment_id     # exp_f02 / exp_f02a
├── figure_id / panel_ids
├── question_focus    # 考察点（含比较对象/指标；association 保持原关系词）
├── required_evidence # 所需证据类型，如 ["direct_observation","experimental_design"]
├── reasoning_operation
├── expected_answer   # 确定性模板答案（仅由已记录槽位合成）
├── evidence_ids      # 全部回链 Evidence Store
├── confidence        # 继承来源图/面板重建置信度
├── status            # 本版只输出 READY；门控不足即整条不生成
└── detail            # 结构化槽位（relationship_type / comparison / data_value…）
```

## 4. CLI 与产物

```bash
python scripts/generate_blueprints.py --doc-id paper_1
python scripts/generate_blueprints.py --doc-id paper_1 --figure 2 --type DATA_STATEMENT --json
```

Exit codes：`0` = 完成；`1` = 输入缺失。产物
`data/paper_semantics/{doc_id}/question_blueprints.json`
（`doc_id → summary{total, by_type, skipped, method} → blueprints[]`；零时间戳、
字节级可复现；**不改动** figures.json / evidence.jsonl / experiments.json /
report.md / manifest.json）。

## 5. paper_1 实测（Nature Immunology，33 页）

```text
blueprints: 257   RI 71 / ED 3 / PRED 6 / DS 177
SIMPLE_PREDICTION: 恰好 6 个 SUFFICIENT 图各 1 条；8 个 PARTIAL 图全部被
                   figure_not_sufficient 门控阻断（零强行出题）
skipped 透明计数： missing_intervention_or_endpoint 43 / incomplete_design_slots 7 /
                   no_literal_numeric_value 14 / …
验证：产物字节可复现 ✓；五个既有产物文件字节不变 ✓；
     257 条蓝图 evidence_ids 全部回链 evidence store ✓；
     association 图零预测题 ✓
```

已知噪声：DATA_STATEMENT 的 question_focus 引用的 DV 名称偶含语义层 DV 抽取噪
声（如 "further trigger CCR7 expression"）；数值本身始终绑定字面句不受影响。
待语义层 DV 词表优化后自动改善。

## 6. 测试（tests/paper_semantics/test_question_blueprint.py）

四类均生成 / 绑定与面板作用域正确 / evidence_ids 全回链 / 百分比+p 值字面提取 /
"significantly" 不产生数值 / 无证据外数字 / association 不升级因果（含双组
association 单元测试）/ PARTIAL·INSUFFICIENT 阻断 / 产物隔离与字节可复现。

```bash
pytest tests/paper_semantics/test_question_blueprint.py -q
pytest -q     # 303 passed（286 既有零修改 + 17 新增）
```

## 7. 后续（不在本阶段）

MCQ 题干与选项渲染（消费 `detail` 结构化槽位）、干扰项生成、MCQ Reviewer、
L2 LLM refinement（在 deterministic baseline 之上做措辞归一，同 Phase 4 的
证据绑定校验模式）。
