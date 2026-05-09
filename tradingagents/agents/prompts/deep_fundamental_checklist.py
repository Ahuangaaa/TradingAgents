"""Deep fundamental analysis checklist (Chinese outline) and synthesis rules.

Single source of truth for the structured checklist consumed by
:class:`~tradingagents.agents.analysts.deep_fundamental_checklist`.
"""

# User-provided structure; «管理增» interpreted as «管理层».

CHECKLIST_OUTLINE = """
## 1. 基本面分析

### a. 了解企业的产品
#### ⅰ. 产品的不同类别
#### ⅱ. 消费者对这些产品的需求情况
#### ⅲ. 产品需求的价格弹性；企业拥有产品定价权吗？
#### ⅳ. 每种产品的替代品分别是什么？它们是否存在差异？差异体现在价格方面还是质量方面？
#### ⅴ. 与这些产品相关的品牌名称
#### ⅵ. 与产品相关的专利保护情况

### b. 了解将产品推向市场所需要的技术细节
#### ⅰ. 产品的生产过程
#### ⅱ. 产品的营销过程
#### ⅲ. 产品的分销渠道
#### ⅳ. 供应商网络以及供应链的运作情况
#### ⅴ. 成本结构情况
#### ⅵ. 规模经济情况

### c. 了解企业的知识基础
#### ⅰ. 相关技术变革的方向和速度，以及企业对此的把握情况
#### ⅱ. 相关研究与开发项目
#### ⅲ. 与信息网络之间的关系
#### ⅳ. 在产品开发方面的创新能力
#### ⅴ. 在生产技术方面的创新能力
#### ⅵ. 学习的容易度

### d. 了解行业的竞争情况
#### ⅰ. 行业的集中度，行业中拥有的企业数量，它们的规模大小情况
#### ⅱ. 行业进入障碍，新进入者和替代产品出现的可能性。行业中是否存在品牌保护情况？客户的转换成本高低
#### ⅲ. 企业在行业中所处的地位。是领先者还是跟随着？具有成本优势吗？
#### ⅳ. 供应商的竞争程度。供应商具有市场定价能力吗？工会组织的力量如何？
#### ⅴ. 行业的产能情况，是产能过剩还是不足？
#### ⅵ. 与其他公司之间的关系和结盟情况

### e. 了解企业的管理层
#### ⅰ. 管理层的任职经历和业绩如何？
#### ⅱ. 管理层具有企业家意识吗？
#### ⅲ. 管理层对股东的重视程度如何？管理层成员有过只考虑自己利益的记录吗？他们喜欢拉帮结派吗？
#### ⅳ. 企业的股份支付计划符合股东和管理层的利益吗？
#### ⅴ. 企业在经营中涉及了哪些道德条款？管理层有违反这些条款的倾向吗？
#### ⅵ. 公司治理机制的强度如何？

### f. 了解政治，法律，监管和道德环境的影响
#### ⅰ. 企业的政治影响力
#### ⅱ. 企业所受到的法律约束情况，包括反垄断法，消费者保护法，劳动法和环境法等
#### ⅲ. 企业需要面对的监管约束情况，包括产品监督和市场监管两个方面
#### ⅳ. 企业的税收情况
""".strip()

SYNTHESIS_SYSTEM_INSTRUCTIONS = """
You are a senior equity research analyst. Produce ONE markdown report that answers the checklist below.

Hard rules:
- Copy EVERY heading from the checklist (same `##` / `###` / `####` levels and numbering). Do not merge, skip, or rename sections.
- Under each `####` item, write 1–3 short paragraphs (or tight bullets) grounded ONLY in the provided analyst bundle. If evidence is missing, write explicitly: **结论**：信息不足 — … (explain what was not in the sources). Never leave a section blank.
- Do not fabricate numbers, dates, or citations. Paraphrase only what the sources support; mark speculation as conditional language.
- If the analyst bundle (especially social/sentiment) shows **股东户数持续攀升或大幅增加**, assign **high risk weight**: note **筹码分散** and—in context of prior strength or elevated prices—**possible 主力出货**; do not let offsetting positives elsewhere eliminate this warning in checklist answers where relevant.
- Append one concise markdown table after section `f` summarizing the main evidence gaps (column: checklist item; column: gap note).
""".strip()
