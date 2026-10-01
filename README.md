# 清流文本卫士

可部署的中文垃圾文本过滤 MVP。默认使用可解释规则引擎；训练模型通过 `FeatureUnion` 并联原始字符 TF-IDF 与拼音字符 TF-IDF，再由 Logistic Regression 分类。存在模型文件时会自动组合模型分数与规则分数。

## 功能

- 单条及批量文本检测，输出 `normal`、`review`、`spam`
- 展示风险分数、类别、命中规则和可读原因
- SQLite 统计审计，只存文本哈希和长度，不存检测原文
- 用户主动纠错时保存原文，形成后续训练候选集
- FastAPI OpenAPI、Vue 操作台、Docker 单容器部署

## 本地运行

要求 Node.js 24+ 和 Python 3.12+。

```powershell
python -m pip install -r backend/requirements-dev.txt
npm install
```

终端一：

```powershell
$env:PYTHONPATH="$PWD/backend"
python -m uvicorn app.main:app --app-dir backend --reload --port 8787
```

终端二：

```powershell
npm run dev
```

开发页面为 `http://127.0.0.1:5173`，API 文档为 `http://127.0.0.1:8787/docs`。

## 生产运行

```powershell
npm run build
$env:PYTHONPATH="$PWD/backend"
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8787
```

或使用容器：

```powershell
docker build -t spam-filter .
docker run --rm -p 8787:8787 -v spam-data:/app/backend/data spam-filter
```

生产页面与 API 同源，由 FastAPI 提供，入口为 `http://127.0.0.1:8787`。

## 训练数据

### 公开语料准备

`backend/prepare_datasets.py` 会从 Hugging Face 拉取三份公开语料，转换后统一为 `text,label` 两列 CSV，并写入 `backend/datasets/`：

| 语料 | 用途 | 条数 | 说明 |
| --- | --- | --- | --- |
| `zjunlp/ChineseHarm-bench` | 训练 | 7,583 | 浙大有害内容检测集，取低俗色情、博彩、欺诈、黑产广告为 `1`，不违规为 `0` |
| `Olivi-9/Chinese-Spam-Dataset` | 训练 | 12,135 | 真实群聊垃圾文本 |
| `zjunlp/ChineseHarm-bench` 谩骂引战 | 辱骂训练 | 673 | 清洗后保留，详见「语料清洗」 |
| `Moricoo/chinese_spam_test` | 外部评测 | 1,989 | 与训练集零重叠，用于校验泛化能力 |

营销类与辱骂类分开建模：「谩骂引战」是人身攻击而非营销垃圾，混入二分类会同时损害两侧的准确率。

```powershell
$env:PYTHONPATH="$PWD/backend"
python backend/prepare_datasets.py
```

原始文件缓存在 `backend/datasets/raw/`，重复执行会复用缓存。各语料许可证由上传方声明，商用前需自行确认；脚本只做格式转换，不代为分发。

### 合并训练

训练器接受多个 CSV 并按归一化文本跨源去重，标签矛盾的行整条剔除。`--holdout` 指定的语料不参与训练，仅用于评估。

```powershell
$env:PYTHONPATH="$PWD/backend"
python backend/train.py backend/datasets/base.csv backend/datasets/augmented.csv --holdout backend/datasets/holdout.csv --abuse-data backend/datasets/abuse.csv --abuse-holdout backend/datasets/abuse_holdout.csv
```

该命令同时训练两个模型：营销垃圾模型以全部语料训练，辱骂模型以 `--abuse-data` 的正样本加上语料中的正常样本作为负样本。

标签 `0` 表示正常，`1` 表示垃圾；同时兼容 `normal/ham/正常` 与 `spam/垃圾`。若留出集列名与训练集不同，可用 `--holdout-text-column` 与 `--holdout-label-column` 指定。

模型默认输出至 `backend/models/spam_pipeline_pinyin.joblib`，指标输出至同目录的 `spam_pipeline_pinyin.metrics.json`，其中同时记录内部留出与外部留出两套结果。拼音特征会把「微信、薇信、维信、wei xin」投影到同一空间。重启 API 后会自动加载模型。

### 双维度检测

系统维护两条互相独立的判定维度，避免把「骂人」和「营销」混为一谈：

- `label` / `score`：营销垃圾（诈骗、博彩、色情引流、黑产广告）
- `abuse_label` / `abuse_score`：辱骂攻击（人身攻击、侮辱性称谓、亲属攻击）

因此「你这种废物也配说话」会得到 `label=normal` 且 `abuse_label=spam`——它不是垃圾短信，但属于应处置的辱骂内容，可以独立统计与处置。两个模型文件分别是 `spam_pipeline_pinyin.joblib` 与 `abuse_pipeline_pinyin.joblib`，可用 `SPAM_ABUSE_MODEL_PATH` 覆盖。

辱骂判定结合了规则与模型：

- 规则覆盖常见辱骂、亲属攻击，以及语料中大量使用的谐音与拼音缩写（`cs`、`sz`、`木琴`、`浮木`、`唐氏` 等规避写法）
- 负权重规则让「反网暴」「依法维权」「理性讨论」等正当表达不会被误判
- 模型负责处理规则覆盖不到的表达

### 语料清洗

ChineseHarm-bench 的「谩骂引战」标签噪声较重：1,904 条中仅约 44% 含明确辱骂特征，其余是游戏梗、口语和反网暴表达。直接训练会让模型把「就是」「这种人」这类中性词学成辱骂信号，实测会导致「他就是这种人，不用太在意」被误判为辱骂。

`prepare_datasets.py` 因此只保留含明确辱骂特征且非正当表达的样本，剔除 1,063 条（56%）。清洗后重训，误报从 3/22 降到 1/22。

### 阈值语义

营销维度有两条互相独立的判定线，配置校验只强制 `review <= model_spam`：

| 阈值 | 默认 | 作用 |
| --- | --- | --- |
| `review` | 0.35 | 低于此判 `normal` |
| `model_spam` | 0.65 | 综合分达到此判 `spam` |
| `rule_spam` | 0.72 | 规则分达到此可**独立**判 `spam` |

综合分 `score = max(rule_score, boosted_model_score)`。由于 `rule_spam(0.72) > model_spam(0.65)`，默认配置下 `rule_spam` 是冗余的——规则分一旦越过它，必然也已越过 `model_spam`，因此**默认配置下三条线对 3000 条评测集产生 0 条差异**（已逐条 diff 确认）。

它真正的价值在于把 `rule_spam` 调到 `model_spam` 以下，构建「高置信规则独立拦截」通道：此时即使模型分很低，单条强规则也能直接拦截。实测将 `model_spam=0.99` / `rule_spam=0.50` 时「博彩送彩金」判 `spam`，改为 `rule_spam=0.99` 后同一条文本判 `review`。

### 实测指标

垃圾模型（1,989 条零重叠留出集）：

| 模型 | 训练样本 | 内部留出 F1 | 外部留出 F1 | 外部留出准确率 |
| --- | --- | --- | --- | --- |
| 仅 `paulkm` 公开集 | 7,715 | 0.9631 | 0.0680 | 0.5038 |
| 合并三份语料 | 21,946 | 0.9417 | 0.8744 | 0.8748 |

原模型内部指标很高但外部接近随机，说明它记住了语料特征而未能泛化。

辱骂模型（673 条正样本 + 11,770 条正常负样本，2,157 条混合留出集）：

| 阶段 | 查准 | 查全 | F1 | 清洗后正样本漏检 |
| --- | --- | --- | --- | --- |
| 未清洗语料 | 0.9500 | 0.8500 | 0.8972 | — |
| 清洗后 | 0.9938 | 0.9464 | 0.9695 | 0/841 |

已知限制：清洗规则基于关键词，仍有约一半被剔除的混合句（如包含「父母双亡」的完整段落）会被判定为辱骂，因为它们确实含辱骂内容——这是源数据标签不精确导致的，模型的判断在多数情况下是合理的。

垃圾语料带来的代价是中性商务用语（如「联系」「项目」）因高频共现而分数偏高，部分正常文本会落入 `review` 灰区；同批样本中误判为 `spam` 的数量由 2 条降为 0 条。上线前仍应使用业务来源、时间后移且完成脱敏的独立测试集校准 `REVIEW_THRESHOLD`、`MODEL_SPAM_THRESHOLD` 与 `ABUSE_REVIEW_THRESHOLD`。

## 验证

```powershell
$env:PYTHONPATH="$PWD/backend"
python -m pytest backend/tests -q
npm run lint
npm run build
```
