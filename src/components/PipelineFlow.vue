<script setup lang="ts">
/**
 * 流水线线框图：按数据流向绘制节点与连线，节点内直接展示可调参数。
 * 纯 CSS/HTML，不引入图形库，便于自定义主题与响应式。
 */
import { computed } from 'vue'
import type { PipelineConfig } from '../api'

const props = defineProps<{ pipeline: PipelineConfig }>()
const emit = defineEmits<{ edit: [field: string, value: number] }>()

const spam = computed(() => props.pipeline.models.find((m) => m.dimension === 'spam'))
const abuse = computed(() => props.pipeline.models.find((m) => m.dimension === 'abuse'))
const spamRules = computed(() => props.pipeline.rules.spam)
const abuseRules = computed(() => props.pipeline.rules.abuse.filter((r) => r.id !== 'abuse-slur'))
const maxRuleWeight = computed(() =>
  Math.max(0.01, ...[...spamRules.value, ...abuseRules.value].map((r) => Math.abs(r.weight))),
)

const SPAM_KEYS = ['review', 'model_spam', 'rule_spam'] as const
const ABUSE_KEYS = ['abuse_review', 'abuse_suspicious', 'abuse_spam'] as const

const THRESHOLD_LABELS: Record<string, string> = {
  rule_spam: '纯规则拦截',
  model_spam: '模型拦截',
  review: '人工复核',
  abuse_spam: '辱骂拦截',
  abuse_suspicious: '辱骂可疑',
  abuse_review: '辱骂复核',
}

function pct(value: number): string {
  return `${Math.max(2, (Math.abs(value) / maxRuleWeight.value) * 100)}%`
}

function num(e: Event): number {
  return Number((e.target as HTMLInputElement).value)
}
</script>

<template>
  <div class="flow">
    <div class="node io">
      <span class="node-kind">入口</span>
      <strong>原始文本</strong>
      <small>≤ 10,000 字</small>
    </div>

    <div class="edge"><i></i></div>

    <div class="node wide">
      <span class="node-kind">预处理</span>
      <strong>归一化 · {{ pipeline.normalization.length }} 步</strong>
      <ol class="steps">
        <li v-for="(s, i) in pipeline.normalization" :key="i">{{ s }}</li>
      </ol>
    </div>

    <div class="edge"><i></i></div>

    <div class="branches">
      <section class="branch">
        <header class="branch-head spam"><span>维度 A</span><strong>营销垃圾</strong></header>

        <div class="node">
          <span class="node-kind">规则引擎 · {{ spamRules.length }} 条</span>
          <div class="rows">
            <label v-for="r in spamRules" :key="r.id" class="rule-row">
              <span class="rule-name" :title="r.reason">{{ r.id }}</span>
              <span class="bar"><i :style="{ width: pct(r.weight) }" :class="{ neg: r.weight < 0 }"></i></span>
              <input class="w-in" type="number" step="0.01" min="-1" max="1" :value="r.weight"
                     @change="emit('edit', `rules.spam.${r.id}`, num($event))"
              />
            </label>
          </div>
          <p class="hint">score = 1 − ∏(1 − w)</p>
        </div>

        <div class="edge"><i></i></div>

        <div class="node">
          <span class="node-kind">统计模型 · {{ spam?.trained_rows.toLocaleString() }} 条</span>
          <strong>{{ spam?.label }}</strong>
          <div class="rows">
            <label v-for="f in spam?.features ?? []" :key="f.name" class="feat-row">
              <span>{{ f.label }}</span>
              <input class="w-in retrain" type="number" step="0.1" min="0" max="5" :value="f.weight"
                     @change="emit('edit', `models.spam.features.${f.name}`, num($event))"
              />
            </label>
          </div>
          <p class="hint warn-hint">改特征权重需重训模型</p>
        </div>

        <div class="edge"><i></i></div>

        <div class="node">
          <span class="node-kind">分数融合</span>
          <p class="formula">
            max(rule, model + (1−model) × rule × <b>{{ pipeline.fusion.rule_boost_factor }}</b>)
          </p>
          <label class="inline">系数
            <input class="w-in narrow" type="number" step="0.05" min="0" max="1"
                   :value="pipeline.fusion.rule_boost_factor"
                   @change="emit('edit', 'fusion.rule_boost_factor', num($event))"
            />
          </label>
        </div>

        <div class="edge"><i></i></div>

        <div class="node">
          <span class="node-kind">三档判定</span>
          <div class="rows">
            <label v-for="key in SPAM_KEYS" :key="key" class="rule-row">
              <span class="rule-name">{{ THRESHOLD_LABELS[key] }}</span>
              <span></span>
              <input class="w-in" type="number" step="0.01" min="0" max="1" :value="pipeline.thresholds[key]"
                     @change="emit('edit', `thresholds.${key}`, num($event))"
              />
            </label>
          </div>
        </div>
      </section>

      <section class="branch">
        <header class="branch-head abuse"><span>维度 B</span><strong>辱骂歧视</strong></header>

        <div class="node">
          <span class="node-kind">规则引擎 · {{ abuseRules.length + 1 }} 条</span>
          <div class="rows">
            <label v-for="r in abuseRules" :key="r.id" class="rule-row">
              <span class="rule-name" :title="r.reason">{{ r.id }}</span>
              <span class="bar"><i :style="{ width: pct(r.weight) }" :class="{ neg: r.weight < 0 }"></i></span>
              <input class="w-in" type="number" step="0.01" min="-1" max="1" :value="r.weight"
                     @change="emit('edit', `rules.abuse.${r.id}`, num($event))"
              />
            </label>
          </div>
          <p class="hint">负权重 = 正当表达，直接归零</p>
        </div>

        <div class="edge"><i></i></div>

        <div class="node">
          <span class="node-kind">统计模型 · {{ abuse?.trained_rows.toLocaleString() }} 条</span>
          <strong>{{ abuse?.label }}</strong>
          <div class="rows">
            <label v-for="f in abuse?.features ?? []" :key="f.name" class="feat-row">
              <span>{{ f.label }}</span>
              <input class="w-in retrain" type="number" step="0.1" min="0" max="5" :value="f.weight"
                     @change="emit('edit', `models.abuse.features.${f.name}`, num($event))"
              />
            </label>
          </div>
          <p class="hint warn-hint">改特征权重需重训模型</p>
        </div>

        <div class="edge"><i></i></div>

        <div class="node">
          <span class="node-kind">三档判定</span>
          <div class="rows">
            <label v-for="key in ABUSE_KEYS" :key="key" class="rule-row">
              <span class="rule-name">{{ THRESHOLD_LABELS[key] }}</span>
              <span></span>
              <input class="w-in" type="number" step="0.01" min="0" max="1" :value="pipeline.thresholds[key]"
                     @change="emit('edit', `thresholds.${key}`, num($event))"
              />
            </label>
          </div>
        </div>
      </section>
    </div>

    <div class="edge"><i></i></div>

    <div class="node io">
      <span class="node-kind">出口</span>
      <strong>双维度结论</strong>
      <div class="out-grid">
        <span>营销 label</span>
        <span>辱骂 abuse_label</span>
        <span>灰区 suspicious</span>
        <span>依据 reasons</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.flow { display: flex; flex-direction: column; align-items: center; }

.node {
  width: 100%; max-width: 560px; padding: 14px 16px; border-radius: 14px;
  border: 1px solid var(--border); background: rgba(2, 6, 16, 0.5);
  display: flex; flex-direction: column; gap: 8px;
}
.node.wide { max-width: 620px; }
.node.io { max-width: 360px; text-align: center; align-items: center; }
.node-kind { font-size: 10px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
.node strong { font-size: 14px; }
.node small { font-size: 11px; color: var(--muted); }

.steps { margin: 0; padding-left: 18px; display: grid; grid-template-columns: 1fr 1fr; gap: 3px 14px; }
.steps li { font-size: 11px; color: var(--muted); }
.steps li::marker { color: var(--cyan); }

.edge { height: 26px; display: flex; justify-content: center; }
.edge i { width: 2px; height: 100%; background: linear-gradient(180deg, var(--cyan), var(--violet)); opacity: 0.55; position: relative; }
.edge i::after {
  content: ''; position: absolute; bottom: -1px; left: -3px;
  border-left: 4px solid transparent; border-right: 4px solid transparent;
  border-top: 6px solid var(--violet);
}

.branches { display: grid; gap: 18px; width: 100%; }
@media (min-width: 1100px) { .branches { grid-template-columns: 1fr 1fr; } }

.branch { display: flex; flex-direction: column; align-items: center; }
.branch-head {
  display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
  padding: 6px 14px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border);
}
.branch-head span { font-size: 10px; opacity: 0.7; }
.branch-head.spam { background: rgba(56, 189, 248, 0.12); color: var(--cyan); }
.branch-head.abuse { background: rgba(167, 139, 250, 0.14); color: var(--violet); }

.rows { display: flex; flex-direction: column; gap: 5px; }
.rule-row { display: grid; grid-template-columns: 116px 1fr 62px; align-items: center; gap: 8px; }
.rule-name { font-size: 10.5px; color: var(--muted); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar { height: 5px; border-radius: 3px; background: rgba(148, 163, 184, 0.14); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--cyan), var(--violet)); }
.bar i.neg { background: linear-gradient(90deg, var(--green), #6ee7b7); }
.feat-row { display: grid; grid-template-columns: 1fr 62px; align-items: center; gap: 8px; font-size: 11.5px; color: var(--muted); }

.w-in {
  width: 100%; padding: 4px 6px; border-radius: 7px; border: 1px solid var(--border);
  background: rgba(2, 6, 16, 0.7); color: var(--ink);
  font-family: var(--mono); font-size: 11.5px; text-align: right; transition: border-color 0.15s ease;
}
.w-in:focus { outline: none; border-color: var(--cyan); }
.w-in.retrain { border-color: rgba(251, 191, 36, 0.4); }
.w-in.narrow { width: 62px; }

.formula { margin: 0; font-size: 11px; color: var(--muted); font-family: var(--mono); }
.formula b { color: var(--cyan); }
.inline { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); }
.hint { margin: 0; font-size: 10.5px; color: var(--muted); opacity: 0.8; }
.warn-hint { color: var(--amber); opacity: 0.9; }
.out-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
.out-grid span { font-size: 11px; color: var(--muted); font-family: var(--mono); }
</style>
