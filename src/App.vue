<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api } from './api'
import type { FilterResult, PipelineConfig, Stats } from './api'
import PipelinePanel from './components/PipelinePanel.vue'
import ScoreRing from './components/ScoreRing.vue'

const text = ref('限时低息借款，无抵押秒到账，咨询13800138000')
const result = ref<FilterResult | null>(null)
const stats = ref<Stats | null>(null)
const pipeline = ref<PipelineConfig | null>(null)
const loading = ref(false)
const error = ref('')
const feedbackSent = ref(false)
const activeTab = ref<'detect' | 'pipeline'>('detect')

const EXAMPLES = [
  '会议调整到下午三点，请带上最新的项目材料。',
  '限时低息借款，无抵押秒到账，咨询13800138000',
  '加我微信：service_888，了解本周活动',
  '你这种废物也配说话，滚远点别恶心人',
  '博彩送彩金，立即下注稳赚不赔',
  '关爱孤儿，不要网暴他',
]

const LABEL_META: Record<string, { title: string; sub: string }> = {
  normal: { title: '建议放行', sub: '未发现明显垃圾信号' },
  review: { title: '需要复核', sub: '存在风险信号，建议人工确认' },
  spam: { title: '建议拦截', sub: '命中高风险垃圾文本特征' },
}

const spamMeta = computed(() => LABEL_META[result.value?.label ?? 'normal'])

const abuseTone = computed(() => {
  if (!result.value) return 'normal'
  if (result.value.abuse_label === 'spam') return 'spam'
  if (result.value.abuse_label === 'review' || result.value.abuse_suspicious) return 'review'
  return 'normal'
})

const abuseTitle = computed(() => {
  if (abuseTone.value === 'spam') return '辱骂命中'
  if (abuseTone.value === 'review') return '辱骂可疑 · 需人工确认'
  return '无辱骂信号'
})

const gauges = computed(() => [
  { key: 'rule', label: '规则分', value: result.value?.rule_score ?? 0 },
  { key: 'model', label: '模型分', value: result.value?.model_score ?? 0 },
  { key: 'spam', label: '综合垃圾分', value: result.value?.score ?? 0, primary: true },
  { key: 'abuse', label: '辱骂分', value: result.value?.abuse_score ?? 0 },
])

const metrics = computed(() => [
  { k: '累计检测', v: stats.value?.total ?? 0 },
  { k: '建议拦截', v: stats.value?.spam ?? 0 },
  { k: '待复核', v: stats.value?.review ?? 0 },
  { k: '辱骂命中', v: stats.value?.abuse ?? 0 },
  { k: '纠错样本', v: stats.value?.feedback ?? 0 },
])

async function loadStats() {
  try {
    stats.value = await api.stats()
  } catch {
    /* 统计失败不阻断主流程 */
  }
}

async function analyze() {
  if (!text.value.trim()) return
  loading.value = true
  error.value = ''
  feedbackSent.value = false
  try {
    result.value = await api.filter(text.value)
    await loadStats()
  } catch (e) {
    error.value = e instanceof Error ? e.message : '网络请求失败'
  } finally {
    loading.value = false
  }
}

async function submitFeedback(corrected: 'spam' | 'normal', correctedAbuse: 'abuse' | 'normal' = 'normal') {
  if (!result.value) return
  try {
    await api.feedback({
      text: result.value.text,
      predicted_label: result.value.label,
      corrected_label: corrected,
      corrected_abuse_label: correctedAbuse,
    })
    feedbackSent.value = true
    await loadStats()
  } catch {
    error.value = '反馈提交失败'
  }
}

function reset() {
  text.value = ''
  result.value = null
  error.value = ''
  feedbackSent.value = false
}

onMounted(async () => {
  await loadStats()
  try {
    pipeline.value = await api.pipeline()
  } catch {
    /* 流水线面板可选，失败时降级 */
  }
})
</script>
<template>
  <div class="console">
    <div class="aurora" aria-hidden="true"></div>

    <header class="topbar">
      <div class="brand">
        <span class="brand-mark">清</span>
        <div class="brand-text">
          <strong>清流 · 文本安全控制台</strong>
          <small>Pipeline v{{ pipeline?.version ?? '—' }}</small>
        </div>
      </div>
      <nav class="tabs" role="tablist">
        <button
          v-for="tab in (['detect', 'pipeline'] as const)"
          :key="tab"
          role="tab"
          type="button"
          :aria-selected="activeTab === tab"
          :class="['tab', { active: activeTab === tab }]"
          @click="activeTab = tab"
        >
          {{ tab === 'detect' ? '实时检测' : '流水线权重' }}
        </button>
      </nav>
      <div class="pulse">
        <span class="dot"></span>
        <em>{{ pipeline?.models.length ?? 0 }} 模型在线</em>
      </div>
    </header>

    <main>
      <Transition name="fade" mode="out-in">
        <section v-if="activeTab === 'detect'" key="detect" class="stage">
          <div class="composer glass">
            <label class="field-label" for="input">待检测文本</label>
            <textarea
              id="input"
              v-model="text"
              maxlength="10000"
              placeholder="粘贴短信、评论或聊天内容…"
              @keydown.ctrl.enter="analyze"
            ></textarea>
            <div class="composer-actions">
              <span class="counter">{{ text.length.toLocaleString() }} / 10,000</span>
              <div class="btn-group">
                <button class="ghost" type="button" @click="reset">清空</button>
                <button class="primary" type="button" :disabled="loading || !text.trim()" @click="analyze">
                  {{ loading ? '检测中…' : '开始检测' }}
                </button>
              </div>
            </div>
            <div class="examples">
              <span>示例</span>
              <button v-for="(ex, i) in EXAMPLES" :key="i" type="button" @click="text = ex">{{ i + 1 }}</button>
            </div>
          </div>

          <div class="verdict-panel glass" aria-live="polite">
            <div v-if="error" class="state error-state">
              <strong>检测失败</strong>
              <p>{{ error }}</p>
            </div>

            <div v-else-if="!result" class="state placeholder">
              <div class="orb"></div>
              <p>等待检测</p>
              <small>结果、风险分数与命中依据将显示在此</small>
            </div>

            <div v-else class="state verdict" :class="result.label">
              <div class="verdict-head">
                <div class="verdict-text">
                  <small>{{ spamMeta.sub }}</small>
                  <h2>{{ spamMeta.title }}</h2>
                </div>
                <ScoreRing :value="result.score" :tone="result.label" />
              </div>

              <div class="gauges">
                <div v-for="g in gauges" :key="g.key" class="gauge" :class="{ primary: g.primary }">
                  <span class="gauge-label">{{ g.label }}</span>
                  <div class="gauge-bar">
                    <i :style="{ width: `${Math.min(100, g.value * 100)}%` }"></i>
                  </div>
                  <b>{{ (g.value * 100).toFixed(0) }}%</b>
                </div>
              </div>

              <div class="abuse-strip" :class="abuseTone">
                <span class="abuse-icon">⚑</span>
                <span>{{ abuseTitle }}</span>
                <b>{{ (result.abuse_score * 100).toFixed(0) }}%</b>
              </div>

              <div class="reasons">
                <div v-if="result.abuse_reasons.length" class="reason-group abuse">
                  <h3>辱骂命中依据</h3>
                  <ul><li v-for="r in result.abuse_reasons" :key="r">{{ r }}</li></ul>
                </div>
                <div class="reason-group">
                  <h3>垃圾命中依据</h3>
                  <ul v-if="result.reasons.length">
                    <li v-for="r in result.reasons" :key="r">{{ r }}</li>
                  </ul>
                  <p v-else class="muted">未命中营销类规则</p>
                </div>
              </div>

              <div class="feedback">
                <span>{{ feedbackSent ? '已记录，感谢反馈' : '判断有误？' }}</span>
                <template v-if="!feedbackSent">
                  <button type="button" @click="submitFeedback('normal')">应放行</button>
                  <button type="button" @click="submitFeedback('spam')">应拦截</button>
                  <button type="button" @click="submitFeedback('normal', 'abuse')">是辱骂</button>
                </template>
              </div>
            </div>
          </div>
        </section>

        <section v-else key="pipeline" class="stage">
          <div class="composer glass">
            <PipelinePanel
              v-if="pipeline"
              :pipeline="pipeline"
              @applied="(config) => (pipeline = config)"
            />
            <p v-else class="muted">流水线配置加载中…</p>
          </div>
        </section>
      </Transition>

      <section class="metrics">
        <div v-for="m in metrics" :key="m.k" class="metric glass">
          <small>{{ m.k }}</small>
          <strong>{{ m.v.toLocaleString() }}</strong>
        </div>
      </section>
    </main>
  </div>
</template>
