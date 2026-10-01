<script setup lang="ts">
import { computed, ref } from 'vue'
import { api } from '../api'
import type { PipelineConfig } from '../api'
import PipelineFlow from './PipelineFlow.vue'

const props = defineProps<{ pipeline: PipelineConfig }>()
const emit = defineEmits<{ applied: [config: PipelineConfig] }>()

/** 把 'rules.spam.gambling' 这类路径翻译成嵌套补丁对象。 */
function toPatch(path: string, value: number): Record<string, unknown> {
  const parts = path.split('.')
  if (parts[0] === 'thresholds') return { thresholds: { [parts[1]]: value } }
  if (parts[0] === 'fusion') return { fusion: { [parts[1]]: value } }
  if (parts[0] === 'rules') {
    return { rules: { [parts[1]]: { [parts[2]]: value } } }
  }
  // models.<id>.features.<name>
  return { models: { [parts[1]]: { features: { [parts[3]]: value } } } }
}

const saving = ref(false)
const message = ref('')
const errorText = ref('')

const touched = ref(0)

async function onEdit(field: string, value: number) {
  if (Number.isNaN(value)) return
  saving.value = true
  errorText.value = ''
  message.value = ''
  try {
    const result = await api.updatePipeline(toPatch(field, value))
    emit('applied', result.pipeline as PipelineConfig)
    touched.value += 1
    message.value = '已生效并保存'
    setTimeout(() => (message.value = ''), 2200)
  } catch (e) {
    errorText.value = e instanceof Error ? e.message : '保存失败'
  } finally {
    saving.value = false
  }
}

async function revert() {
  saving.value = true
  try {
    const fresh = await api.pipeline()
    emit('applied', fresh)
    message.value = '已重置为保存的配置'
    setTimeout(() => (message.value = ''), 2200)
  } finally {
    saving.value = false
  }
}

const isDefault = computed(() => touched.value === 0)
</script>

<template>
  <div class="panel">
    <header class="panel-head">
      <div>
        <h2>检测流水线 · 权重配置</h2>
        <p class="lead">
          框内数值可直接编辑，保存后立即生效并写入 <code>backend/pipeline.json</code>。
          <span class="warn">橙色边框的特征权重需重新训练模型才会生效</span>
        </p>
      </div>
      <div class="panel-actions">
        <span v-if="saving" class="saving">保存中…</span>
        <span v-else-if="message" class="ok">{{ message }}</span>
        <span v-else-if="errorText" class="err">{{ errorText }}</span>
        <button type="button" class="ghost" :disabled="saving || isDefault" @click="revert">重置</button>
      </div>
    </header>

    <PipelineFlow :pipeline="pipeline" @edit="onEdit" />
  </div>
</template>

<style scoped>
.panel { display: flex; flex-direction: column; gap: 20px; }
.panel-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.panel-head h2 { margin: 0 0 6px; font-size: 16px; }
.lead { margin: 0; font-size: 12px; color: var(--muted); line-height: 1.7; }
.warn { display: inline-block; margin-left: 6px; color: var(--amber); }
.panel-actions { display: flex; align-items: center; gap: 10px; }
.saving { font-size: 12px; color: var(--muted); }
.ok { font-size: 12px; color: var(--green); }
.err { font-size: 12px; color: var(--red); max-width: 320px; }
.ghost {
  padding: 7px 16px; border-radius: 9px; cursor: pointer;
  border: 1px solid var(--border); background: rgba(148, 163, 184, 0.08);
  color: var(--muted); font-size: 12px; transition: all 0.15s ease;
}
.ghost:disabled { opacity: 0.4; cursor: not-allowed; }
.ghost:not(:disabled):hover { color: var(--ink); border-color: rgba(56, 189, 248, 0.4); }
</style>


