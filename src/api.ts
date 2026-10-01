export type Label = 'spam' | 'normal' | 'review'

export interface FilterResult {
  text: string
  label: Label
  score: number
  reasons: string[]
  matched_rules: string[]
  categories: string[]
  model_version: string
  rule_score: number
  model_score: number | null
  abuse_score: number
  abuse_label: Label
  abuse_suspicious: boolean
  abuse_reasons: string[]
  abuse_categories: string[]
}

export interface Stats {
  total: number
  spam: number
  review: number
  abuse: number
  feedback: number
  average_score: number
}

/** 流水线配置：模型、特征权重、规则权重、融合系数与阈值。 */
export interface FeatureSpec {
  name: string
  label: string
  analyzer: string
  ngram_range: [number, number]
  max_features: number
  weight: number
  note: string
}

export interface ModelSpec {
  id: string
  label: string
  dimension: string
  path: string
  estimator: string
  trained_rows: number
  note: string
  features: FeatureSpec[]
}

export interface RuleSpec {
  id: string
  category: string
  reason: string
  pattern: string
  weight: number
}

export interface PipelineConfig {
  version: string
  normalization: string[]
  models: ModelSpec[]
  rules: { spam: RuleSpec[]; abuse: RuleSpec[] }
  fusion: {
    strategy: string
    rule_boost_factor: number
    aggregation: string
    long_text_bonus: number
    long_text_threshold: number
  }
  thresholds: Record<string, number>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    // 后端对配置校验失败会返回 422 + detail，直接把原因透出到界面
    let detail = `请求失败（${response.status}）`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      /* 非 JSON 响应沿用默认文案 */
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const api = {
  filter: (text: string) =>
    request<FilterResult>('/api/filter', { method: 'POST', body: JSON.stringify({ text }) }),
  batch: (texts: string[]) =>
    request<{ results: FilterResult[] }>('/api/filter/batch', {
      method: 'POST',
      body: JSON.stringify({ texts }),
    }),
  stats: () => request<Stats>('/api/stats'),
  pipeline: () => request<PipelineConfig>('/api/pipeline'),
  updatePipeline: (patch: Record<string, unknown>) =>
    request<{ status: string; requires_retrain: string[]; pipeline: PipelineConfig }>('/api/pipeline', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),
  feedback: (payload: {
    text: string
    predicted_label: string
    corrected_label: 'spam' | 'normal'
    corrected_abuse_label?: 'abuse' | 'normal'
  }) => request<{ id: number; status: string }>('/api/feedback', { method: 'POST', body: JSON.stringify(payload) }),
}
