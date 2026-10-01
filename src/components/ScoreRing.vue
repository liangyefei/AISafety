<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ value: number; tone: string }>()

const RADIUS = 34
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

const offset = computed(() => CIRCUMFERENCE * (1 - Math.min(1, Math.max(0, props.value))))
const percent = computed(() => Math.round(props.value * 100))
</script>

<template>
  <div class="ring" :class="tone">
    <svg viewBox="0 0 80 80" aria-hidden="true">
      <circle class="ring-track" cx="40" cy="40" :r="RADIUS" />
      <circle
        class="ring-value"
        cx="40"
        cy="40"
        :r="RADIUS"
        :stroke-dasharray="CIRCUMFERENCE"
        :stroke-dashoffset="offset"
      />
    </svg>
    <div class="ring-label">
      <strong>{{ percent }}</strong>
      <small>%</small>
    </div>
  </div>
</template>

<style scoped>
.ring { position: relative; width: 78px; height: 78px; flex-shrink: 0; }
.ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.ring-track { fill: none; stroke: rgba(148, 163, 184, 0.18); stroke-width: 7; }
.ring-value {
  fill: none;
  stroke-width: 7;
  stroke-linecap: round;
  transition: stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.ring.normal .ring-value { stroke: #34d399; }
.ring.review .ring-value { stroke: #fbbf24; }
.ring.spam .ring-value { stroke: #f87171; }
.ring-label {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 1px;
}
.ring-label strong { font-size: 21px; font-weight: 800; }
.ring-label small { font-size: 10px; opacity: 0.6; }
</style>
