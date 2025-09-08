// src/utils/safeJson.js
export function safeJsonParse(value, fallback = null) {
  if (value == null) return fallback;          // undefined/null
  if (typeof value !== 'string') return value; // 이미 객체면 그대로 반환
  try { return JSON.parse(value); }
  catch { return fallback; }
}
