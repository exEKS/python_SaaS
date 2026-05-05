// Maps 0–1 probability → rgb color (green → yellow → orange → red)
export function probToColor(p) {
  if (p === null || p === undefined) return '#30363d'
  const stops = [
    [0,    [46, 160, 67]],
    [0.35, [187, 128, 9]],
    [0.65, [188, 90, 32]],
    [1.0,  [218, 54, 51]],
  ]
  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (p >= stops[i][0] && p <= stops[i + 1][0]) {
      lo = stops[i]; hi = stops[i + 1]; break
    }
  }
  const t = (p - lo[0]) / (hi[0] - lo[0])
  const c = lo[1].map((v, i) => Math.round(v + t * (hi[1][i] - v)))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export function riskInfo(p) {
  if (p === null || p === undefined) return { label: 'Немає даних', color: '#8b949e', bg: '#21262d' }
  if (p < 0.35) return { label: 'Низький',    color: '#3fb950', bg: 'rgba(63,185,80,0.12)'  }
  if (p < 0.65) return { label: 'Середній',   color: '#d29922', bg: 'rgba(210,153,34,0.12)' }
  if (p < 0.80) return { label: 'Високий',    color: '#db6d28', bg: 'rgba(219,109,40,0.12)' }
  return               { label: 'Критичний',  color: '#f85149', bg: 'rgba(248,81,73,0.12)'  }
}

export function fmtPct(p) {
  if (p === null || p === undefined) return '—'
  return `${Math.round(p * 100)}%`
}
