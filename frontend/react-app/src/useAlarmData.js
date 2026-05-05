import { useState, useCallback } from 'react'
import { REGIONS, SKIP_FETCH } from './regions.js'

// Demo probabilities shown before first real fetch
const DEMO = {
  "UA-63": 0.91, "UA-14": 0.87, "UA-09": 0.83, "UA-59": 0.72,
  "UA-23": 0.68, "UA-65": 0.75, "UA-12": 0.55, "UA-53": 0.44,
  "UA-48": 0.52, "UA-51": 0.38, "UA-32": 0.35, "UA-30": 0.35,
  "UA-74": 0.42, "UA-18": 0.28, "UA-71": 0.31, "UA-05": 0.22,
  "UA-35": 0.30, "UA-68": 0.18, "UA-56": 0.14, "UA-07": 0.12,
  "UA-46": 0.10, "UA-61": 0.12, "UA-26": 0.11, "UA-21": 0.08,
  "UA-77": 0.13,
}

export function useAlarmData() {
  const [data, setData]       = useState({})
  const [loading, setLoading] = useState(false)
  const [status, setStatus]   = useState(null)   // { type: 'demo'|'live'|'error', text: string }
  const [updatedAt, setUpdatedAt] = useState(null)

  const fetchAll = useCallback(async (apiBase, date) => {
    setLoading(true)
    setStatus(null)

    const base = apiBase.replace(/\/$/, '')
    const ids  = Object.keys(REGIONS).filter(id => !SKIP_FETCH.has(id))

    const results = await Promise.allSettled(
      ids.map(async (id) => {
        const region = REGIONS[id]
        if (!region.apiName) return null
        const url = `${base}/predict?region=${encodeURIComponent(region.apiName)}&date=${date}`
        const res = await fetch(url, { signal: AbortSignal.timeout(20000) })
        if (!res.ok) throw new Error(`${res.status}`)
        const json = await res.json()
        return { id, prob: json.alarm_prob ?? null }
      })
    )

    const newData = {}
    let ok = 0
    results.forEach(r => {
      if (r.status === 'fulfilled' && r.value) {
        newData[r.value.id] = r.value.prob
        ok++
      }
    })

    if (ok === 0) {
      setData({})
      setStatus({ type: 'error', text: 'API недоступне — live-дані не завантажено' })
    } else {
      setData(newData)
      setUpdatedAt(new Date())
      setStatus({ type: 'live', text: `${ok} з ${ids.length} областей` })
    }
    setLoading(false)
  }, [])

  const useDemoData = useCallback(() => {
    setData(DEMO)
    setUpdatedAt(new Date())
    setStatus({ type: 'demo', text: 'Увімкнено демо-режим (фіксовані прикладні значення)' })
  }, [])

  return { data, loading, status, updatedAt, fetchAll, useDemoData }
}
