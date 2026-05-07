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
  const [loadingById, setLoadingById] = useState({})
  const [loading, setLoading] = useState(false)
  const [status, setStatus]   = useState(null)   // { type: 'demo'|'live'|'error', text: string }
  const [updatedAt, setUpdatedAt] = useState(null)

  const fetchAll = useCallback(async (apiBase, date) => {
    setLoading(true)
    setStatus(null)

    const base = apiBase.replace(/\/$/, '')
    const ids  = Object.keys(REGIONS).filter(id => !SKIP_FETCH.has(id))
    const initLoading = Object.fromEntries(ids.map(id => [id, true]))
    setLoadingById(initLoading)

    const regionJobs = ids
      .map((id) => ({ id, region: REGIONS[id] }))
      .filter((x) => x.region?.apiName)

    let newData = {}
    let ok = 0

    // Primary path: one batch call for low-resource servers.
    try {
      const payload = {
        date,
        regions: regionJobs.map(j => j.region.apiName),
      }
      const res = await fetch(`${base}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(25000),
      })
      if (res.ok) {
        const json = await res.json()
        const byRegion = new Map((json.items || []).map(item => [item.region, item]))
        regionJobs.forEach(({ id, region }) => {
          const item = byRegion.get(region.apiName)
          if (item) {
            newData[id] = item.alarm_prob ?? null
            ok++
          }
        })
      } else {
        throw new Error(`batch:${res.status}`)
      }
    } catch {
      // Fallback path: limit concurrency to avoid choking t3.micro.
      const queue = [...regionJobs]
      const MAX_CONCURRENCY = 2
      const workers = Array.from({ length: MAX_CONCURRENCY }, async () => {
        while (queue.length) {
          const job = queue.shift()
          if (!job) break
          const { id, region } = job
          const url = `${base}/predict?region=${encodeURIComponent(region.apiName)}&date=${date}`
          try {
            const res = await fetch(url, { signal: AbortSignal.timeout(20000) })
            if (res.ok) {
              const json = await res.json()
              const prob = json.alarm_prob ?? null
              newData = { ...newData, [id]: prob }
              ok++
              setData(prev => ({ ...prev, [id]: prob }))
            }
          } finally {
            setLoadingById(prev => ({ ...prev, [id]: false }))
          }
        }
      })
      await Promise.all(workers)
    }

    if (ok > 0) {
      // In batch mode this applies final map values all at once.
      setData(newData)
      setUpdatedAt(new Date())
    }
    setLoadingById(prev => {
      const next = { ...prev }
      ids.forEach(id => {
        next[id] = false
      })
      return next
    })

    if (ok === 0) {
      setData({})
      setStatus({ type: 'error', text: 'API недоступне — live-дані не завантажено' })
    } else {
      setStatus({ type: 'live', text: `${ok} з ${ids.length} областей` })
    }
    setLoading(false)
  }, [])

  const useDemoData = useCallback(() => {
    setData(DEMO)
    setLoadingById({})
    setUpdatedAt(new Date())
    setStatus({ type: 'demo', text: 'Увімкнено демо-режим (фіксовані прикладні значення)' })
  }, [])

  return { data, loadingById, loading, status, updatedAt, fetchAll, useDemoData }
}
