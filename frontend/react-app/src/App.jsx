import { useState } from 'react'
import UkraineMap    from './UkraineMap.jsx'
import RegionDetail  from './RegionDetail.jsx'
import { useAlarmData } from './useAlarmData.js'

const TODAY = new Date().toISOString().split('T')[0]

export default function App() {
  const [apiBase, setApiBase] = useState('/api')   // proxied by Vite → http://127.0.0.1:8000
  const [date,    setDate]    = useState(TODAY)
  const [selected, setSelected] = useState(null)

  const { data, loading, status, updatedAt, fetchAll, useDemoData } = useAlarmData()

  function handleFetch() {
    fetchAll(apiBase, date)
  }

  const statusType = status?.type
  const isLive = statusType === 'live'
  const isDemo = statusType === 'demo'

  return (
    <div style={{
      minHeight:  '100vh',
      background: '#0d1117',
      padding:    '24px 28px',
      maxWidth:   900,
      margin:     '0 auto',
    }}>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 500, color: '#e6edf3', letterSpacing: '-0.3px' }}>
            WarWatch
          </h1>
          <p style={{ fontSize: 12, color: '#6e7681', marginTop: 2 }}>
            Прогноз ймовірності повітряних тривог · 24 год
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {updatedAt && (
            <span style={{ fontSize: 11, color: '#6e7681' }}>
              {updatedAt.toLocaleTimeString('uk', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <span style={{
            fontSize:     11,
            fontWeight:   500,
            padding:      '3px 8px',
            borderRadius: 4,
            background:   isLive ? 'rgba(63,185,80,0.15)' : isDemo ? 'rgba(210,153,34,0.15)' : 'rgba(248,81,73,0.15)',
            color:        isLive ? '#3fb950' : isDemo ? '#d29922' : '#f85149',
          }}>
            {isLive ? '● live' : isDemo ? '● демо' : '● no data'}
          </span>
        </div>
      </div>

      {/* ── Controls ── */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 16 }}>
        <input
          type="text"
          value={apiBase}
          onChange={e => setApiBase(e.target.value)}
          placeholder="/api  або  http://127.0.0.1:8000"
          style={inputStyle}
        />
        <input
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
          style={{ ...inputStyle, width: 'auto' }}
        />
        <button onClick={handleFetch} disabled={loading} style={btnStyle(loading)}>
          {loading ? 'Завантаження…' : 'Оновити прогноз'}
        </button>
        <button onClick={useDemoData} disabled={loading} style={btnStyle(loading)}>
          Демо режим
        </button>
      </div>

      {/* ── Map ── */}
      <UkraineMap data={data} selected={selected} onSelect={setSelected} />

      {/* ── Legend ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 10, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: '#6e7681' }}>Ризик тривоги:</span>
        <div>
          <div style={{
            width: 160, height: 7, borderRadius: 4,
            background: 'linear-gradient(to right, rgb(46,160,67), rgb(187,128,9), rgb(188,90,32), rgb(218,54,51))',
          }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', width: 160, fontSize: 10, color: '#6e7681', marginTop: 2 }}>
            <span>0%</span><span>35%</span><span>65%</span><span>100%</span>
          </div>
        </div>
        {[['Низький','#3fb950'],['Середній','#d29922'],['Високий','#db6d28'],['Критичний','#f85149']].map(([l,c]) => (
          <span key={l} style={{
            fontSize: 11, padding: '2px 7px', borderRadius: 3,
            background: c + '22', color: c, fontWeight: 500,
          }}>{l}</span>
        ))}
      </div>

      {/* ── Region detail ── */}
      <RegionDetail id={selected} data={data} onClose={() => setSelected(null)} />

      {/* ── Status bar ── */}
      {status && (
        <div style={{ marginTop: 10, fontSize: 11, color: status.type === 'error' ? '#f85149' : '#6e7681' }}>
          {status.text}
        </div>
      )}
    </div>
  )
}

const inputStyle = {
  fontSize:   13,
  padding:    '6px 10px',
  border:     '0.5px solid rgba(255,255,255,0.12)',
  borderRadius: 6,
  background: '#161b22',
  color:      '#e6edf3',
  outline:    'none',
  width:      200,
}

const btnStyle = (disabled) => ({
  fontSize:     13,
  padding:      '6px 16px',
  border:       '0.5px solid rgba(255,255,255,0.14)',
  borderRadius: 6,
  background:   disabled ? '#161b22' : '#1c2333',
  color:        disabled ? '#6e7681' : '#e6edf3',
  cursor:       disabled ? 'not-allowed' : 'pointer',
})
