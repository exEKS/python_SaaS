export default function HourlyModal({ open, regionName, hourly, loading, note, onClose }) {
  if (!open) return null

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div>
            <div style={{ fontSize: 16, color: '#e6edf3', fontWeight: 600 }}>{regionName}</div>
            <div style={{ fontSize: 12, color: '#8b949e' }}>Погодинний прогноз (00:00-23:00)</div>
          </div>
          <button onClick={onClose} style={closeStyle}>×</button>
        </div>

        {loading ? (
          <div style={{ color: '#8b949e', fontSize: 13 }}>Завантаження погодинного прогнозу…</div>
        ) : (
          <div style={{ maxHeight: 360, overflowY: 'auto', paddingRight: 4 }}>
            {hourly.map((row) => {
              const pct = Math.round((row.alarm_prob ?? 0) * 100)
              return (
                <div key={row.hour} style={{ display: 'grid', gridTemplateColumns: '56px 1fr 44px', gap: 10, alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ color: '#8b949e', fontSize: 12 }}>{String(row.hour).padStart(2, '0')}:00</span>
                  <div style={{ height: 7, background: '#21262d', borderRadius: 999 }}>
                    <div style={{ height: '100%', width: `${pct}%`, borderRadius: 999, background: '#f85149' }} />
                  </div>
                  <span style={{ color: '#e6edf3', fontSize: 12, textAlign: 'right' }}>{pct}%</span>
                </div>
              )
            })}
          </div>
        )}

        {note && <div style={{ marginTop: 10, fontSize: 11, color: '#8b949e' }}>{note}</div>}
      </div>
    </div>
  )
}

const overlayStyle = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0,0,0,0.5)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 90,
}

const modalStyle = {
  width: 'min(680px, 92vw)',
  background: '#0d1117',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 10,
  padding: 14,
}

const closeStyle = {
  background: 'none',
  border: 'none',
  color: '#8b949e',
  fontSize: 20,
  cursor: 'pointer',
  lineHeight: 1,
}
