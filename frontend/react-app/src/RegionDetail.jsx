import { REGIONS } from './regions.js'
import { probToColor, riskInfo, fmtPct } from './colors.js'

export default function RegionDetail({ id, data, loadingById = {}, onClose }) {
  if (!id || !REGIONS[id]) return null

  const region = REGIONS[id]
  const prob   = data[id]
  const isLoading = !!loadingById[id]
  const color  = probToColor(prob)
  const ri     = riskInfo(prob)

  return (
    <div style={{
      marginTop:    12,
      background:   '#161b22',
      border:       '0.5px solid rgba(255,255,255,0.10)',
      borderRadius: 10,
      padding:      '14px 18px',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 500, color: '#e6edf3', marginBottom: 10 }}>
            {region.ukName}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span style={{ fontSize: 36, fontWeight: 500, color, lineHeight: 1 }}>
              {isLoading ? '…' : fmtPct(prob)}
            </span>
            <div>
              <span style={{
                display:      'inline-block',
                fontSize:     11,
                fontWeight:   500,
                padding:      '3px 8px',
                borderRadius: 4,
                background:   ri.bg,
                color:        ri.color,
              }}>
                {isLoading ? 'Завантаження' : ri.label}
              </span>
              <div style={{ fontSize: 11, color: '#6e7681', marginTop: 4 }}>
                ймовірність тривоги · 24 год
              </div>
            </div>
          </div>

          {prob !== null && prob !== undefined && (
            <div style={{ marginTop: 12, width: 260, height: 4, borderRadius: 2, background: '#21262d' }}>
              <div style={{
                height:       '100%',
                width:        `${Math.round(prob * 100)}%`,
                background:   color,
                borderRadius: 2,
                transition:   'width 0.5s ease',
              }} />
            </div>
          )}
        </div>

        <button
          onClick={onClose}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: '#6e7681', fontSize: 18, lineHeight: 1, padding: '2px 4px',
          }}
        >×</button>
      </div>
    </div>
  )
}
