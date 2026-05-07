import { useRef, useState } from 'react'
import { REGIONS } from './regions.js'
import { probToColor } from './colors.js'

const STROKE_DEFAULT  = 'rgba(13,17,23,0.7)'
const STROKE_HOVER    = '#58a6ff'
const STROKE_SELECTED = '#e6edf3'

export default function UkraineMap({ data, loadingById = {}, selected, onSelect }) {
  const svgRef = useRef(null)
  const [hovered, setHovered]   = useState(null)
  const [tooltip, setTooltip]   = useState(null) // { x, y, id }

  function handleMouseMove(e, id) {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, id })
    setHovered(id)
  }

  function handleMouseLeave() {
    setTooltip(null)
    setHovered(null)
  }

  const prob   = tooltip ? data[tooltip.id] : undefined
  const isLoading = tooltip ? !!loadingById[tooltip.id] : false
  const region = tooltip ? REGIONS[tooltip.id] : null

  return (
    <div style={{ position: 'relative' }}>
      <svg
        ref={svgRef}
        viewBox="0 0 612.47 408.02"
        width="100%"
        style={{ display: 'block' }}
      >
        {Object.entries(REGIONS).map(([id, r]) => {
          const p        = data[id]
          const fill     = r.occupied ? '#1c2333' : loadingById[id] ? '#1f6feb' : probToColor(p)
          const isHov    = hovered   === id
          const isSel    = selected  === id
          const stroke   = isSel ? STROKE_SELECTED : isHov ? STROKE_HOVER : STROKE_DEFAULT
          const sw       = isSel ? 1.2 : isHov ? 0.9 : 0.4

          return (
            <path
              key={id}
              d={r.d}
              fill={fill}
              stroke={stroke}
              strokeWidth={sw}
              style={{
                cursor:     r.occupied ? 'default' : 'pointer',
                opacity:    r.occupied ? 0.35 : 1,
                transition: 'fill 0.35s ease',
              }}
              onMouseMove={r.occupied ? undefined : e => handleMouseMove(e, id)}
              onMouseLeave={handleMouseLeave}
              onClick={r.occupied ? undefined : () => onSelect(id === selected ? null : id)}
            />
          )
        })}
      </svg>

      {tooltip && region && (
        <Tooltip
          x={tooltip.x}
          y={tooltip.y}
          name={region.ukName}
          prob={prob}
          isLoading={isLoading}
        />
      )}
    </div>
  )
}

function Tooltip({ x, y, name, prob, isLoading }) {
  const color = probToColor(prob)
  const pct   = isLoading ? '…' : prob !== null && prob !== undefined ? `${Math.round(prob * 100)}%` : '—'
  return (
    <div style={{
      position:      'absolute',
      left:          x + 14,
      top:           y - 12,
      pointerEvents: 'none',
      background:    '#161b22',
      border:        '0.5px solid rgba(255,255,255,0.12)',
      borderRadius:  8,
      padding:       '8px 12px',
      minWidth:      140,
      zIndex:        30,
    }}>
      <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 2 }}>{name}</div>
      <div style={{ fontSize: 22, fontWeight: 500, color, lineHeight: 1 }}>{pct}</div>
      <div style={{ fontSize: 11, color: '#6e7681', marginTop: 2 }}>
        {isLoading ? 'завантаження…' : 'ймовірність тривоги'}
      </div>
      {prob !== null && prob !== undefined && (
        <div style={{ marginTop: 6, height: 3, borderRadius: 2, background: '#30363d' }}>
          <div style={{
            height: '100%',
            width:  `${Math.round(prob * 100)}%`,
            background: color,
            borderRadius: 2,
          }} />
        </div>
      )}
    </div>
  )
}
