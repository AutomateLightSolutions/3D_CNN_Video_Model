import { useState } from 'react'

export const CLASSES = [
  { key: 'try',         label: 'Try',         shortcut: '1' },
  { key: 'conversion',  label: 'Conversion',  shortcut: '2' },
  { key: 'kick_off',    label: 'Kick Off',    shortcut: '3' },
  { key: 'penalty_kick',label: 'Penalty Kick',shortcut: '4' },
  { key: 'drop_goal',   label: 'Drop Goal',   shortcut: '5' },
  { key: 'near_try',    label: 'Near Try',    shortcut: '6' },
  { key: 'scrum',       label: 'Scrum',       shortcut: '7' },
  { key: 'lineout',     label: 'Lineout',     shortcut: '8' },
  { key: 'touch_kick',  label: 'Touch Kick',  shortcut: '9' },
  { key: 'intercept',   label: 'Intercept',   shortcut: 'q' },
  { key: 'tackle',      label: 'Tackle',      shortcut: 'w' },
  { key: 'ruck',        label: 'Ruck',        shortcut: 'e' },
  { key: 'maul',        label: 'Maul',        shortcut: 'r' },
  { key: 'red_card',    label: 'Red Card',    shortcut: 't' },
  { key: 'yellow_card', label: 'Yellow Card', shortcut: 'y' },
  { key: 'turnover',    label: 'Turnover',    shortcut: 'u' },
  { key: 'line_break',  label: 'Line Break',  shortcut: 'i' },
  { key: 'cross_kick',  label: 'Cross Kick',  shortcut: 'o' },
  { key: 'penalty',     label: 'Penalty',     shortcut: 'p' },
  { key: 'knock_on',    label: 'Knock On',    shortcut: '[' },
  { key: 'replay',      label: 'Replay',      shortcut: ']' },
  { key: 'tmo_review',  label: 'TMO Review',  shortcut: '\\' },
  { key: 'normal_play', label: 'Normal Play', shortcut: '`' },
]

export const SHORTCUT_MAP = Object.fromEntries(CLASSES.map(c => [c.shortcut, c.key]))

export default function ClassSelector({ selected, onChange }) {
  const [showLegend, setShowLegend] = useState(false)

  return (
    <div>
      <label>Event Class</label>
      <div className="class-grid">
        {CLASSES.map(({ key, label, shortcut }) => (
          <button
            key={key}
            className={`class-btn${selected === key ? ' selected' : ''}`}
            onClick={() => onChange(key)}
            type="button"
          >
            <span className="shortcut">[{shortcut}]</span>
            {label}
          </button>
        ))}
      </div>

      <div style={{ marginTop: 8 }}>
        <button
          type="button"
          onClick={() => setShowLegend(s => !s)}
          style={{
            background: 'none', border: 'none', padding: 0,
            color: 'var(--text-2, #888)', fontSize: 11,
            cursor: 'pointer', textDecoration: 'underline',
          }}
        >
          {showLegend ? 'hide shortcuts' : 'show shortcuts'}
        </button>

        {showLegend && (
          <div style={{
            marginTop: 6, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '2px 12px', fontSize: 11, color: 'var(--text-2, #888)',
          }}>
            {CLASSES.map(({ key, label, shortcut }) => (
              <span key={key}>
                <kbd style={{
                  background: 'var(--surface-2, #2a2a2a)',
                  border: '1px solid var(--border, #444)',
                  borderRadius: 3, padding: '1px 4px', fontSize: 10,
                  fontFamily: 'monospace',
                }}>{shortcut}</kbd>
                {' '}{label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
