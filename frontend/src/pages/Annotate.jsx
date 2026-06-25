import { useState, useEffect, useCallback, useRef } from "react";
import {
  listMatches,
  listClips,
  getNextClip,
  getClip,
  saveLabel,
  skipClip,
} from "../api/client.js";
import VideoPlayer from "../components/VideoPlayer.jsx";
import ClassSelector, {
  CLASSES,
  SHORTCUT_MAP,
} from "../components/ClassSelector.jsx";
import TimelineEditor from "../components/TimelineEditor.jsx";

const BASE_SCORES = {
  try:         1,
  goal_kick:   0.53,
  card_event:  0.55,
  lineout:     0.3,
  scrum:       0.25,
  maul:        0.2,
  kick_off:    0.15,
  tmo_replay:  0.05,
  normal_play: 0,
}

const SCORE_TABLE = [
  { key: "try",         label: "Try",          score: 1,    source: "Official — 5 pts" },
  { key: "goal_kick",   label: "Goal Kick",     score: 0.53, source: "Official — avg(conversion, penalty, drop goal)" },
  { key: "card_event",  label: "Card Event",    score: 0.55, source: "Broadcast tally — pending" },
  { key: "lineout",     label: "Lineout",       score: 0.3,  source: "Broadcast tally — pending" },
  { key: "scrum",       label: "Scrum",         score: 0.25, source: "Broadcast tally — pending" },
  { key: "maul",        label: "Maul",          score: 0.2,  source: "Broadcast tally — pending" },
  { key: "kick_off",    label: "Kick Off",      score: 0.15, source: "Broadcast tally — pending" },
  { key: "tmo_replay",  label: "TMO / Replay",  score: 0.05, source: "Broadcast tally — pending" },
  { key: "normal_play", label: "Normal Play",   score: 0,    source: "No highlight value" },
]

function useAnnotateState(clip) {
  const [eventClass, setEventClass] = useState("normal_play");
  const [tStartAdj, setTStartAdj] = useState(0);
  const [tEndAdj, setTEndAdj] = useState(0);
  const [notes, setNotes] = useState("");

  // Score is always derived from the selected event class — not editable
  const score = BASE_SCORES[eventClass] ?? 0;

  useEffect(() => {
    if (!clip) return;
    if (clip.label) {
      setEventClass(clip.label.event_class);
      setTStartAdj(clip.label.t_start_adjusted);
      setTEndAdj(clip.label.t_end_adjusted);
      setNotes(clip.label.notes || "");
    } else {
      setEventClass("normal_play");
      setTStartAdj(clip.t_start);
      setTEndAdj(clip.t_end);
      setNotes("");
    }
  }, [clip?.id]);

  const onTimelineChange = useCallback(({ tStart, tEnd }) => {
    setTStartAdj(tStart);
    setTEndAdj(tEnd);
  }, []);

  return {
    eventClass,
    setEventClass,
    score,
    tStartAdj,
    setTStartAdj,
    tEndAdj,
    setTEndAdj,
    notes,
    setNotes,
    onTimelineChange,
  };
}

function scoreColor(score) {
  if (score >= 0.7) return 'var(--green)';
  if (score >= 0.3) return '#f59e0b';
  return 'var(--text-2)';
}

function scoreRangeClass(score) {
  if (score >= 0.7)  return 'score-range score-range-6';
  if (score >= 0.4)  return 'score-range score-range-3';
  if (score >= 0.15) return 'score-range score-range-2';
  return 'score-range score-range-0';
}

function BaseScoreDisplay({ score, eventClass }) {
  const color = scoreColor(score);
  const label = eventClass.replaceAll('_', ' ');
  return (
    <div className="card">
      <h2 style={{ marginBottom: 12 }}>Base Score</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 10 }}>
        <div style={{ fontSize: 36, fontWeight: 700, color }}>
          {score.toFixed(2)}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>
          Base score for <strong style={{ color: 'var(--text-1)' }}>{label}</strong>
          <br />
          Full VisualScore computed at training time
        </div>
      </div>
      <div style={{ height: 8, borderRadius: 4, background: 'var(--surface-2)' }}>
        <div style={{
          height: 8, borderRadius: 4,
          width: `${score * 100}%`,
          background: color,
          transition: 'width 0.2s ease, background 0.2s ease',
        }} />
      </div>
    </div>
  )
}

export default function Annotate() {
  const [matches, setMatches] = useState([]);
  const [matchId, setMatchId] = useState("");
  const [windowFilter, setWindowFilter] = useState("all");
  const [clip, setClip] = useState(null);
  const [matchDuration, setMatchDuration] = useState(0);
  const [history, setHistory] = useState([]);
  const [totalUnlabeled, setTotalUnlabeled] = useState(0);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [labeledClips, setLabeledClips] = useState([]);

  const ann = useAnnotateState(clip);
  const notesRef = useRef(null);

  useEffect(() => {
    listMatches()
      .then(setMatches)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!matchId) return;
    const m = matches.find((m) => String(m.id) === String(matchId));
    if (m) setMatchDuration(m.duration_seconds || 0);
  }, [matchId, matches]);

  const windowSize =
    windowFilter !== "all" ? parseInt(windowFilter) : undefined;

  const refreshCounts = useCallback(async () => {
    if (!matchId) return;
    try {
      const [unlabeled, labeled] = await Promise.all([
        listClips({ matchId, status: "unlabeled", windowSize }),
        listClips({ matchId, status: "labeled", windowSize }),
      ]);
      setTotalUnlabeled(unlabeled.length);
      setLabeledClips(labeled);
    } catch {}
  }, [matchId, windowSize]);

  const loadNext = useCallback(
    async (afterId) => {
      if (!matchId) return;
      try {
        const params = { matchId, windowSize };
        if (afterId) params.afterClipId = afterId;
        const next = await getNextClip(params);
        if (next) {
          setClip(next);
          setPosition((p) => p + 1);
        } else {
          setClip(null);
        }
      } catch (e) {
        setError(e.message);
      }
    },
    [matchId, windowSize],
  );

  const loadFirst = useCallback(async () => {
    if (!matchId) return;
    setHistory([]);
    setPosition(0);
    setError("");
    try {
      const next = await getNextClip({ matchId, windowSize });
      setClip(next || null);
      if (next) setPosition(1);
    } catch (e) {
      setError(e.message);
    }
    refreshCounts();
  }, [matchId, windowSize, refreshCounts]);

  useEffect(() => {
    setLabeledClips([]);
    loadFirst();
  }, [matchId, windowFilter]);

  const handleSave = useCallback(async () => {
    if (!clip) return;
    setSaving(true);
    setError("");
    try {
      await saveLabel({
        clip_id: clip.id,
        event_class: ann.eventClass,
        highlight_score: ann.score,
        t_start_adjusted: ann.tStartAdj,
        t_end_adjusted: ann.tEndAdj,
        notes: ann.notes || null,
      });
      setHistory((h) => [...h, clip.id]);
      await loadNext(clip.id);
      refreshCounts();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }, [clip, ann, loadNext, refreshCounts]);

  const handleSkip = useCallback(async () => {
    if (!clip) return;
    try {
      await skipClip(clip.id);
      setHistory((h) => [...h, clip.id]);
      await loadNext(clip.id);
      refreshCounts();
    } catch (e) {
      setError(e.message);
    }
  }, [clip, loadNext, refreshCounts]);

  const handlePrevious = useCallback(async () => {
    if (history.length === 0) return;
    const prevId = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    try {
      const c = await getClip(prevId);
      setClip(c);
      setPosition((p) => Math.max(1, p - 1));
    } catch (e) {
      setError(e.message);
    }
  }, [history]);

  // Keyboard shortcuts — only fire when not typing in an input/textarea/select
  useEffect(() => {
    const handler = (e) => {
      const tag = e.target.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;

      if (e.code === "Enter") {
        e.preventDefault();
        handleSave();
        return;
      }
      if (e.code === "KeyS") {
        e.preventDefault();
        handleSkip();
        return;
      }
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        handlePrevious();
        return;
      }

      const classKey = SHORTCUT_MAP[e.key];
      if (classKey) {
        e.preventDefault();
        ann.setEventClass(classKey);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleSave, handleSkip, handlePrevious, ann]);

  const videoSrc = clip ? clip.clip_url : null;

  return (
    <div>
      <h1>Annotate</h1>
      {error && <div className="error-box">{error}</div>}

      <div className="annotate-layout">
        {/* Left: video */}
        <div className="annotate-left">
          <div className="annotate-header">
            <select
              value={matchId}
              onChange={(e) => setMatchId(e.target.value)}
            >
              <option value="">— Select match —</option>
              {matches.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <select
              value={windowFilter}
              onChange={(e) => setWindowFilter(e.target.value)}
            >
              <option value="all">All windows</option>
              <option value="8">8s</option>
              <option value="16">16s</option>
              <option value="32">32s</option>
            </select>
            <span className="progress-counter">
              Clip {position} · {totalUnlabeled} unlabeled
            </span>
          </div>

          {matchId && labeledClips.length > 0 && (
            <div className="labeled-browser">
              <span className="labeled-browser-label">Edit labeled clip:</span>
              <select
                value=""
                onChange={async (e) => {
                  const id = parseInt(e.target.value);
                  if (!id) return;
                  try {
                    const c = await getClip(id);
                    setClip(c);
                  } catch (err) {
                    setError(err.message);
                  }
                }}
              >
                <option value="">— pick a clip —</option>
                {labeledClips.map((c) => (
                  <option key={c.id} value={c.id}>
                    #{c.id} · {c.window_size}s · {c.t_start.toFixed(1)}s–
                    {c.t_end.toFixed(1)}s
                    {c.label
                      ? ` · ${c.label.event_class} (${c.label.highlight_score.toFixed(2)})`
                      : ""}
                  </option>
                ))}
              </select>
            </div>
          )}

          {!matchId && (
            <div className="empty-state card">
              <p>Select a match to start annotating.</p>
            </div>
          )}
          {matchId && !clip && (
            <div className="empty-state card">
              <p>No unlabeled clips left for this match.</p>
            </div>
          )}
          {clip && videoSrc && (
            <div style={{ margin: '-4px 0' }}>
              <VideoPlayer src={videoSrc} />
            </div>
          )}
          {clip && (
            <div
              className="card"
              style={{
                fontSize: 12,
                color: "var(--text-1)",
                padding: "10px 14px",
              }}
            >
              <div>
                Clip #{clip.id} · {clip.window_size}s window ·{" "}
                {clip.t_start.toFixed(1)}s–{clip.t_end.toFixed(1)}s
              </div>
              <div style={{ marginTop: 2 }}>
                Status:{" "}
                <span
                  className={`badge badge-${clip.status === "labeled" ? "green" : "gray"}`}
                >
                  {clip.status}
                </span>
              </div>
            </div>
          )}

          <div className="card score-guide">
            <h3 className="score-guide-title">Base Score Reference</h3>
            <table className="score-guide-table">
              <thead>
                <tr>
                  <th>Event Class</th>
                  <th>Base Score</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {SCORE_TABLE.map(({ key, label, score, source }) => (
                  <tr key={key}>
                    <td style={{ fontWeight: 500 }}>{label}</td>
                    <td className={scoreRangeClass(score)}>{score.toFixed(2)}</td>
                    <td style={{ fontSize: 11, color: 'var(--text-2)' }}>{source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 8 }}>
              VisualScore = 0.60 × BaseScore + 0.40 × OpticalFlow — computed at training
            </p>
          </div>
        </div>

        {/* Right: annotation controls */}
        <div className="annotate-right">
          {clip ? (
            <>
              <div className="card">
                <ClassSelector
                  selected={ann.eventClass}
                  onChange={ann.setEventClass}
                />
              </div>

              <BaseScoreDisplay score={ann.score} eventClass={ann.eventClass} />

              <div className="card">
                <h2>Timeline</h2>
                <TimelineEditor
                  matchDuration={matchDuration}
                  tStart={ann.tStartAdj}
                  tEnd={ann.tEndAdj}
                  clipStart={clip.t_start}
                  clipEnd={clip.t_end}
                  onChange={ann.onTimelineChange}
                />
              </div>

              <div className="card">
                <label>Notes (optional)</label>
                <textarea
                  ref={notesRef}
                  value={ann.notes}
                  onChange={(e) => ann.setNotes(e.target.value)}
                  placeholder="Optional notes…"
                  rows={3}
                  style={{ width: "100%" }}
                />
              </div>

              <div className="annotate-actions">
                <button
                  className="btn btn-secondary"
                  onClick={handlePrevious}
                  disabled={history.length === 0}
                >
                  ← Previous [←]
                </button>
                <button className="btn btn-secondary" onClick={handleSkip}>
                  Skip [S]
                </button>
                <button
                  className="btn btn-success"
                  onClick={handleSave}
                  disabled={saving}
                  style={{ marginLeft: "auto" }}
                >
                  {saving ? "Saving…" : "Save & Next [Enter]"}
                </button>
              </div>
            </>
          ) : (
            <div className="card empty-state">
              <p>
                {matchId ? "All clips labeled!" : "Select a match to begin."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
