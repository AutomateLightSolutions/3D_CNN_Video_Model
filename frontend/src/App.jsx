import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import MatchList          from './pages/MatchList.jsx'
import Annotate           from './pages/Annotate.jsx'
import FeatureExtraction  from './pages/FeatureExtraction.jsx'
import Predict            from './pages/Predict.jsx'
import MergeCalibration   from './pages/MergeCalibration.jsx'
import TrainR3D           from './pages/TrainR3D.jsx'
import TrainVideoMAE      from './pages/TrainVideoMAE.jsx'
import TrainSlowFast      from './pages/TrainSlowFast.jsx'
import TrainRF            from './pages/TrainRF.jsx'
import TrainMLP           from './pages/TrainMLP.jsx'
import TrainHybrid        from './pages/TrainHybrid.jsx'
import Compare            from './pages/Compare.jsx'
import History            from './pages/History.jsx'
import Export             from './pages/Export.jsx'

const navCls = ({ isActive }) => isActive ? 'nav-item active' : 'nav-item'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <nav className="sidebar">
          <div className="sidebar-logo">Highlight<br />Annotator</div>
          <NavLink to="/"                  className={navCls} end>Matches</NavLink>
          <NavLink to="/annotate"          className={navCls}>Annotate</NavLink>
          <NavLink to="/features"          className={navCls}>Feature Extraction</NavLink>
          <NavLink to="/predict"           className={navCls}>Predict Match</NavLink>
          <div className="nav-group-label">Deep Learning</div>
          <NavLink to="/training/r3d"      className={navCls}>R3D-18</NavLink>
          <NavLink to="/training/videomae" className={navCls}>VideoMAE</NavLink>
          <NavLink to="/training/slowfast" className={navCls}>SlowFast</NavLink>
          <div className="nav-group-label">Interpretable</div>
          <NavLink to="/training/rf"       className={navCls}>Interp + RF</NavLink>
          <NavLink to="/training/mlp"      className={navCls}>Interp + MLP</NavLink>
          <NavLink to="/training/hybrid"   className={navCls}>Interp + Deep</NavLink>
          <div className="nav-group-label">Results</div>
          <NavLink to="/compare"           className={navCls}>Compare</NavLink>
          <NavLink to="/history"           className={navCls}>History</NavLink>
          <NavLink to="/export"            className={navCls}>Export</NavLink>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/"                  element={<MatchList />} />
            <Route path="/annotate"          element={<Annotate />} />
            <Route path="/features"          element={<FeatureExtraction />} />
            <Route path="/predict"           element={<Predict />} />
            <Route path="/predict/:runId/calibrate" element={<MergeCalibration />} />
            <Route path="/training/r3d"      element={<TrainR3D />} />
            <Route path="/training/videomae" element={<TrainVideoMAE />} />
            <Route path="/training/slowfast" element={<TrainSlowFast />} />
            <Route path="/training/rf"       element={<TrainRF />} />
            <Route path="/training/mlp"      element={<TrainMLP />} />
            <Route path="/training/hybrid"   element={<TrainHybrid />} />
            <Route path="/compare"           element={<Compare />} />
            <Route path="/history"           element={<History />} />
            <Route path="/export"            element={<Export />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
