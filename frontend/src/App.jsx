import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import MatchList from './pages/MatchList.jsx'
import Annotate from './pages/Annotate.jsx'
import Training from './pages/Training.jsx'
import Export from './pages/Export.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <nav className="sidebar">
          <div className="sidebar-logo">Highlight<br />Annotator</div>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Matches
          </NavLink>
          <NavLink to="/annotate" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Annotate
          </NavLink>
          <NavLink to="/training" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Training
          </NavLink>
          <NavLink to="/export" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Export
          </NavLink>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<MatchList />} />
            <Route path="/annotate" element={<Annotate />} />
            <Route path="/training" element={<Training />} />
            <Route path="/export" element={<Export />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
