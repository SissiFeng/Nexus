import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import CampaignDetail from "./pages/CampaignDetail";
import Compare from "./pages/Compare";
import Reports from "./pages/Reports";

function App() {
  return (
    <div className="app">
      <nav className="nav-bar">
        <div className="nav-brand">
          <Link to="/">Optimization Copilot</Link>
        </div>
        <div className="nav-links">
          <Link to="/" className="nav-link">
            Dashboard
          </Link>
          <Link to="/compare" className="nav-link">
            Compare
          </Link>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/campaigns/:id" element={<CampaignDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/reports/:id" element={<Reports />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
