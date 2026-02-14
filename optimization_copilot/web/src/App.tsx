import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import CampaignDetail from "./pages/CampaignDetail";
import Compare from "./pages/Compare";
import Reports from "./pages/Reports";
import LoopView from "./pages/LoopView";
import AnalysisView from "./pages/AnalysisView";
import NewCampaign from "./pages/NewCampaign";
import Workspace from "./pages/Workspace";

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
          <Link to="/new-campaign" className="nav-link">
            New Campaign
          </Link>
          <Link to="/loop" className="nav-link">
            Loop
          </Link>
          <Link to="/analysis" className="nav-link">
            Analysis
          </Link>
          <Link to="/compare" className="nav-link">
            Compare
          </Link>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/new-campaign" element={<NewCampaign />} />
          <Route path="/workspace/:id" element={<Workspace />} />
          <Route path="/campaigns/:id" element={<CampaignDetail />} />
          <Route path="/loop" element={<LoopView />} />
          <Route path="/analysis" element={<AnalysisView />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/reports/:id" element={<Reports />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
