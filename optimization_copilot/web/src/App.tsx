import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import CampaignDetail from "./pages/CampaignDetail";
import Compare from "./pages/Compare";
import Reports from "./pages/Reports";
import LoopView from "./pages/LoopView";
import AnalysisView from "./pages/AnalysisView";
import NewCampaign from "./pages/NewCampaign";
import DemoGallery from "./pages/DemoGallery";
import Workspace from "./pages/Workspace";
import ErrorBoundary from "./components/ErrorBoundary";
import ThemeToggle from "./components/ThemeToggle";
import { ToastProvider } from "./components/Toast";

function App() {
  return (
    <ToastProvider>
    <div className="app">
      <nav className="nav-bar">
        <div className="nav-brand">
          <Link to="/"><span className="nav-brand-icon">◆</span> Nexus</Link>
        </div>
        <div className="nav-links">
          <Link to="/" className="nav-link">
            Dashboard
          </Link>
          <Link to="/new-campaign" className="nav-link">
            New Campaign
          </Link>
          <Link to="/demos" className="nav-link">
            Demos
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
        <div className="nav-right">
          <ThemeToggle />
        </div>
      </nav>
      <main className="main-content">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new-campaign" element={<NewCampaign />} />
            <Route path="/demos" element={<DemoGallery />} />
            <Route path="/workspace/:id" element={<Workspace />} />
            <Route path="/campaigns/:id" element={<CampaignDetail />} />
            <Route path="/loop" element={<LoopView />} />
            <Route path="/analysis" element={<AnalysisView />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/reports/:id" element={<Reports />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
    </ToastProvider>
  );
}

export default App;
