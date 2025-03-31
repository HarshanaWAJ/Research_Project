import './App.css';

// React Router Importing
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

// Pages Importing
import UmpireAssistance from './pages/UmpireAssistance';
import LbwClassification from './pages/LbwClassification';
import WideClassification from './pages/WideClassification';
import NoBallClassification from './pages/NoBallClassification';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UmpireAssistance />} />
        <Route path="/lbw-clasification" element={<LbwClassification />} />
        <Route path="/wide-clasification" element={<WideClassification />} />
        <Route path="/noball-clasification" element={<NoBallClassification />} />
      </Routes>
    </Router>
  );
}

export default App;
