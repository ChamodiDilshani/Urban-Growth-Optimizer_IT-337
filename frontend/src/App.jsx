import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import EncroachmentCheck from "./pages/EncroachmentCheck.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<EncroachmentCheck />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}