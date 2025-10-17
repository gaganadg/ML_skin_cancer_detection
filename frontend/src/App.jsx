import React, { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import "./styles.css";

const API_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function App(){
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [symptoms, setSymptoms] = useState({
    itching: false,
    bleeding: false,
    pain: false,
    color_change: false,
    rapid_growth: false,
    border_irregularity: false,
    diameter_mm: "",
    age: "",
    family_history: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const dropRef = useRef(null);

  useEffect(() => {
    if (!file) { setPreviewUrl(""); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
    dropRef.current?.classList.remove("drag");
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.add("drag");
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.remove("drag");
  }, []);

  const onUpload = async (e) => {
    e.preventDefault();
    if(!file) return;
    setError("");
    setLoading(true);
    try{
      const form = new FormData();
      form.append("image", file);
      form.append("symptoms", JSON.stringify(symptoms));
      const { data } = await axios.post(`${API_URL}/predict`, form, { headers: { 'Content-Type': 'multipart/form-data' }});
      setResult(data);
      // Immediate history refresh
      try { const hist = await axios.get(`${API_URL}/results`); setHistory(hist.data); } catch {}
    }catch(err){
      setError("Prediction failed. Ensure ML service is running on 8001.");
    }finally{ setLoading(false); }
  };

  const loadHistory = async () => {
    try {
      const { data } = await axios.get(`${API_URL}/results`);
      setHistory(data);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => { loadHistory(); }, []);

  return (
    <div className="container">
      <div className="header">
        <div className="title">Skin Cancer Prediction</div>
        <button className="btn secondary" onClick={loadHistory}>Refresh history</button>
      </div>

      <div className="grid">
        <div className="card">
          <div
            ref={dropRef}
            className="dropzone"
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <div>Drag & drop an image here, or click to select</div>
            <div className="hint">Recommended: clear lesion photo. PNG/JPG.</div>
            <input id="fileInput" type="file" accept="image/*" onChange={e=>setFile(e.target.files?.[0]||null)} style={{ display: 'none' }} />
          </div>

          {previewUrl && (
            <div className="preview" style={{ marginTop: 12 }}>
              <img src={previewUrl} alt="preview" />
              <div>
                <div className="badge">{file?.name}</div>
                <div className="hint">{Math.round((file?.size||0)/1024)} KB</div>
              </div>
            </div>
          )}

          {/* Symptoms form */}
          <div style={{ marginTop: 12 }}>
            <h3 style={{ marginTop: 0 }}>Symptoms</h3>
            <div className="list">
              <label className="item"><input type="checkbox" checked={symptoms.itching} onChange={e=>setSymptoms(s=>({ ...s, itching: e.target.checked }))} /> Itching</label>
              <label className="item"><input type="checkbox" checked={symptoms.bleeding} onChange={e=>setSymptoms(s=>({ ...s, bleeding: e.target.checked }))} /> Bleeding</label>
              <label className="item"><input type="checkbox" checked={symptoms.pain} onChange={e=>setSymptoms(s=>({ ...s, pain: e.target.checked }))} /> Pain</label>
              <label className="item"><input type="checkbox" checked={symptoms.color_change} onChange={e=>setSymptoms(s=>({ ...s, color_change: e.target.checked }))} /> Color change</label>
              <label className="item"><input type="checkbox" checked={symptoms.rapid_growth} onChange={e=>setSymptoms(s=>({ ...s, rapid_growth: e.target.checked }))} /> Rapid growth</label>
              <label className="item"><input type="checkbox" checked={symptoms.border_irregularity} onChange={e=>setSymptoms(s=>({ ...s, border_irregularity: e.target.checked }))} /> Border irregularity</label>
              <label className="item">Diameter (mm): <input type="number" min="0" value={symptoms.diameter_mm} onChange={e=>setSymptoms(s=>({ ...s, diameter_mm: e.target.value }))} style={{ width: 100, marginLeft: 8 }} /></label>
              <label className="item">Age: <input type="number" min="0" value={symptoms.age} onChange={e=>setSymptoms(s=>({ ...s, age: e.target.value }))} style={{ width: 100, marginLeft: 8 }} /></label>
              <label className="item"><input type="checkbox" checked={symptoms.family_history} onChange={e=>setSymptoms(s=>({ ...s, family_history: e.target.checked }))} /> Family history of skin cancer</label>
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <button className="btn" disabled={!file || loading} onClick={onUpload}>{loading? 'Predicting...' : 'Predict'}</button>
          </div>

          {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
        </div>

        <div className="card result">
          <h2>Result</h2>
          {!result && <div className="hint">No prediction yet.</div>}
          {result && (
            <>
              <div className="row">
                <div><b>Prediction</b></div>
                <div className="badge">{result.prediction}</div>
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <div><b>Confidence</b></div>
                <div>{(result.confidence*100).toFixed(2)}%</div>
              </div>
              {result.meta?.ood_rejected && (
                <div className="error" style={{ marginTop: 6 }}>Unrelated image detected. Result forced to benign.</div>
              )}
              {result.source && (
                <div className="hint" style={{ marginTop: 4 }}>
                  Source: {result.source === 'ml' ? 'ML service' : 'Mock fallback'}
                </div>
              )}
              {result.symptom_risk !== undefined && (
                <div className="row" style={{ marginTop: 6 }}>
                  <div><b>Symptom risk prior</b></div>
                  <div>{(result.symptom_risk*100).toFixed(1)}%</div>
                </div>
              )}
              {result.heatmap_base64 && !result.meta?.ood_rejected && (
                <img className="heatmap" style={{ marginTop: 12 }} alt="heatmap" src={result.heatmap_base64} />
              )}
            </>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>History</h2>
          <button className="btn secondary" onClick={loadHistory}>Refresh</button>
        </div>
        <ul className="list">
          {history.map(item => (
            <li className="item" key={item._id}>
              <div className="row">
                <div>{new Date(item.createdAt).toLocaleString()}</div>
                <div className="badge">{(item.confidence*100).toFixed(1)}%</div>
              </div>
              <div className="hint">{item.filename} – {item.prediction}</div>
              {item.symptom_risk !== undefined && (
                <div className="hint">Symptom risk: {(Number(item.symptom_risk)*100).toFixed(1)}%</div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}


