import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, ReferenceLine,
} from "recharts";

// ── helpers ──────────────────────────────────────────────────────────
const rnd = (n, d = 2) => +n.toFixed(d);
const fmtUSD = (n) => "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = (n) => (n * 100).toFixed(1) + "%";

function seededRng(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

function generateData(seed = 42) {
  const rng = seededRng(seed);
  const N = 300;
  let balance = 10000, equity = 10000;
  const equityCurve = [], trades = [], signals = [], confHistory = [];

  for (let i = 0; i < N; i++) {
    const date = new Date(Date.now() - (N - i) * 3600000);
    const label = date.toLocaleDateString("id-ID", { month: "short", day: "numeric" }) +
      " " + date.getHours().toString().padStart(2, "0") + ":00";

    const sig = rng() < 0.3 ? "BUY" : rng() < 0.5 ? "SELL" : "HOLD";
    signals.push(sig);

    if (sig !== "HOLD" && trades.length < 120) {
      const win = rng() < 0.535;
      const pnl = win ? rnd(rng() * 60 + 20) : rnd(-(rng() * 45 + 10));
      balance += pnl;
      equity = balance + (rng() - 0.5) * 40;
      const conf = rnd(rng() * 0.35 + 0.55, 3);
      trades.push({ date: label, dir: sig, pnl, result: win ? "WIN" : "LOSS", conf, balance: rnd(balance) });
    }

    equity = balance + (rng() - 0.5) * 60;
    equityCurve.push({ date: label, equity: rnd(equity), balance: rnd(balance) });
    confHistory.push({ date: label, conf: rnd(rng() * 0.35 + 0.55, 3) });
  }

  const wins = trades.filter(t => t.result === "WIN");
  const losses = trades.filter(t => t.result === "LOSS");
  const grossProfit = wins.reduce((s, t) => s + t.pnl, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnl, 0));
  const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);

  // Monthly PnL
  const monthMap = {};
  trades.forEach(t => {
    const m = t.date.slice(0, 6);
    monthMap[m] = (monthMap[m] || 0) + t.pnl;
  });
  const monthlyPnl = Object.entries(monthMap).map(([m, v]) => ({ month: m, pnl: rnd(v) })).slice(-8);

  // PnL buckets for histogram
  const buckets = {};
  trades.forEach(t => {
    const b = Math.round(t.pnl / 10) * 10;
    if (!buckets[b]) buckets[b] = { bucket: b, win: 0, loss: 0 };
    t.result === "WIN" ? buckets[b].win++ : buckets[b].loss++;
  });
  const pnlHist = Object.values(buckets).sort((a, b) => a.bucket - b.bucket);

  // Confusion-style signal distribution
  const sigDist = [
    { name: "BUY",  value: trades.filter(t => t.dir === "BUY").length },
    { name: "SELL", value: trades.filter(t => t.dir === "SELL").length },
  ];

  return {
    equityCurve: equityCurve.filter((_, i) => i % 3 === 0),
    trades, monthlyPnl, pnlHist, sigDist,
    metrics: {
      winRate: rnd(wins.length / trades.length, 4),
      sharpe: rnd(1.05 + rng() * 0.6, 2),
      maxDD: rnd(-0.04 - rng() * 0.07, 4),
      profitFactor: rnd(grossProfit / (grossLoss || 1), 2),
      totalTrades: trades.length,
      totalPnl: rnd(totalPnl),
      accuracy: rnd(0.54 + rng() * 0.08, 3),
      balance: rnd(balance),
    },
    signal: {
      label: ["BUY", "SELL", "HOLD"][Math.floor(rng() * 3)],
      conf: rnd(rng() * 0.35 + 0.58, 3),
      price: rnd(1.0823 + rng() * 0.01, 5),
      sl: rnd(1.0800 + rng() * 0.002, 5),
      tp: rnd(1.0870 + rng() * 0.005, 5),
      sl_pips: rnd(rng() * 15 + 15),
      tp_pips: rnd(rng() * 20 + 25),
      lot: rnd(rng() * 0.08 + 0.01, 2),
      risk_ok: rng() > 0.15,
      p_buy: rnd(rng() * 0.4 + 0.25, 3),
      p_sell: rnd(rng() * 0.35 + 0.2, 3),
      p_hold: rnd(rng() * 0.3 + 0.15, 3),
    },
    recentTrades: trades.slice(-12).reverse(),
  };
}

// ── palette ──────────────────────────────────────────────────────────
const C = {
  buy:  "#1d9e75",
  sell: "#e24b4a",
  hold: "#888780",
  blue: "#378add",
  purp: "#7f77dd",
  amber:"#ba7517",
  bg:   "var(--color-background-secondary)",
  brd:  "var(--color-border-tertiary)",
  txt:  "var(--color-text-primary)",
  txt2: "var(--color-text-secondary)",
};

// ── sub-components ───────────────────────────────────────────────────
function MetricCard({ label, value, sub, ok }) {
  return (
    <div style={{ background: C.bg, borderRadius: 8, padding: "14px 16px", minWidth: 0 }}>
      <p style={{ margin: 0, fontSize: 12, color: C.txt2, marginBottom: 4 }}>{label}</p>
      <p style={{ margin: 0, fontSize: 22, fontWeight: 500, color: C.txt }}>{value}</p>
      {sub != null && (
        <p style={{ margin: "4px 0 0", fontSize: 11, color: ok === true ? C.buy : ok === false ? C.sell : C.txt2 }}>
          {ok === true ? "✓ " : ok === false ? "⚠ " : ""}{sub}
        </p>
      )}
    </div>
  );
}

function Badge({ label, color }) {
  const bg = { BUY: "#e1f5ee", SELL: "#fcebeb", HOLD: "#f1efe8" }[label] || "#f1efe8";
  const fg = { BUY: C.buy, SELL: C.sell, HOLD: C.hold }[label] || C.hold;
  return (
    <span style={{ background: bg, color: fg, fontSize: 13, fontWeight: 500,
      padding: "3px 10px", borderRadius: 6 }}>
      {label === "BUY" ? "▲ " : label === "SELL" ? "▼ " : "— "}{label}
    </span>
  );
}

function ProbBar({ label, val, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12,
        color: C.txt2, marginBottom: 3 }}>
        <span>{label}</span><span>{(val * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: C.brd, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${val * 100}%`, background: color,
          borderRadius: 3, transition: "width .4s" }} />
      </div>
    </div>
  );
}

const TABS = ["Equity Curve", "Sinyal & Harga", "Distribusi Trade", "Analisis Model"];

export default function Dashboard() {
  const [data, setData] = useState(() => generateData(42));
  const [tab, setTab] = useState(0);
  const [sym, setSym] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setTick(t => t + 1);
      setData(generateData(Math.floor(Date.now() / 30000)));
    }, 30000);
    return () => clearInterval(id);
  }, []);

  const { metrics: m, signal: s, equityCurve, monthlyPnl, pnlHist, sigDist, recentTrades } = data;

  const signalColor = { BUY: C.buy, SELL: C.sell, HOLD: C.hold }[s.label];

  return (
    <div style={{ fontFamily: "var(--font-sans)", color: C.txt, padding: "0 0 2rem" }}>
      <h2 style={{ fontSize: 18, fontWeight: 500, margin: "1.5rem 0 0.25rem" }}>
        ML Forex Advisor — Monitoring Dashboard
      </h2>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: "1.25rem",
        flexWrap: "wrap" }}>
        <select value={sym} onChange={e => setSym(e.target.value)}
          style={{ fontSize: 13 }}>
          {["EURUSD","GBPUSD","USDJPY","XAUUSD"].map(s => <option key={s}>{s}</option>)}
        </select>
        <select value={tf} onChange={e => setTf(e.target.value)}
          style={{ fontSize: 13 }}>
          {["M15","H1","H4","D1"].map(t => <option key={t}>{t}</option>)}
        </select>
        <button onClick={() => setData(generateData(Date.now() % 9999))}
          style={{ fontSize: 13 }}>
          <i className="ti ti-refresh" aria-hidden /> Refresh
        </button>
        <span style={{ fontSize: 12, color: C.txt2, marginLeft: 4 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%",
            background: C.buy, marginRight: 4 }} />
          API Demo Mode
        </span>
      </div>

      {/* ── Signal Panel ── */}
      <div style={{ border: `0.5px solid ${C.brd}`, borderRadius: 12,
        padding: "1rem 1.25rem", marginBottom: "1rem",
        background: "var(--color-background-primary)" }}>
        <p style={{ margin: "0 0 12px", fontSize: 13, fontWeight: 500, color: C.txt2 }}>
          📡 Sinyal Terkini — {sym} {tf}
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 12 }}>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Signal</p>
            <div style={{ marginTop: 4 }}><Badge label={s.label} /></div>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Confidence</p>
            <p style={{ margin: "4px 0 0", fontSize: 20, fontWeight: 500,
              color: s.conf >= 0.65 ? C.buy : C.amber }}>
              {(s.conf * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Harga</p>
            <p style={{ margin: "4px 0 0", fontSize: 20, fontWeight: 500 }}>{s.price}</p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Stop Loss</p>
            <p style={{ margin: "4px 0 0", fontSize: 16, fontWeight: 500, color: C.sell }}>
              {s.sl} <span style={{ fontSize: 12, fontWeight: 400 }}>({s.sl_pips} pips)</span>
            </p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Take Profit</p>
            <p style={{ margin: "4px 0 0", fontSize: 16, fontWeight: 500, color: C.buy }}>
              {s.tp} <span style={{ fontSize: 12, fontWeight: 400 }}>({s.tp_pips} pips)</span>
            </p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Lot Size</p>
            <p style={{ margin: "4px 0 0", fontSize: 20, fontWeight: 500 }}>{s.lot}</p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 12, color: C.txt2 }}>Risk Filter</p>
            <p style={{ margin: "4px 0 0", fontSize: 14, fontWeight: 500,
              color: s.risk_ok ? C.buy : C.sell }}>
              {s.risk_ok ? "✓ Lolos" : "✗ Ditolak"}
            </p>
          </div>
        </div>

        {/* Probability bars */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginTop: 16 }}>
          <ProbBar label="BUY"  val={s.p_buy}  color={C.buy} />
          <ProbBar label="SELL" val={s.p_sell} color={C.sell} />
          <ProbBar label="HOLD" val={s.p_hold} color={C.hold} />
        </div>
      </div>

      {/* ── Metric Cards ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))",
        gap: 10, marginBottom: "1.25rem" }}>
        <MetricCard label="Win Rate" value={fmtPct(m.winRate)}
          sub={m.winRate >= 0.45 ? "di atas threshold" : "⚠ di bawah threshold"}
          ok={m.winRate >= 0.45} />
        <MetricCard label="Sharpe Ratio" value={m.sharpe}
          sub={m.sharpe >= 0.5 ? "baik" : "rendah"} ok={m.sharpe >= 0.5} />
        <MetricCard label="Max Drawdown" value={fmtPct(m.maxDD)}
          sub={m.maxDD > -0.10 ? "terkendali" : "⚠ tinggi"} ok={m.maxDD > -0.10} />
        <MetricCard label="Profit Factor" value={m.profitFactor}
          sub={m.profitFactor >= 1.2 ? "profitable" : "marginal"} ok={m.profitFactor >= 1.2} />
        <MetricCard label="Total Trades" value={m.totalTrades} />
        <MetricCard label="Total PnL" value={fmtUSD(m.totalPnl)}
          sub={m.totalPnl >= 0 ? "profit" : "loss"} ok={m.totalPnl >= 0} />
        <MetricCard label="Model Accuracy" value={fmtPct(m.accuracy)} />
        <MetricCard label="Balance" value={fmtUSD(m.balance)} />
      </div>

      {/* ── Tabs ── */}
      <div style={{ display: "flex", gap: 0, marginBottom: "1rem",
        borderBottom: `0.5px solid ${C.brd}` }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)}
            style={{
              background: "none", border: "none", borderBottom: i === tab ?
                `2px solid ${C.blue}` : "2px solid transparent",
              padding: "8px 14px", fontSize: 13, cursor: "pointer",
              color: i === tab ? C.blue : C.txt2, fontWeight: i === tab ? 500 : 400,
            }}>{t}</button>
        ))}
      </div>

      {/* ── Tab 0: Equity Curve ── */}
      {tab === 0 && (
        <div>
          <p style={{ fontSize: 13, color: C.txt2, margin: "0 0 8px" }}>
            Equity vs balance — {equityCurve.length} data point
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={equityCurve} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.blue} stopOpacity={0.18} />
                  <stop offset="95%" stopColor={C.blue} stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.brd} />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={Math.floor(equityCurve.length / 6)} />
              <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]}
                tickFormatter={v => "$" + v.toLocaleString()} />
              <Tooltip formatter={(v, n) => [fmtUSD(v), n === "equity" ? "Equity" : "Balance"]} />
              <ReferenceLine y={10000} stroke={C.txt2} strokeDasharray="4 2" label={{ value: "Start", fontSize: 11 }} />
              <Area type="monotone" dataKey="equity" stroke={C.blue} fill="url(#eqGrad)"
                strokeWidth={2} dot={false} name="equity" />
              <Line type="monotone" dataKey="balance" stroke={C.purp} strokeWidth={1.5}
                dot={false} name="balance" strokeDasharray="4 3" />
            </AreaChart>
          </ResponsiveContainer>

          {/* Monthly PnL */}
          <p style={{ fontSize: 13, color: C.txt2, margin: "1.5rem 0 8px" }}>PnL bulanan</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={monthlyPnl} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.brd} />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={v => "$" + v} />
              <Tooltip formatter={v => [fmtUSD(v), "PnL"]} />
              <ReferenceLine y={0} stroke={C.txt2} />
              <Bar dataKey="pnl" name="PnL"
                fill={C.buy}
                label={false}>
                {monthlyPnl.map((e, i) => (
                  <Cell key={i} fill={e.pnl >= 0 ? C.buy : C.sell} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Tab 1: Sinyal & Harga ── */}
      {tab === 1 && (
        <div>
          <p style={{ fontSize: 13, color: C.txt2, margin: "0 0 8px" }}>
            Equity curve dengan anotasi sinyal BUY/SELL
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={equityCurve.slice(-80)} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.brd} />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={10} />
              <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]}
                tickFormatter={v => "$" + v.toLocaleString()} />
              <Tooltip formatter={(v) => [fmtUSD(v), "Equity"]} />
              <Line type="monotone" dataKey="equity" stroke={C.blue}
                strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>

          {/* Recent trades on a table */}
          <p style={{ fontSize: 13, color: C.txt2, margin: "1.5rem 0 8px" }}>Trade terbaru</p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: `0.5px solid ${C.brd}` }}>
                  {["Waktu","Arah","PnL","Hasil","Confidence","Balance"].map(h => (
                    <th key={h} style={{ padding: "6px 10px", textAlign: "left",
                      color: C.txt2, fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((t, i) => (
                  <tr key={i} style={{ borderBottom: `0.5px solid ${C.brd}` }}>
                    <td style={{ padding: "6px 10px", color: C.txt2 }}>{t.date}</td>
                    <td style={{ padding: "6px 10px" }}>
                      <span style={{ color: t.dir === "BUY" ? C.buy : C.sell, fontWeight: 500 }}>
                        {t.dir === "BUY" ? "▲" : "▼"} {t.dir}
                      </span>
                    </td>
                    <td style={{ padding: "6px 10px",
                      color: t.pnl >= 0 ? C.buy : C.sell, fontWeight: 500 }}>
                      {t.pnl >= 0 ? "+" : ""}{fmtUSD(t.pnl)}
                    </td>
                    <td style={{ padding: "6px 10px" }}>
                      <span style={{
                        background: t.result === "WIN" ? "#e1f5ee" : "#fcebeb",
                        color: t.result === "WIN" ? C.buy : C.sell,
                        padding: "2px 8px", borderRadius: 4, fontSize: 11,
                      }}>{t.result}</span>
                    </td>
                    <td style={{ padding: "6px 10px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div style={{ flex: 1, height: 4, background: C.brd, borderRadius: 2 }}>
                          <div style={{ width: `${t.conf * 100}%`, height: "100%",
                            background: C.blue, borderRadius: 2 }} />
                        </div>
                        <span style={{ color: C.txt2, minWidth: 32 }}>{(t.conf * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td style={{ padding: "6px 10px" }}>{fmtUSD(t.balance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab 2: Distribusi Trade ── */}
      {tab === 2 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <p style={{ fontSize: 13, color: C.txt2, margin: "0 0 8px" }}>Distribusi PnL per trade</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={pnlHist} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.brd} />
                  <XAxis dataKey="bucket" tick={{ fontSize: 10 }}
                    tickFormatter={v => (v > 0 ? "+" : "") + v} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v, n) => [v, n === "win" ? "Win" : "Loss"]} />
                  <Bar dataKey="win"  fill={C.buy}  stackId="a" name="Win" />
                  <Bar dataKey="loss" fill={C.sell} stackId="a" name="Loss" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <p style={{ fontSize: 13, color: C.txt2, margin: "0 0 8px" }}>Rasio WIN / LOSS</p>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={[
                    { name: "Win",  value: recentTrades.filter(t => t.result === "WIN").length  + 40 },
                    { name: "Loss", value: recentTrades.filter(t => t.result === "LOSS").length + 20 },
                  ]} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                    dataKey="value" label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`}>
                    <Cell fill={C.buy} />
                    <Cell fill={C.sell} />
                  </Pie>
                  <Legend />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <p style={{ fontSize: 13, color: C.txt2, margin: "1.5rem 0 8px" }}>Distribusi sinyal</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={sigDist} layout="vertical" margin={{ top: 4, right: 40, bottom: 0, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.brd} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" name="Jumlah sinyal" radius={[0, 4, 4, 0]}>
                <Cell fill={C.buy} />
                <Cell fill={C.sell} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Tab 3: Analisis Model ── */}
      {tab === 3 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            {/* Gauge cards */}
            {[
              { label: "LSTM Val Accuracy", value: 57.8, max: 100, unit: "%", ok: 57.8 > 55 },
              { label: "XGBoost Val Accuracy", value: 61.2, max: 100, unit: "%", ok: true },
              { label: "Ensemble Confidence Avg", value: 68.4, max: 100, unit: "%", ok: true },
              { label: "Model Drift Score", value: 4.2, max: 15, unit: "%", ok: 4.2 < 10 },
            ].map(g => (
              <div key={g.label} style={{ background: C.bg, borderRadius: 10, padding: "14px 16px" }}>
                <p style={{ margin: "0 0 8px", fontSize: 12, color: C.txt2 }}>{g.label}</p>
                <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                  <span style={{ fontSize: 24, fontWeight: 500,
                    color: g.ok ? C.buy : C.amber }}>{g.value}</span>
                  <span style={{ fontSize: 13, color: C.txt2 }}>{g.unit}</span>
                </div>
                <div style={{ marginTop: 8, height: 5, background: C.brd, borderRadius: 3 }}>
                  <div style={{ height: "100%", width: `${(g.value / g.max) * 100}%`,
                    background: g.ok ? C.buy : C.amber, borderRadius: 3 }} />
                </div>
              </div>
            ))}
          </div>

          {/* Status checklist */}
          <div style={{ border: `0.5px solid ${C.brd}`, borderRadius: 10,
            padding: "1rem 1.25rem", marginBottom: 16 }}>
            <p style={{ margin: "0 0 12px", fontSize: 13, fontWeight: 500 }}>Status Komponen Sistem</p>
            {[
              { label: "MT5 Connection", ok: false, detail: "Demo mode — MT5 tidak terhubung" },
              { label: "Data Collector", ok: true,  detail: "Menggunakan data simulasi" },
              { label: "Feature Engineering", ok: true, detail: "80 fitur aktif" },
              { label: "XGBoost Model", ok: true, detail: "Loaded — 500 estimators" },
              { label: "LSTM Model", ok: true, detail: "Loaded — 2 layers, hidden=128" },
              { label: "Ensemble Predictor", ok: true, detail: "LSTM 45% + XGB 55%" },
              { label: "Risk Filter", ok: true, detail: "Confidence ≥ 60%, spread ≤ 3 pips" },
              { label: "API Server", ok: false, detail: "Offline — jalankan main.py --mode api" },
              { label: "MT5 Expert Advisor", ok: false, detail: "Belum attach ke chart" },
            ].map(item => (
              <div key={item.label} style={{ display: "flex", alignItems: "center",
                gap: 10, padding: "7px 0",
                borderBottom: `0.5px solid ${C.brd}` }}>
                <span style={{ fontSize: 14,
                  color: item.ok ? C.buy : C.sell }}>{item.ok ? "✓" : "✗"}</span>
                <span style={{ fontSize: 13, fontWeight: 500, minWidth: 180 }}>{item.label}</span>
                <span style={{ fontSize: 12, color: C.txt2 }}>{item.detail}</span>
              </div>
            ))}
          </div>

          {/* Config summary */}
          <div style={{ background: C.bg, borderRadius: 10, padding: "1rem 1.25rem" }}>
            <p style={{ margin: "0 0 10px", fontSize: 13, fontWeight: 500 }}>Konfigurasi Aktif</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 8 }}>
              {[
                ["Symbol",         sym],
                ["Timeframe",      tf],
                ["Sequence Length","60 bar"],
                ["Confidence Min", "60%"],
                ["Risk per Trade", "1%"],
                ["Max Spread",     "3 pips"],
                ["SL Multiplier",  "2× ATR"],
                ["TP Multiplier",  "3× ATR"],
                ["Max DD Harian",  "5%"],
                ["Retrain",        "Tiap 7 hari"],
              ].map(([k, v]) => (
                <div key={k} style={{ fontSize: 12 }}>
                  <span style={{ color: C.txt2 }}>{k}: </span>
                  <span style={{ fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <p style={{ fontSize: 11, color: C.txt2, marginTop: "1.5rem", textAlign: "center" }}>
        ML Forex Advisor Dashboard · Demo Mode · Data simulasi — bukan sinyal trading nyata
      </p>
    </div>
  );
}
