import { usePredictions } from "../api";
import { Card, ErrorBox, pct } from "../ui";

export default function Predictions() {
  const { data, error } = usePredictions(7);
  if (error) return <ErrorBox error={error} />;
  const preds = data?.predictions ?? [];

  return (
    <Card title={`Upcoming predictions (${preds.length})`}>
      {preds.length === 0 ? (
        <div className="text-muted text-sm">No predictions yet (off-season — World Cup fixtures appear as they enter the 48h window).</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabnum">
            <thead className="text-muted">
              <tr className="text-left">
                <th className="py-1.5 pr-3">kickoff</th><th>comp</th><th>match</th>
                <th>P(H)</th><th>P(D)</th><th>P(A)</th>
              </tr>
            </thead>
            <tbody>
              {preds.map((p, i) => (
                <tr key={i} className="border-t border-[#21262d]">
                  <td className="py-1.5 pr-3">{p.kickoff ? new Date(p.kickoff).toLocaleString() : "—"}</td>
                  <td>{p.competition_code}</td>
                  <td className="pr-3">{p.home_team} vs {p.away_team}</td>
                  <td>{pct(p.p_home, 0)}</td>
                  <td>{pct(p.p_draw, 0)}</td>
                  <td>{pct(p.p_away, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
