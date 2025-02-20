import { Graph } from "../types/graph";
import { License } from "../types/license";

interface SolvePayload {
  graph: Graph;
  licenses: License[];
  solver: string;
}

export const solveAssignment = async (payload: SolvePayload): Promise<any> => {
  const response = await fetch("http://localhost:8000/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.assignments;
};
