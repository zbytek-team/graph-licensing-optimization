import { Graph } from "../types/graph";

export const fetchGraph = async (graphType: string): Promise<Graph> => {
  const response = await fetch("http://localhost:8000/graph/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph_type: graphType }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.graph;
};
