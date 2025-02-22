import { Graph } from "../types/graph";
import { License } from "../types/license";
import { AssignmentRecord } from "../types/assignment";
import { useAppStore } from "@/store/useAppStore";

interface SolvePayload {
  graph: Graph;
  licenses: License[];
  solver: string;
}

export const solveAssignment = async ({
  graph,
  licenses,
  solver,
}: SolvePayload): Promise<AssignmentRecord> => {
  const response = await fetch("http://localhost:8000/solve/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, licenses, solver }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.assignments;
};

export const useSolve = () => {
  const { graphData, licenses, setAssignments } = useAppStore();

  const solve = async (solver: string) => {
    if (!graphData || licenses.length === 0) {
      console.error("Graph data or licenses are missing");
      return;
    }

    try {
      const newAssignments = await solveAssignment({
        graph: graphData,
        licenses,
        solver,
      });
      setAssignments(newAssignments);
    } catch (error) {
      console.error(error);
    }
  };

  return solve;
};
