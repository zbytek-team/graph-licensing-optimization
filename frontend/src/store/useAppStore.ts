import { create } from "zustand";
import { Graph } from "../types/graph";
import { License } from "../types/license";
import { AssignmentRecord } from "../types/assignment";

interface AppState {
  graphData: Graph | null;
  licenses: License[];
  assignments: AssignmentRecord | null;
  setGraphData: (graph: Graph) => void;
  setLicenses: (licenses: License[]) => void;
  setAssignments: (assignments: AssignmentRecord) => void;
}

export const useAppStore = create<AppState>((set) => ({
  graphData: null,
  licenses: [],
  assignments: null,
  setGraphData: (graph) => set({ graphData: graph }),
  setLicenses: (licenses) => set({ licenses }),
  setAssignments: (assignments) => set({ assignments }),
}));
