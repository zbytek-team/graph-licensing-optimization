import GraphView from "@/components/GraphView";
import GraphPanel from "@/components/panels/GraphPanel";
import LicensesPanel from "@/components/panels/LicensesPanel";
import ResultsPanel from "@/components/panels/ResultsPanel";
import TopBar from "@/components/TopBar";
import { solveAssignment } from "@/api/solveApi";
import { useAppStore } from "@/store/useAppStore";

export default function App() {
  const { graphData, licenses, setAssignments } = useAppStore();

  const onSolveRun = async (solver: string) => {
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

  return (
    <div className="flex h-screen bg-background text-foreground p-4 gap-4">
      <div className="flex flex-col flex-grow gap-4">
        <div className="h-16 flex-shrink-0">
          <TopBar onSolveRun={onSolveRun} />
        </div>
        <div className="flex-grow">
          <GraphView graphData={graphData} />
        </div>
      </div>
      <div className="w-96 flex flex-col gap-4">
        <div className="flex-1/10">
          <GraphPanel />
        </div>
        <div className="flex-7/10">
          <LicensesPanel />
        </div>
        <div className="flex-1/5">
          <ResultsPanel />
        </div>
      </div>
    </div>
  );
}
