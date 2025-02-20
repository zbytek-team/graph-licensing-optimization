import TopPanel from "@/components/TopPanel";
import GraphView from "@/components/GraphView";
import GraphTypePanel from "./components/GraphTypePanel";
import LicensesPanel from "./components/LicensesPanel";
import ResultsPanel from "@/components/ResultsPanel";

export default function App() {
  const onSolveRun = (solver: string) => {
    console.log("Solver:", solver);
  };

  return (
    <div className="flex h-screen bg-background text-foreground p-4 gap-4">
      <div className="flex flex-col flex-grow gap-4">
        <div className="h-16 flex-shrink-0">
          <TopPanel onSolveRun={onSolveRun} />
        </div>
        <div className="flex-grow">
          <GraphView />
        </div>
      </div>
      <div className="w-96 flex flex-col gap-4">
        <div className="flex-16">
          <GraphTypePanel />
        </div>
        <div className="h-grow">
          <LicensesPanel />
        </div>
        <div className="h-16">
          <ResultsPanel />
        </div>
      </div>
    </div>
  );
}
