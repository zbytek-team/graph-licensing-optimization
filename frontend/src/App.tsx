import GraphView from "@/components/GraphView";
import GraphPanel from "@/components/panels/GraphPanel";
import LicensesPanel from "@/components/panels/LicensesPanel";
import ResultsPanel from "@/components/panels/ResultsPanel";
import TopBar from "@/components/TopBar";

export default function App() {
  return (
    <div className="flex h-screen bg-background text-foreground p-4 gap-4">
      <div className="flex flex-col flex-grow gap-4">
        <div className="h-16 flex-shrink-0">
          <TopBar />
        </div>
        <div className="flex-grow">
          <GraphView />
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
