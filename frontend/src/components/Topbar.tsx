import { Play, PlusCircle } from "lucide-react"
import { Button } from "./ui/button"

interface TopbarProps {
  selectedAlgorithm: string | null
  onAddNode: () => void
  onRunSimulation: () => void
}

function Topbar({ selectedAlgorithm, onAddNode, onRunSimulation }: TopbarProps) {
  return (
    <div className="flex items-center justify-between bg-background p-4 shadow border-b border-border">
      <div className="text-xl font-bold text-primary">Optimal License Distribution Simulation</div>
      <div className="flex space-x-2">
        <Button onClick={onAddNode} variant="outline">
          <PlusCircle className="mr-2 h-4 w-4" />
          Add Node
        </Button>
        <Button onClick={onRunSimulation} disabled={!selectedAlgorithm} variant="outline">
          <Play className="mr-2 h-4 w-4" />
          Run Simulation
        </Button>
      </div>
    </div >
  )
}

export default Topbar