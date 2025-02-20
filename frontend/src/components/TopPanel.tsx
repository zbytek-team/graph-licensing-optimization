import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Play } from "lucide-react";

interface TopPanelProps {
  onSolveRun: (solver: string) => void;
}

export default function TopPanel({ onSolveRun }: TopPanelProps) {
  const [selectedSolver, setSelectedSolver] = useState<string>("");

  const handleRun = () => {
    if (selectedSolver) {
      onSolveRun(selectedSolver);
    } else {
      console.warn("Please select a solver method before running.");
    }
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4 h-full flex items-center justify-between">
        <h1 className="text-2xl font-bold">License Distributor</h1>
        <div className="flex items-center space-x-4">
          <Select onValueChange={(value) => setSelectedSolver(value)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select solver method" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="greedy">Greedy</SelectItem>
              <SelectItem value="mip">MIP</SelectItem>
              <SelectItem value="genetic">Genetic</SelectItem>
            </SelectContent>
          </Select>
          <Button size="icon" onClick={handleRun}>
            <Play className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
