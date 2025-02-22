import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Play } from "lucide-react";
import { useState } from "react";
import { useSolve } from "@/api/solveApi";

export default function TopBar() {
  const [selectedSolver, setSelectedSolver] = useState<string>("");
  const solve = useSolve();

  return (
    <Card className="h-full">
      <CardContent className="p-4 h-full flex items-center justify-between">
        <h1 className="text-2xl font-bold">License Distributor</h1>
        <div className="flex items-center space-x-4">
          <Select onValueChange={setSelectedSolver}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select solver method" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="greedy">Greedy</SelectItem>
              <SelectItem value="ants">Ants</SelectItem>
              <SelectItem value="ants_multiprocessing">
                Ants Multiprocessing
              </SelectItem>
            </SelectContent>
          </Select>
          <Button
            size="icon"
            onClick={() => solve(selectedSolver)}
            disabled={!selectedSolver}
          >
            <Play className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
