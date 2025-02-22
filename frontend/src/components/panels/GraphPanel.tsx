import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { fetchGraph } from "@/api/graphApi";
import { useAppStore } from "@/store/useAppStore";
import { Graph } from "@/types/graph";

export default function GraphPanel() {
  const [graphType, setGraphType] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const setGraphData = useAppStore((state) => state.setGraphData);
  const setAssignments = useAppStore((state) => state.setAssignments);

  const handleSendRequest = async () => {
    if (!graphType) {
      alert("Please select a graph type");
      return;
    }
    setLoading(true);
    try {
      const graph: Graph = await fetchGraph(graphType);
      setGraphData(graph);
      setAssignments({});
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4 flex flex-col gap-4">
        <div className="flex items-center space-x-2">
          <Select onValueChange={setGraphType}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose graph type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="watts_strogatz">Watts Strogatz</SelectItem>
              <SelectItem value="barabasi_albert">Barabasi Albert</SelectItem>
              <SelectItem value="erdos_renyi">Erdos Renyi</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button onClick={handleSendRequest} disabled={loading}>
          {loading ? "Sending..." : "Generate graph"}
        </Button>
      </CardContent>
    </Card>
  );
}
