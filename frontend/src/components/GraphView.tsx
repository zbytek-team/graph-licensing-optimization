import CytoscapeComponent from "react-cytoscapejs";
import { Card, CardContent } from "@/components/ui/card";
import { Graph } from "@/types/graph";

interface GraphViewProps {
  graphData: Graph | null;
}

export default function GraphView({ graphData }: GraphViewProps) {
  if (!graphData) {
    return (
      <Card className="h-full">
        <CardContent className="p-4 h-full flex items-center justify-center text-gray-500">
          No graph data available.
        </CardContent>
      </Card>
    );
  }

  const elements = [
    ...graphData.nodes.map((node) => ({
      data: { id: String(node), label: String(node) },
    })),
    ...graphData.edges.map(([source, target]) => ({
      data: {
        id: `${source}-${target}`,
        source: String(source),
        target: String(target),
      },
    })),
  ];

  return (
    <Card className="h-full">
      <CardContent className="p-4 h-full">
        <CytoscapeComponent
          elements={elements}
          style={{ width: "100%", height: "100%" }}
          layout={{ name: "cose" }}
          stylesheet={[
            {
              selector: "node",
              style: {
                label: "data(label)",
                width: 30,
                height: 30,
                "background-color": "#007bff",
                color: "#ffffff",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
              },
            },
            {
              selector: "edge",
              style: {
                width: 2,
                "line-color": "#999",
                "curve-style": "bezier",
              },
            },
          ]}
        />
      </CardContent>
    </Card>
  );
}
