import CytoscapeComponent from "react-cytoscapejs";
import { Card, CardContent } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";
import { useEffect, useRef } from "react";
import type { Core } from "cytoscape";

const LICENSE_COLORS = {
  Individual: "#23529e",
  Family: "#9e4e23",
};

const LICENSE_HOLDER_COLORS = {
  Individual: "#23529e",
  Family: "#542106",
};

export default function GraphView() {
  const graphData = useAppStore((state) => state.graphData);
  const assignments = useAppStore((state) => state.assignments);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (cyRef.current) {
      cyRef.current.layout({ name: "cose" }).run();

      cyRef.current.elements().forEach((element) => {
        element.style("background-color", "blue");
      });
    }
  }, [graphData]);

  useEffect(() => {
    if (!assignments) return;

    let licenseAssignments: {node: number, license_holder: number, license_type: string}[] = [];

    for (const [key, value] of Object.entries(assignments)) {
      for (const assignment of value) {
        for (const node of assignment.covered_nodes) {
          licenseAssignments.push({
            node: node,
            license_holder: assignment.license_holder,
            license_type: key,
          });
        }
      }
    }

    if (cyRef.current) {
      console.log(cyRef.current.elements());
      cyRef.current.elements().forEach((element) => {
        const node = licenseAssignments.find((n) => n.node === parseInt(element.data("id")));
        if (node) {
          if (node.license_holder === node.node) { 
            element.style("background-color", LICENSE_HOLDER_COLORS[node.license_type as keyof typeof LICENSE_HOLDER_COLORS]);
          } else {
            element.style("background-color", LICENSE_COLORS[node.license_type as keyof typeof LICENSE_COLORS]);
          }
        }
      });
      cyRef.current.edges().forEach((edge) => {
        const source = edge.data("source");
        const target = edge.data("target");
        const sourceNode = licenseAssignments.find((n) => n.node === parseInt(source));
        const targetNode = licenseAssignments.find((n) => n.node === parseInt(target));
        if (sourceNode && targetNode) {
          if (sourceNode.license_holder === targetNode.license_holder) {
            edge.style("line-color", LICENSE_HOLDER_COLORS[sourceNode.license_type as keyof typeof LICENSE_HOLDER_COLORS]);
          } else {
            edge.style("line-color", "#999");
          }
        }
      });
      
    }
  }, [assignments]);

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
      data: {
        id: String(node),
        label: String(node),
        backgroundColor: "blue"
      },
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
          cy={(cy) => {
            cyRef.current = cy;
          }}
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
                "background-color": "data(backgroundColor)",
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
