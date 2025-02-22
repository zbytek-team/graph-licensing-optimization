import CytoscapeComponent from "react-cytoscapejs";
import { Card, CardContent } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Core } from "cytoscape";
import { adjustColor } from "@/lib/colorUtils";
import { processAssignments, LicenseAssignment } from "@/lib/licenseUtils";

export default function GraphView() {
  const graphData = useAppStore((state) => state.graphData);
  const assignments = useAppStore((state) => state.assignments);
  const cyRef = useRef<Core | null>(null);
  const [legendData, setLegendData] = useState<{ type: string; color: string }[]>([]);

  const elements = useMemo(() => {
    if (!graphData) return [];
    return [
      ...graphData.nodes.map((node) => ({
        data: {
          id: String(node),
          label: String(node),
          backgroundColor: "blue",
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
  }, [graphData]);

  useEffect(() => {
    if (cyRef.current && graphData) {
      const cy = cyRef.current;
      cy.layout({ name: "cose" }).run();
      cy.elements().forEach((element) => {
        element.style({
          "background-color": "blue",
          "border-width": 0,
          "border-color": "transparent",
          "border-opacity": 0,
        });
      });
    }
  }, [graphData]);

  useEffect(() => {
    if (!assignments || !cyRef.current) return;
    const cy = cyRef.current;

    cy.elements().forEach((element) => {
      element.style({
        "border-width": 0,
        "border-color": "transparent",
        "border-opacity": 0,
      });
    });

    const { licenseAssignments, licenseToColor } = processAssignments(assignments);

    setLegendData(
      Object.entries(licenseToColor).map(([type, color]) => ({ type, color }))
    );

    const licenseHolderColor: Record<number, string> = {};
    licenseAssignments.forEach((assignment: LicenseAssignment) => {
      if (!licenseHolderColor[assignment.license_holder]) {
        licenseHolderColor[assignment.license_holder] = adjustColor(
          licenseToColor[assignment.license_type]
        );
      }
    });

    cy.elements().forEach((element) => {
      const elementId = parseInt(element.data("id"));
      const assignment = licenseAssignments.find((item) => item.node === elementId);
      if (assignment) {
        element.style("background-color", licenseHolderColor[assignment.license_holder]);
        if (assignment.node === assignment.license_holder) {
          element.style({
            "border-width": 2,
            "border-color": "#ffffff",
            "border-opacity": 0.3,
          });
        }
      }
    });

    cy.edges().forEach((edge) => {
      const sourceId = parseInt(edge.data("source"));
      const targetId = parseInt(edge.data("target"));
      const sourceAssignment = licenseAssignments.find((item) => item.node === sourceId);
      const targetAssignment = licenseAssignments.find((item) => item.node === targetId);

      if (sourceAssignment && targetAssignment) {
        edge.style(
          "line-color",
          sourceAssignment.license_holder === targetAssignment.license_holder
            ? licenseHolderColor[sourceAssignment.license_holder]
            : "#999"
        );
      }
    });
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

  return (
    <Card className="h-full">
      <CardContent className="p-4 h-full relative">
        {legendData.length > 0 && (
          <div className="absolute top-4 left-4 bg-background p-3 rounded-lg shadow-md z-10 flex flex-col gap-1">
            <h3 className="text-sm font-semibold mb-2">License Types</h3>
            {legendData.map((item, index) => (
              <div key={index} className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-xs">{item.type}</span>
              </div>
            ))}
          </div>
        )}
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
                opacity: 0.8,
              },
            },
          ]}
        />
      </CardContent>
    </Card>
  );
}
