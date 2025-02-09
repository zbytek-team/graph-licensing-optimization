// View.tsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Box, Button, Group } from "@mantine/core";
import styles from "../App.module.scss";
import { SimulationResponse } from "../App";

// Typy dla danych D3
interface NodeDatum extends d3.SimulationNodeDatum {
  id: string;
}
interface LinkDatum extends d3.SimulationLinkDatum<NodeDatum> {
  id: string;
  source: string | NodeDatum;
  target: string | NodeDatum;
}

// Mapowanie typów licencji na kolory
const licenseColors: { [key: string]: string } = {
  individual: "#1f77b4",
  group: "#2ca02c",
  mega: "#d62728",
};

interface ViewProps {
  simulationResult?: SimulationResponse | null;
  nodes: number[];
  links: number[][];
  setNodes: (nodes: number[]) => void;
  setLinks: (links: number[][]) => void;
}

function View({
  simulationResult,
  nodes,
  links,
  setNodes,
  setLinks,
}: ViewProps) {
  // Konwertujemy numeryczne węzły i krawędzie do formatu D3
  const d3Nodes: NodeDatum[] = nodes.map((n) => ({ id: String(n) }));
  const d3Links: LinkDatum[] = links.map(([s, t]) => ({
    id: `${s}-${t}`,
    source: String(s),
    target: String(t),
  }));

  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<NodeDatum, LinkDatum>>();
  const zoomGroupRef =
    useRef<d3.Selection<SVGGElement, unknown, null, undefined>>();
  const selectedNodeRef = useRef<NodeDatum | null>(null);
  const selectedEdgeRef = useRef<LinkDatum | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const width = svgRef.current.clientWidth || 600;
    const height = svgRef.current.clientHeight || 400;
    const svg = d3.select(svgRef.current);

    // Utwórz grupę podlegającą transformacji zoom/pan
    const zoomGroup = svg.append("g").attr("class", "zoom-group");
    zoomGroupRef.current = zoomGroup;

    // Grupy dla krawędzi i węzłów
    const linkGroup = zoomGroup.append("g").attr("class", "links");
    const nodeGroup = zoomGroup.append("g").attr("class", "nodes");

    // Ustaw zoom i panning
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 5])
      .on("zoom", (event) => {
        zoomGroup.attr("transform", event.transform);
      });
    svg.call(zoom);

    // Inicjalizacja symulacji
    const simulation = d3
      .forceSimulation<NodeDatum>(d3Nodes)
      .force(
        "link",
        d3
          .forceLink<NodeDatum, LinkDatum>(d3Links)
          .id((d) => d.id)
          .distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));
    simulationRef.current = simulation;

    simulation.on("tick", () => {
      linkGroup
        .selectAll("line")
        .attr("x1", (d: any) => (d.source as NodeDatum).x)
        .attr("y1", (d: any) => (d.source as NodeDatum).y)
        .attr("x2", (d: any) => (d.target as NodeDatum).x)
        .attr("y2", (d: any) => (d.target as NodeDatum).y);
      nodeGroup
        .selectAll("circle")
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);
    });

    function updateElements() {
      // Aktualizacja krawędzi
      const linkSelection = linkGroup
        .selectAll("line")
        .data(d3Links, (d: any) => d.id);
      linkSelection.exit().remove();
      linkSelection
        .enter()
        .append("line")
        .attr("stroke", "#ccc")
        .attr("stroke-width", 2)
        .on("click", (event, d: LinkDatum) => {
          event.stopPropagation();
          selectedEdgeRef.current = d;
          svg.selectAll("line").attr("stroke", "#ccc");
          d3.select(event.currentTarget).attr("stroke", "red");
          selectedNodeRef.current = null;
          svg.selectAll("circle").attr("fill", "#69b3a2");
        });

      // Aktualizacja węzłów
      const nodeSelection = nodeGroup
        .selectAll("circle")
        .data(d3Nodes, (d: any) => d.id);
      nodeSelection.exit().remove();
      const nodeEnter = nodeSelection
        .enter()
        .append("circle")
        .attr("r", 10)
        .attr("fill", "#69b3a2")
        .call(
          d3
            .drag<SVGCircleElement, NodeDatum>()
            .on("start", (event, d) => {
              if (!event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
            })
            .on("drag", (event, d) => {
              d.fx = event.x;
              d.fy = event.y;
            })
            .on("end", (event, d) => {
              if (!event.active) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            })
        )
        .on("click", (event, d) => {
          event.stopPropagation();
          selectedEdgeRef.current = null;
          svg.selectAll("line").attr("stroke", "#ccc");
          if (!selectedNodeRef.current) {
            selectedNodeRef.current = d;
            d3.select(event.currentTarget).attr("fill", "orange");
          } else if (selectedNodeRef.current.id === d.id) {
            selectedNodeRef.current = null;
            d3.select(event.currentTarget).attr("fill", "#69b3a2");
          } else {
            // Jeśli kliknięty inny węzeł, twórz krawędź (jeśli nie istnieje)
            const sourceId = selectedNodeRef.current.id;
            const targetId = d.id;
            const exists = d3Links.some(
              (link) =>
                (link.source === sourceId && link.target === targetId) ||
                (link.source === targetId && link.target === sourceId)
            );
            if (!exists) {
              const newLink: LinkDatum = {
                id: `${sourceId}-${targetId}`,
                source: sourceId,
                target: targetId,
              };
              d3Links.push(newLink);
              // Aktualizujemy stan grafu (przekazujemy nową tablicę krawędzi)
              setLinks([...links, [Number(sourceId), Number(targetId)]]);
              simulation
                .force<d3.ForceLink<NodeDatum, LinkDatum>>("link")!
                .links(d3Links);
              simulation.alpha(1).restart();
            }
            svg.selectAll("circle").attr("fill", "#69b3a2");
            selectedNodeRef.current = null;
          }
        });
      nodeEnter.merge(nodeSelection);
    }

    updateElements();

    // Kliknięcie w tło – czyści zaznaczenia
    svg.on("click", () => {
      selectedNodeRef.current = null;
      selectedEdgeRef.current = null;
      svg.selectAll("circle").attr("fill", "#69b3a2");
      svg.selectAll("line").attr("stroke", "#ccc");
    });

    return () => {
      simulation.stop();
      svg.selectAll("*").remove();
    };
  }, [d3Nodes, d3Links, links]);

  // Aktualizacja kolorów węzłów i krawędzi na podstawie simulationResult
  useEffect(() => {
    if (!simulationResult || !svgRef.current) return;
    // Tworzymy mapę: id węzła -> typ licencji
    const nodeLicenseMap = new Map<string, string>();
    simulationResult.forEach((assignment) => {
      assignment.item.forEach((item) => {
        item.users.forEach((user) => {
          nodeLicenseMap.set(String(user), assignment.license_type);
        });
      });
    });
    const svg = d3.select(svgRef.current);
    svg.selectAll("circle").attr("fill", (d: any) => {
      const lt = nodeLicenseMap.get(d.id);
      return lt ? licenseColors[lt] || "#69b3a2" : "#69b3a2";
    });
    svg.selectAll("line").attr("stroke", (d: any) => {
      const sourceId = d.source.id ? d.source.id : d.source;
      const targetId = d.target.id ? d.target.id : d.target;
      const sourceLT = nodeLicenseMap.get(sourceId);
      const targetLT = nodeLicenseMap.get(targetId);
      if (sourceLT && targetLT && sourceLT === targetLT) {
        return licenseColors[sourceLT] || "#ccc";
      }
      return "#ccc";
    });
  }, [simulationResult]);

  // Funkcja dodająca nowy węzeł (bez krawędzi)
  const addNode = () => {
    const newId = String(d3Nodes.length);
    const newNode: NodeDatum = { id: newId };
    d3Nodes.push(newNode);
    setNodes([...nodes, Number(newId)]);
    simulationRef.current?.nodes(d3Nodes);
    simulationRef.current?.alpha(1).restart();
  };

  // Funkcja usuwająca zaznaczony element (węzeł lub krawędź)
  const removeSelected = () => {
    if (!svgRef.current) return;
    if (selectedNodeRef.current) {
      const nodeId = selectedNodeRef.current.id;
      const newNodes = d3Nodes.filter((n) => n.id !== nodeId);
      const newLinks = d3Links.filter(
        (l) => l.source !== nodeId && l.target !== nodeId
      );
      // Aktualizujemy lokalne tablice D3
      d3Nodes.length = 0;
      d3Nodes.push(...newNodes);
      d3Links.length = 0;
      d3Links.push(...newLinks);
      // Aktualizujemy stan w App.tsx
      setNodes(nodes.filter((n) => String(n) !== nodeId));
      setLinks(
        links.filter(([s, t]) => String(s) !== nodeId && String(t) !== nodeId)
      );
      selectedNodeRef.current = null;
      d3.select(svgRef.current).selectAll("circle").attr("fill", "#69b3a2");
    }
    if (selectedEdgeRef.current) {
      const edgeId = selectedEdgeRef.current.id;
      const newLinks = d3Links.filter((l) => l.id !== edgeId);
      d3Links.length = 0;
      d3Links.push(...newLinks);
      setLinks(links.filter(([s, t]) => `${s}-${t}` !== edgeId));
      selectedEdgeRef.current = null;
      d3.select(svgRef.current).selectAll("line").attr("stroke", "#ccc");
    }
  };

  return (
    <Box
      className={styles.view}
      style={{ width: "100%", height: "100%", position: "relative" }}
    >
      <svg ref={svgRef} style={{ width: "100%", height: "100%" }}></svg>
      <Box style={{ position: "absolute", bottom: 10, left: 10 }}>
        <Group>
          <Button onClick={addNode}>Dodaj węzeł</Button>
          <Button onClick={removeSelected}>Usuń zaznaczone</Button>
        </Group>
      </Box>
    </Box>
  );
}

export default View;
