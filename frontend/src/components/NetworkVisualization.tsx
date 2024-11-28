"use client";

import { Background, Controls, Edge, Node, OnEdgesChange, OnNodesChange, ReactFlow, addEdge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback } from "react";

interface NetworkVisualizationProps {
  nodes: Node[]
  edges: Edge[]
  onNodesChange: OnNodesChange
  onEdgesChange: OnEdgesChange
  setEdges: (edges: any) => void // TODO: Fix type
}

function NetworkVisualization({ nodes, edges, onNodesChange, onEdgesChange, setEdges }: NetworkVisualizationProps) {
  const onConnect = useCallback(
    (params: any) => setEdges((eds: any) => addEdge(params, eds)), // TODO: Fix type
    []
  );

  return (
    <div className="flex-1">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodesDraggable={true}
        colorMode="dark"
        fitView
        className="bg-background text-foreground"
      >
        <Controls className="bg-background text-foreground" />
        <Background color="#374151" />
      </ReactFlow>
    </div>
  );
}

export default NetworkVisualization;
