"use client";

import NetworkVisualization from "@/components/NetworkVisualization";
import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";

import { useEdgesState, useNodesState } from "@xyflow/react";

import { useState } from "react";

const nodeStyle = {
  width: 50,
  height: 50,
  borderRadius: 50,
  fontSize: 18,
  backgroundColor: "#333333",
};

const initNodes = [
  { id: "1", data: { label: "1" }, position: { x: 0, y: 0 }, style: { ...nodeStyle } },
  { id: "2", data: { label: "2" }, position: { x: -100, y: 100 }, style: { ...nodeStyle } },
  { id: "3", data: { label: "3" }, position: { x: 100, y: 100 }, style: { ...nodeStyle } },
  { id: "4", data: { label: "4" }, position: { x: 100, y: 200 }, style: { ...nodeStyle } },
];

const initEdges = [
  { id: "e1-2", source: "1", target: "2" },
  { id: "e1-3", source: "1", target: "3" },
  { id: "e3-4", source: "3", target: "4" },
];

interface SimulationResult {
  individual: string[]
  group_owner: string[]
  group_member: string[]
}

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges);

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [maxGroupSize, setMaxGroupSize] = useState(6);
  const [groupLicensePrice, setGroupLicensePrice] = useState(349.99);
  const [individualLicensePrice, setIndividualLicensePrice] = useState(167.99);

  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);


  const handleAddNode = () => {
    const nodeNumber = (nodes.length + 1).toString();
    const newNode = {
      id: nodeNumber,
      data: { label: nodeNumber },
      position: { x: 0, y: 0 },
      style: { ...nodeStyle },
    };

    setNodes((nds) => nds.concat(newNode));
  };

  const handleRunSimulation = () => {
    fetch("http://localhost:8000/run-simulation?max_group_size=" + maxGroupSize + "&selected_algorithm=" + selectedAlgorithm, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        graph: {
          nodes: nodes,
          edges: edges,
        },
        prices: [individualLicensePrice, groupLicensePrice],
      },),
    })
      .then((response) => response.json())
      .then((data) => {
        setSimulationResult(data);

        setNodes((currentNodes) =>
          currentNodes.map((node) => {
            if (data.individual.includes(node.id)) {
              return {
                ...node,
                style: { ...node.style, backgroundColor: "#184a2e" },
              };
            } else if (data.group_owner.includes(node.id)) {
              return {
                ...node,
                style: { ...node.style, backgroundColor: "#251742" },
              };
            } else if (data.group_member.includes(node.id)) {
              return {
                ...node,
                style: { ...node.style, backgroundColor: "#18314a" },
              };
            }
            return {
              ...node,
              style: { ...node.style, backgroundColor: "#333333" },
            };
          })
        );
      })
      .catch((error) => console.error("Error during simulation:", error));
  };

  return (
    <div className="flex h-screen w-screen">
      <div className="flex flex-1 flex-col overflow-hidden">
        <Topbar onAddNode={handleAddNode} onRunSimulation={handleRunSimulation} />
        <NetworkVisualization
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          setEdges={setEdges}
        />
      </div>
      <Sidebar
        selectedAlgorithm={selectedAlgorithm}
        maxGroupSize={maxGroupSize}
        groupLicensePrice={groupLicensePrice}
        individualLicensePrice={individualLicensePrice}
        setSelectedAlgorithm={setSelectedAlgorithm}
        setMaxGroupSize={setMaxGroupSize}
        setGroupLicensePrice={setGroupLicensePrice}
        setIndividualLicensePrice={setIndividualLicensePrice}
        simulationResult={simulationResult}
      />
    </div>
  );
}

export default App;
