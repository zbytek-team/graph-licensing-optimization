// App.tsx
import { Box } from "@mantine/core";
import { useState } from "react";
import styles from "./App.module.scss";
import TopBar from "./components/TopBar";
import SimParams from "./components/SimParams";
import SimResults from "./components/SimResults";
import View from "./components/View";

// Interfejsy wyników symulacji – zgodnie z response z API
export interface LicenseAssignmentItem {
  owner: number;
  users: number[];
}

export interface LicenseAssignment {
  license_type: string;
  item: LicenseAssignmentItem[];
}

// Wynik symulacji to tablica obiektów LicenseAssignment
export type SimulationResponse = LicenseAssignment[];

function App() {
  // Stan wyniku symulacji
  const [simulationResult, setSimulationResult] =
    useState<SimulationResponse | null>(null);

  // Stan grafu – przechowujemy listę węzłów (liczby) oraz krawędzi (tablice dwóch liczb)
  const [graphNodes, setGraphNodes] = useState<number[]>([0, 1, 2, 3, 4]);
  const [graphEdges, setGraphEdges] = useState<number[][]>([
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 3],
    [1, 4],
  ]);

  // Funkcja wywołująca API symulacji – używa aktualnego stanu grafu
  const runSimulation = async (simulationRequest?: any) => {
    const req = simulationRequest || {
      algorithm: "greedy",
      license_types: [
        { name: "individual", cost: 10, limit: 1 },
        { name: "group", cost: 25, limit: 6 },
        { name: "mega", cost: 40, limit: 10 },
      ],
      graph: {
        nodes: graphNodes,
        edges: graphEdges,
      },
    };

    try {
      const response = await fetch("http://localhost:8000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      if (!response.ok) {
        throw new Error("Simulation error");
      }
      const data: SimulationResponse = await response.json();
      setSimulationResult(data);
    } catch (error) {
      console.error("Error during simulation:", error);
    }
  };

  return (
    <Box className={styles.container}>
      <Box className={styles.left}>
        {/* TopBar – przycisk "Run Simulation" wywołuje funkcję z aktualnym stanem grafu */}
        <TopBar onRunSimulation={() => runSimulation()} />
        {/* View – interaktywny widok grafu; otrzymuje aktualny stan grafu oraz funkcje do jego modyfikacji */}
        <View
          simulationResult={simulationResult}
          nodes={graphNodes}
          links={graphEdges}
          setNodes={setGraphNodes}
          setLinks={setGraphEdges}
        />
      </Box>
      <Box className={styles.right}>
        {/* SimParams – umożliwia zmianę parametrów symulacji */}
        <SimParams />
        {/* SimResults – wyświetla wyniki symulacji */}
        <SimResults simulationResult={simulationResult} />
      </Box>
    </Box>
  );
}

export default App;
