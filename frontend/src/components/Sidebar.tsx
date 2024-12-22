import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { useState, useEffect } from "react";

interface SidebarProps {
  selectedAlgorithm: string | null;
  setSelectedAlgorithm: (algorithm: string) => void;
  selectedNetwork: string;
  setSelectedNetwork: (network: string) => void;
  maxGroupSize: number;
  setMaxGroupSize: (size: number) => void;
  groupLicensePrice: number;
  setGroupLicensePrice: (price: number) => void;
  individualLicensePrice: number;
  setIndividualLicensePrice: (price: number) => void;
  simulationResult: {
    individual: string[];
    group_owner: string[];
    group_member: string[];
  } | null;
}

function Sidebar({
  selectedAlgorithm,
  setSelectedAlgorithm,
  selectedNetwork,
  setSelectedNetwork,
  maxGroupSize,
  setMaxGroupSize,
  groupLicensePrice,
  setGroupLicensePrice,
  individualLicensePrice,
  setIndividualLicensePrice,
  simulationResult,
}: SidebarProps) {
  const [algorithms, setAlgorithms] = useState<string[]>([]);
  const networks = ["Basic Graph", "Star Graph", "Florentine Families Graph", "Les Miserables Graph"];

  // Pobieranie algorytmów
  useEffect(() => {
    fetch("http://localhost:8000/algorithms")
      .then((response) => response.json())
      .then((data) => {
        setAlgorithms(data);
      });
  }, []);

  // Ustawienie domyślnych wartości na start
  useEffect(() => {
    setSelectedNetwork("Basic Graph");
    setSelectedAlgorithm("Greedy");
  }, [setSelectedNetwork, setSelectedAlgorithm]);

  return (
    <div className="w-64 bg-background p-4 shadow-lg border-l border-border">
      <h2 className="mb-4 text-lg font-semibold">Parameters</h2>

      <div className="space-y-4">
        <div>
          <Label htmlFor="network-select">Select Network</Label>
          <Select onValueChange={setSelectedNetwork} defaultValue="Basic Graph">
            <SelectTrigger id="network-select">
              <SelectValue>{selectedNetwork || "Basic Graph"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {networks.map((network) => (
                <SelectItem key={network} value={network}>
                  {network}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="algorithm-select">Select Algorithm</Label>
          <Select onValueChange={setSelectedAlgorithm} defaultValue="Greedy">
            <SelectTrigger id="algorithm-select">
              <SelectValue>{selectedAlgorithm || "Greedy"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {algorithms.map((algorithm) => (
                <SelectItem key={algorithm} value={algorithm}>
                  {algorithm}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="max-group-size">Max Group Size</Label>
          <Slider
            id="max-group-size"
            min={2}
            max={10}
            step={1}
            value={[maxGroupSize]}
            onValueChange={(value) => setMaxGroupSize(value[0])}
          />
          <div className="mt-1 text-sm text-muted-foreground">{maxGroupSize}</div>
        </div>

        <div>
          <Label htmlFor="group-license-price">Group License Price</Label>
          <Input
            id="group-license-price"
            type="number"
            value={groupLicensePrice}
            onChange={(e) => setGroupLicensePrice(Number(e.target.value))}
          />
        </div>

        <div>
          <Label htmlFor="individual-license-price">Individual License Price</Label>
          <Input
            id="individual-license-price"
            type="number"
            value={individualLicensePrice}
            onChange={(e) => setIndividualLicensePrice(Number(e.target.value))}
          />
        </div>

        {simulationResult && (
          <div className="shadow-lg border-t border-border pt-4 space-y-2">
            <h3 className="text-lg font-bold">Simulation Results</h3>
            <p>
              <span className="font-semibold">Overall Cost:</span>{" "}
              {individualLicensePrice * simulationResult.individual.length +
                groupLicensePrice * simulationResult.group_owner.length}{" "}
              PLN
            </p>
            <p>
              <span className="font-semibold">Individual Licenses:</span> {simulationResult.individual.length}
            </p>
            <p>
              <span className="font-semibold">Group Owners:</span> {simulationResult.group_owner.length}
            </p>
            <p>
              <span className="font-semibold">Group Members:</span> {simulationResult.group_member.length}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Sidebar;
