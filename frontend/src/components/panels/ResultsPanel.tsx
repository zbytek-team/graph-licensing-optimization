import { Card, CardContent } from "@/components/ui/card";
import { useAppStore } from "@/store/useAppStore";

export default function ResultsPanel() {
  const assignments = useAppStore((state) => state.assignments);
  const licenses = useAppStore((state) => state.licenses);

  if (!assignments || licenses.length === 0) {
    return (
      <Card className="h-full">
        <CardContent className="p-4 flex gap-2 text-gray-400">
          Waiting for solver results...
        </CardContent>
      </Card>
    );
  }

  const licenseCostMap = Object.fromEntries(
    licenses.map((license) => [license.name, license.cost])
  );
  let totalCost = 0;
  const licenseCounts: Record<string, number> = {};

  for (const licenseType in assignments) {
    const count = assignments[licenseType].length;
    const cost = licenseCostMap[licenseType] || 0;
    totalCost += count * cost;
    licenseCounts[licenseType] = count;
  }

  return (
    <Card className="h-full">
      <CardContent className="p-4 space-y-2">
        <div className="flex gap-2">
          <p className="font-bold">Total Cost:</p>
          <p>{totalCost.toFixed(2)}</p>
        </div>
        <div className="mt-4">
          <p className="font-bold">Assignments per License:</p>
          <ul className="list-disc pl-4">
            {Object.entries(licenseCounts).map(([license, count]) => (
              <li key={license}>
                {license}: {count}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
