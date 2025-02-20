import { Card, CardContent } from "@/components/ui/card";

export default function ResultsPanel() {
  return (
    <Card className="h-full">
      <CardContent className="p-4 flex gap-2">
        <p className="font-bold">Total Cost: </p>
        <p>156.96</p>
      </CardContent>
    </Card>
  );
}
