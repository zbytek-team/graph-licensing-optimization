import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";

export default function GraphTypePanel() {
  return (
    <Card className="h-full">
      <CardContent className="p-4 flex gap-2">
        <div className="flex items-center space-x-2">
          <Select>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select graph type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="bipartite">Bipartite</SelectItem>
              <SelectItem value="star">Star</SelectItem>
              <SelectItem value="complete">Complete</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
}
