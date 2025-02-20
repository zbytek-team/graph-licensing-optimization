"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Plus, X } from "lucide-react";
import { useState } from "react";

type License = {
  id: number;
  name: string;
  cost: number;
  limit: number;
};

const defaultLicenses: License[] = [
  { id: 1, name: "Individual", cost: 1.2, limit: 1 },
  { id: 2, name: "Family", cost: 1.8, limit: 6 },
];

export default function SidePanel() {
  const [licenses, setLicenses] = useState<License[]>(defaultLicenses);
  const [nextId, setNextId] = useState(3);

  const addLicense = () => {
    const newLicense: License = {
      id: nextId,
      name: "",
      cost: 0,
      limit: 1,
    };
    setLicenses([...licenses, newLicense]);
    setNextId(nextId + 1);
  };

  const updateLicense = (
    id: number,
    field: keyof License,
    value: string | number
  ) => {
    setLicenses(
      licenses.map((license) =>
        license.id === id
          ? {
              ...license,
              [field]:
                field === "cost" ? Number.parseFloat(value as string) : value,
            }
          : license
      )
    );
  };

  const deleteLicense = (id: number) => {
    setLicenses(licenses.filter((license) => license.id !== id));
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4">
        <div className="space-y-4">
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
            <Button onClick={addLicense}>
              <Plus className="mr-2 h-4 w-4" /> Add License
            </Button>
          </div>
          <ScrollArea className="h-[calc(100vh-300px)]">
            {licenses.map((license) => (
              <div key={license.id} className="mb-4 p-4 border rounded-md">
                <div className="flex justify-between items-center mb-2">
                  <Input
                    placeholder="License name"
                    value={license.name}
                    onChange={(e) =>
                      updateLicense(license.id, "name", e.target.value)
                    }
                    className="w-3/4"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteLicense(license.id)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <div className="space-y-2">
                  <div>
                    <label className="text-sm font-medium">Cost:</label>
                    <Input
                      type="number"
                      min={0}
                      step={0.01}
                      value={license.cost}
                      onChange={(e) =>
                        updateLicense(license.id, "cost", e.target.value)
                      }
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">
                      Group Limit: {license.limit}
                    </label>
                    <Slider
                      min={1}
                      max={20}
                      step={1}
                      value={[license.limit]}
                      onValueChange={(value) =>
                        updateLicense(license.id, "limit", value[0])
                      }
                    />
                  </div>
                </div>
              </div>
            ))}
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  );
}
