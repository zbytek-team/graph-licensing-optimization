import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, X } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { License } from "@/types/license";

const defaultLicenses: License[] = [
  { name: "Individual", cost: 1.2, limit: 1 },
  { name: "Family", cost: 1.8, limit: 6 },
];

export default function LicensesPanel() {
  const [licenses, setLocalLicenses] = useState<License[]>(defaultLicenses);
  const setLicenses = useAppStore((state) => state.setLicenses);

  useEffect(() => {
    setLicenses(licenses);
  }, [licenses, setLicenses]);

  const addLicense = () => {
    const newLicense: License = { name: "", cost: 0, limit: 1 };
    setLocalLicenses([...licenses, newLicense]);
  };

  const updateLicense = (
    index: number,
    field: keyof License,
    value: string | number
  ) => {
    const updatedLicenses = licenses.map((license, i) =>
      i === index
        ? { ...license, [field]: field === "cost" ? Number(value) : value }
        : license
    );
    setLocalLicenses(updatedLicenses);
  };

  const deleteLicense = (index: number) => {
    setLocalLicenses(licenses.filter((_, i) => i !== index));
  };

  return (
    <Card className="h-full">
      <CardContent className="p-4">
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Button onClick={addLicense}>
              <Plus className="mr-2 h-4 w-4" /> Add License
            </Button>
          </div>
          <ScrollArea className="h-[500px]">
            {licenses.map((license, index) => (
              <div key={index} className="mb-4 p-4 border rounded-md">
                <div className="flex justify-between items-center mb-2">
                  <Input
                    placeholder="License name"
                    value={license.name}
                    onChange={(e) =>
                      updateLicense(index, "name", e.target.value)
                    }
                    className="w-3/4"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteLicense(index)}
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
                        updateLicense(index, "cost", e.target.value)
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
                        updateLicense(index, "limit", value[0])
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
