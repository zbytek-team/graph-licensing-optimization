import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"

import { Slider } from "./ui/slider"

interface SidebarProps {
  selectedNetwork: string
  setSelectedNetwork: (network: string) => void
  selectedAlgorithm: string
  setSelectedAlgorithm: (algorithm: string) => void
  maxGroupSize: number
  setMaxGroupSize: (size: number) => void
  groupLicensePrice: number
  setGroupLicensePrice: (price: number) => void
  individualLicensePrice: number
  setIndividualLicensePrice: (price: number) => void
}

const networks = ['Facebook', 'Twitter', 'LinkedIn', 'Custom']
const algorithms = ['Greedy', 'ILP', 'Genetic']


function Sidebar({ selectedNetwork, setSelectedNetwork, selectedAlgorithm, setSelectedAlgorithm, maxGroupSize, setMaxGroupSize, groupLicensePrice, setGroupLicensePrice, individualLicensePrice, setIndividualLicensePrice }: SidebarProps) {
  return (
    <div className="w-64 bg-background p-4 shadow-lg border-l border-border">
      <h2 className="mb-4 text-lg font-semibold">Parameters</h2>

      <div className="space-y-4">
        <div>
          <Label htmlFor="network-select">Select Network</Label>
          <Select value={selectedNetwork} onValueChange={setSelectedNetwork}>
            <SelectTrigger id="network-select">
              <SelectValue>{selectedNetwork}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {networks.map((network) => (
                <SelectItem key={network} value={network}>{network}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label htmlFor="algorithm-select">Select Algorithm</Label>
          <Select value={selectedAlgorithm} onValueChange={setSelectedAlgorithm}>
            <SelectTrigger id="algorithm-select">
              <SelectValue>{selectedAlgorithm}</SelectValue>
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
      </div>
    </div>
  )
}

export default Sidebar