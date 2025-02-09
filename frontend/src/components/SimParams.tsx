// SimParams.tsx
import { Box, Select, Text, Slider, Fieldset, Button } from "@mantine/core";
import { useState } from "react";
import styles from "../App.module.scss";

function SimParams() {
  const [selectedNetwork, setSelectedNetwork] = useState("network1");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("greedy");
  const [individualCost, setIndividualCost] = useState(10);
  const [individualLimit, setIndividualLimit] = useState(1);
  const [groupCost, setGroupCost] = useState(25);
  const [groupLimit, setGroupLimit] = useState(6);

  return (
    <Box className={styles.simparams}>
      <Fieldset
        className={styles.fieldset}
        legend="Simulation Parameters"
        variant="unstyled"
      >
        <Select
          label="Select network"
          placeholder="Select network"
          data={[
            { value: "network1", label: "Network 1" },
            { value: "network2", label: "Network 2" },
            { value: "network3", label: "Network 3" },
          ]}
          value={selectedNetwork}
          onChange={(value) => value && setSelectedNetwork(value)}
        />
        <Select
          label="Select algorithm"
          placeholder="Select algorithm"
          data={[{ value: "greedy", label: "Greedy" }]}
          value={selectedAlgorithm}
          onChange={(value) => value && setSelectedAlgorithm(value)}
        />
        <Box mt="md">
          <Text>Individual License</Text>
          <Text>Cost</Text>
          <Slider
            value={individualCost}
            onChange={setIndividualCost}
            min={0}
            max={50}
            step={1}
            labelAlwaysOn
          />
          <Text>Limit</Text>
          <Slider
            value={individualLimit}
            onChange={setIndividualLimit}
            min={1}
            max={10}
            step={1}
            labelAlwaysOn
          />
        </Box>
        <Box mt="md">
          <Text>Group License</Text>
          <Text>Cost</Text>
          <Slider
            value={groupCost}
            onChange={setGroupCost}
            min={0}
            max={50}
            step={1}
            labelAlwaysOn
          />
          <Text>Limit</Text>
          <Slider
            value={groupLimit}
            onChange={setGroupLimit}
            min={1}
            max={10}
            step={1}
            labelAlwaysOn
          />
        </Box>
      </Fieldset>
    </Box>
  );
}

export default SimParams;
