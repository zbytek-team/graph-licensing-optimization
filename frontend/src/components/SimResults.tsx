// SimResults.tsx
import { Box, Text, Title } from "@mantine/core";
import styles from "../App.module.scss";

interface LicenseAssignmentItem {
  owner: number;
  users: number[];
}

interface LicenseAssignment {
  license_type: string;
  item: LicenseAssignmentItem[];
}

interface SimResultsProps {
  simulationResult: LicenseAssignment[] | null;
}

function SimResults({ simulationResult }: SimResultsProps) {
  return (
    <Box className={styles.simresults}>
      <Title order={4}>Simulation Results</Title>
      {simulationResult && simulationResult.length > 0 ? (
        simulationResult.map((assignment, index) => (
          <Box key={index} mt="md">
            <Text weight={500}>License Type: {assignment.license_type}</Text>
            {assignment.item.map((item, idx) => (
              <Box key={idx} ml="sm">
                <Text>
                  Owner: {item.owner}, Users: {item.users.join(", ")}
                </Text>
              </Box>
            ))}
          </Box>
        ))
      ) : (
        <Text>No simulation results yet.</Text>
      )}
    </Box>
  );
}

export default SimResults;
