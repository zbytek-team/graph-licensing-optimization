// TopBar.tsx
import { Box, Button, Title } from "@mantine/core";
import styles from "../App.module.scss";

interface TopBarProps {
  onRunSimulation: () => void;
}

function TopBar({ onRunSimulation }: TopBarProps) {
  return (
    <Box className={styles.topbar}>
      <Title className={styles.title}>Optimal License Distribution</Title>
      <Button className={styles.button} onClick={onRunSimulation}>
        Run Simulation
      </Button>
    </Box>
  );
}

export default TopBar;
