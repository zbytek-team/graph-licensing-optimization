export interface Assignment {
  license_holder: number;
  covered_nodes: number[];
}

export type AssignmentRecord = Record<string, Assignment[]>;
