import { LICENSE_COLORS } from "./colorUtils";

export type LicenseAssignment = {
  node: number;
  license_holder: number;
  license_type: string;
};

export function processAssignments(assignments: Record<string, any>): {
  licenseAssignments: LicenseAssignment[];
  licenseToColor: Record<string, string>;
} {
  const licenseAssignments: LicenseAssignment[] = [];
  const licenseToColor: Record<string, string> = {};
  let colorIndex = 0;

  Object.entries(assignments).forEach(([licenseType, assignmentArray]) => {
    assignmentArray.forEach((assignment: any) => {
      assignment.covered_nodes.forEach((node: number) => {
        licenseAssignments.push({
          node,
          license_holder: assignment.license_holder,
          license_type: licenseType,
        });
        if (!licenseToColor[licenseType]) {
          licenseToColor[licenseType] = LICENSE_COLORS[colorIndex];
          colorIndex++;
        }
      });
    });
  });

  return { licenseAssignments, licenseToColor };
}
