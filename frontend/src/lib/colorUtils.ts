// colors for the license types, add more if needed
export const LICENSE_COLORS = [
    "#073b4c", // dark blueish
    "#ef476f", // redish
    "#06d6a0", // greenish
    "#ffd166", // yellowish
    "#118ab2", // light blueish
  ];
  

export function adjustColor(color: string, variance: number = 30): string {
    if (!/^#([0-9A-Fa-f]{6})$/.test(color)) {
      throw new Error("Invalid color format. Use #RRGGBB.");
    }
  
    let r = parseInt(color.slice(1, 3), 16);
    let g = parseInt(color.slice(3, 5), 16);
    let b = parseInt(color.slice(5, 7), 16);
  
    const clamp = (value: number) => Math.max(0, Math.min(255, value));
  
    r = Math.round(clamp(r + (Math.random() * 2 - 1) * variance));
    g = Math.round(clamp(g + (Math.random() * 2 - 1) * variance));
    b = Math.round(clamp(b + (Math.random() * 2 - 1) * variance));
  
    return `#${r.toString(16).padStart(2, "0")}${g
      .toString(16)
      .padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }
  