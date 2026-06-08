/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0d1117", surface: "#161b22", border: "#30363d",
        muted: "#8b949e", text: "#c9d1d9",
        accent: "#58a6ff", success: "#3fb950", danger: "#f85149",
      },
    },
  },
  plugins: [],
};
