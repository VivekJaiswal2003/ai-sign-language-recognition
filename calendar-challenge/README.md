# Interactive Wall Calendar (Frontend Challenge)

This folder contains a standalone React + Vite implementation of the interactive wall calendar challenge.

## What I built

- **Wall calendar aesthetic** with a hero image panel and month grid as the main anchor.
- **Date range selection** (start/end) with clear highlighted states for start, end, and in-between days.
- **Integrated notes**:
  - month-level memo
  - range-specific memo (enabled after selecting both start and end)
- **Responsive layout**:
  - desktop: three columns (hero, calendar grid, notes)
  - mobile/tablet: stacked layout with full touch support
- **Extra polish**:
  - theme switcher (paper/forest/night)
  - basic holiday markers
  - localStorage persistence for monthly and range notes

## Run locally

```bash
cd calendar-challenge
npm install
npm run dev
```

Then open the URL printed by Vite (usually `http://localhost:5173`).

## Build for production

```bash
npm run build
npm run preview
```

## Architecture notes

- `src/WallCalendar.tsx` handles date math, selection state, and localStorage sync.
- `src/styles.css` provides the wall-calendar look and responsive behavior.
- `src/main.tsx` boots the component.

## Submission checklist hints

For a real submission package, include:
1. A public repo link.
2. A short video walkthrough (Loom/YouTube) showing range select, notes, and responsive behavior.
3. Optional deployed demo (Vercel/Netlify).
