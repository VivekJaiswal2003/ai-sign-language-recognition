import React from 'react';
import { createRoot } from 'react-dom/client';
import { WallCalendar } from './WallCalendar';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <WallCalendar />
  </React.StrictMode>
);
