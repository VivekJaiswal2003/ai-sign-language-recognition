import { useEffect, useMemo, useState } from 'react';

type Theme = 'paper' | 'forest' | 'night';

type DayCell = {
  date: Date;
  iso: string;
  inMonth: boolean;
};

const WEEK_DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const HERO_IMAGES: Record<Theme, string> = {
  paper:
    'https://images.unsplash.com/photo-1495904786722-d2b5a19a8537?auto=format&fit=crop&w=1200&q=80',
  forest:
    'https://images.unsplash.com/photo-1459262838948-3e2de6c1ec80?auto=format&fit=crop&w=1200&q=80',
  night:
    'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=80'
};

const HOLIDAY_LABELS: Record<string, string> = {
  '01-01': "New Year's Day",
  '07-04': 'Independence Day',
  '12-25': 'Christmas'
};

const getISO = (date: Date) => date.toISOString().slice(0, 10);

const sameDay = (a: Date, b: Date) => getISO(a) === getISO(b);

const inRange = (date: Date, start?: Date, end?: Date) => {
  if (!start || !end) return false;
  const value = date.getTime();
  return value > start.getTime() && value < end.getTime();
};

const normalizeRange = (a: Date, b: Date) =>
  a.getTime() <= b.getTime() ? [a, b] : [b, a];

const monthKey = (date: Date) => date.toLocaleString('en-US', { month: 'long', year: 'numeric' });

const buildGrid = (currentMonth: Date): DayCell[] => {
  const first = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
  const start = new Date(first);
  start.setDate(first.getDate() - first.getDay());

  return Array.from({ length: 42 }).map((_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    return {
      date,
      iso: getISO(date),
      inMonth: date.getMonth() === currentMonth.getMonth()
    };
  });
};

export function WallCalendar() {
  const [theme, setTheme] = useState<Theme>('paper');
  const [month, setMonth] = useState(() => new Date());
  const [rangeStart, setRangeStart] = useState<Date | undefined>();
  const [rangeEnd, setRangeEnd] = useState<Date | undefined>();
  const [monthlyNote, setMonthlyNote] = useState('');
  const [rangeNote, setRangeNote] = useState('');

  const grid = useMemo(() => buildGrid(month), [month]);
  const monthLabel = useMemo(() => monthKey(month), [month]);

  useEffect(() => {
    const storageKey = `calendar:month-note:${month.getFullYear()}-${month.getMonth() + 1}`;
    const note = localStorage.getItem(storageKey) ?? '';
    setMonthlyNote(note);
  }, [month]);

  useEffect(() => {
    if (!rangeStart || !rangeEnd) {
      setRangeNote('');
      return;
    }
    const key = `calendar:range-note:${getISO(rangeStart)}_${getISO(rangeEnd)}`;
    setRangeNote(localStorage.getItem(key) ?? '');
  }, [rangeStart, rangeEnd]);

  const saveMonthlyNote = (value: string) => {
    setMonthlyNote(value);
    const storageKey = `calendar:month-note:${month.getFullYear()}-${month.getMonth() + 1}`;
    localStorage.setItem(storageKey, value);
  };

  const saveRangeNote = (value: string) => {
    setRangeNote(value);
    if (!rangeStart || !rangeEnd) return;
    const key = `calendar:range-note:${getISO(rangeStart)}_${getISO(rangeEnd)}`;
    localStorage.setItem(key, value);
  };

  const onDateClick = (picked: Date) => {
    if (!rangeStart || (rangeStart && rangeEnd)) {
      setRangeStart(picked);
      setRangeEnd(undefined);
      return;
    }

    const [start, end] = normalizeRange(rangeStart, picked);
    setRangeStart(start);
    setRangeEnd(end);
  };

  const clearSelection = () => {
    setRangeStart(undefined);
    setRangeEnd(undefined);
    setRangeNote('');
  };

  return (
    <main className={`calendar-page theme-${theme}`}>
      <article className="wall-calendar">
        <section className="hero-panel" style={{ backgroundImage: `url(${HERO_IMAGES[theme]})` }}>
          <div className="hero-overlay">
            <h1>Wall Calendar</h1>
            <p>{monthLabel}</p>
            <div className="theme-switch">
              {(['paper', 'forest', 'night'] as Theme[]).map((option) => (
                <button
                  key={option}
                  className={option === theme ? 'active' : ''}
                  onClick={() => setTheme(option)}
                  type="button"
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="grid-panel">
          <header className="toolbar">
            <button type="button" onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() - 1, 1))}>
              ← Prev
            </button>
            <h2>{monthLabel}</h2>
            <button type="button" onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() + 1, 1))}>
              Next →
            </button>
          </header>

          <div className="weekdays">
            {WEEK_DAYS.map((day) => (
              <span key={day}>{day}</span>
            ))}
          </div>

          <div className="calendar-grid" role="grid" aria-label="Calendar days">
            {grid.map((cell) => {
              const mmdd = cell.iso.slice(5);
              const isHoliday = HOLIDAY_LABELS[mmdd];
              const start = rangeStart && sameDay(cell.date, rangeStart);
              const end = rangeEnd && sameDay(cell.date, rangeEnd);
              const between = inRange(cell.date, rangeStart, rangeEnd);

              return (
                <button
                  key={cell.iso}
                  type="button"
                  onClick={() => onDateClick(cell.date)}
                  className={[
                    'day-cell',
                    !cell.inMonth ? 'muted' : '',
                    start ? 'start' : '',
                    end ? 'end' : '',
                    between ? 'between' : ''
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  aria-label={`${cell.iso}${isHoliday ? ` ${isHoliday}` : ''}`}
                >
                  <span>{cell.date.getDate()}</span>
                  {isHoliday ? <small>{isHoliday}</small> : null}
                </button>
              );
            })}
          </div>
        </section>

        <section className="notes-panel">
          <header>
            <h3>Notes</h3>
            <button type="button" onClick={clearSelection}>
              Clear Range
            </button>
          </header>

          <label>
            Month memo
            <textarea
              value={monthlyNote}
              onChange={(event) => saveMonthlyNote(event.target.value)}
              placeholder="Plan deadlines, birthdays, or reminders..."
            />
          </label>

          <label>
            {rangeStart && rangeEnd
              ? `Range memo (${getISO(rangeStart)} → ${getISO(rangeEnd)})`
              : 'Range memo (select start and end date)'}
            <textarea
              value={rangeNote}
              onChange={(event) => saveRangeNote(event.target.value)}
              placeholder="Attach notes to your selected range..."
              disabled={!rangeStart || !rangeEnd}
            />
          </label>
        </section>
      </article>
    </main>
  );
}
