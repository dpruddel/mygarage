import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '../../__tests__/test-utils'
import type { FleetHealth } from '../../types/dashboard'
import { METRIC_UNITS } from '../../__tests__/factories'

// FleetHealthStrip -> useCurrencyPreference/useUnitPreference -> useAuth. The
// shared render has no AuthProvider, so stub the context (finding 3). Pin units
// to metric so the mileage format ("160,000 km") is deterministic.
vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: null, isAuthenticated: false }),
}))
vi.mock('../../hooks/useUnitPreference', () => {
  const pref = () => ({
    system: 'metric',
    showBoth: false,
    gallonStandard: 'us',
    // The RESOLVED set, not just the collapsed system: the strip reads its
    // mileage through a unit formatter, which closes over `units`.
    units: METRIC_UNITS,
  })
  // The strip formats its one row through `useUnitFormatFor`, which reads the
  // ACCOUNT hook (#172).
  return { useUnitPreference: pref, useAccountUnitPreference: pref }
})

import FleetHealthStrip from '../FleetHealthStrip'

const FLEET: FleetHealth = {
  overdue_count: 3,
  upcoming_30d_count: 2,
  year: 2026,
  spent_this_year: '1234.50',
  next_due: {
    vin: 'TEST0000000000001',
    label: 'Oil change soon',
    due_date: '2026-08-01',
    due_mileage_km: null,
  },
}

describe('FleetHealthStrip', () => {
  it('renders the four cells with real values (counts, spent, year, next-due date)', () => {
    render(<FleetHealthStrip fleet={FLEET} />)
    // Counts (data).
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    // Captions (i18n keys under the vitest mock).
    expect(screen.getByText('dashboard.fleet.overdueCaption')).toBeInTheDocument()
    expect(screen.getByText('dashboard.fleet.upcomingCaption')).toBeInTheDocument()
    // Spent value — real formatCurrency('1234.50') output.
    expect(screen.getByText(/1,234\.50/)).toBeInTheDocument()
    // Year — rendered as data adjacent to the "Spent" caption.
    expect(
      screen.getByText('dashboard.fleet.spentLabel', { exact: false }).textContent,
    ).toContain('2026')
    // Next-due label (data) + formatted date.
    expect(screen.getByText('Oil change soon')).toBeInTheDocument()
    expect(screen.getByText(/1 Aug 2026/)).toBeInTheDocument()
  })

  it('renders the next-due mileage (unit-formatted) for a mileage-only reminder', () => {
    render(
      <FleetHealthStrip
        fleet={{
          ...FLEET,
          next_due: {
            vin: 'TEST0000000000001',
            label: 'Brakes at 160k',
            due_date: null,
            due_mileage_km: '160000.00',
          },
        }}
      />,
    )
    expect(screen.getByText('Brakes at 160k')).toBeInTheDocument()
    // 160000 km formatted via the user's (metric) distance preference.
    const mileage = screen.getByText(/160[,.\s]?000/)
    expect(mileage.textContent).toMatch(/km/)
    // No date row for a mileage-only reminder.
    expect(screen.queryByText(/1 Aug 2026/)).not.toBeInTheDocument()
  })

  it("renders the next-due mileage in the due vehicle's own unit, not the account's (#172)", () => {
    render(
      <FleetHealthStrip
        fleet={{
          ...FLEET,
          next_due: {
            vin: 'TEST0000000000001',
            label: 'Rotate at 10k',
            due_date: null,
            // 16093.44 km / 1.609344 = 10,000 mi exactly.
            due_mileage_km: '16093.44',
            distance_unit: 'mi',
          },
        }}
      />,
    )
    expect(screen.getByText(/10,000 mi/)).toBeInTheDocument()
    expect(screen.queryByText(/16,093 km/)).not.toBeInTheDocument()
  })

  it('shows the nothing-scheduled label when next_due is null', () => {
    render(<FleetHealthStrip fleet={{ ...FLEET, next_due: null }} />)
    expect(screen.getByText('dashboard.fleet.nextDueNone')).toBeInTheDocument()
  })
})
