import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '../../__tests__/test-utils'
import ServiceVisitAttachmentList from '../ServiceVisitAttachmentList'
import type { Attachment } from '../../types/attachment'

/**
 * Regression cover for #107: `download_url` arrives from the API as an
 * absolute path (`/api/attachments/1/download`), but axios already has
 * `/api` as its baseURL — passing the raw value produces `/api/api/...`.
 * `apiRelative()` strips the prefix.
 *
 * This test previously lived on ServiceAttachmentList, which was the legacy
 * ServiceRecord component and had become unreachable; the fix shipped to both
 * it and this one, but only the dead component was covered. Ported here so the
 * live path is the one under test.
 */

const attachment: Attachment = {
  id: 1,
  file_name: 'receipt.pdf',
  file_size: 1234,
  file_type: 'application/pdf',
  download_url: '/api/attachments/1/download',
  record_id: 5,
  record_type: 'service_visit',
  uploaded_at: '2025-01-06T12:00:00',
} as Attachment

const apiGet = vi.fn()

vi.mock('../../services/api', () => ({
  default: {
    get: (...args: unknown[]) => apiGet(...args),
  },
}))

// URL.createObjectURL is not implemented in jsdom
beforeEach(() => {
  apiGet.mockReset()
  apiGet.mockImplementation((url: string) => {
    if (url.endsWith('/attachments')) {
      return Promise.resolve({ data: { attachments: [attachment] } })
    }
    return Promise.resolve({ data: new Blob() })
  })
  window.URL.createObjectURL = vi.fn(() => 'blob:mock')
  window.URL.revokeObjectURL = vi.fn()
  // jsdom cannot navigate to the blob: URL a download anchor points at, and
  // logs "Not implemented: navigation to another Document" on every click.
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
})

describe('ServiceVisitAttachmentList download (baseURL-relative axios arg)', () => {
  it('calls api.get with the /api-stripped path, not the raw download_url', async () => {
    render(<ServiceVisitAttachmentList visitId={5} />)

    const downloadButton = await screen.findByRole('button', {
      name: /serviceVisitAttachments\.download|download/i,
    })
    fireEvent.click(downloadButton)

    await waitFor(() => {
      expect(apiGet).toHaveBeenLastCalledWith(
        '/attachments/1/download',
        expect.objectContaining({ responseType: 'blob' }),
      )
    })
  })
})

describe('ServiceVisitAttachmentList uploaded timestamp', () => {
  it('renders the upload date from the wire-shaped datetime', async () => {
    render(<ServiceVisitAttachmentList visitId={5} />)
    expect(await screen.findByText(/6 Jan 2025/)).toBeInTheDocument()
    expect(screen.queryByText(/Invalid Date/)).toBeNull()
  })
})
