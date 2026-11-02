/**
 * Returns the locale string for date formatting, derived from the current i18n language
 * (day-first for English; see languageToDateLocale).
 */
import { useTranslation } from 'react-i18next'
import { languageToDateLocale } from '../constants/i18n'

export function useDateLocale(): string {
  const { i18n } = useTranslation()
  return languageToDateLocale(i18n.language)
}
