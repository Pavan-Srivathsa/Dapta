# CORD Receipt Extractor — Single-Image Contract

## Purpose

Extract line items and totals from one receipt image. This specialist is trained on CORD v2
receipts and accepts exactly one JPEG, PNG, or WebP image. Return raw JSON only.

## Required output

Return exactly:

```json
{
  "items": [
    {
      "name": "item description",
      "count": 1,
      "unit_price": 1000,
      "price": 1000
    }
  ],
  "subtotal": 1000,
  "discount": null,
  "service": null,
  "tax": null,
  "total": 1000
}
```

All seven top-level keys are required. `items` must be an array. Every item must contain
`name`, `count`, `unit_price`, and `price`. Numeric fields may be numbers or `null`.

## Extraction rules

1. Use only information visible in the supplied receipt image.
2. Remove currency symbols and thousands separators from amounts.
3. Use `null` when a numeric value is not printed or cannot be determined.
4. Do not invent merchant, address, date, currency, payment, or customer fields; they are
   outside this adapter's learned schema.
5. Return no Markdown, explanation, confidence score, or additional keys.

## Scope

The adapter was trained on Indonesian retail and restaurant receipts from CORD v2. Accuracy
may be lower for other languages, countries, handwritten receipts, or unfamiliar layouts.
