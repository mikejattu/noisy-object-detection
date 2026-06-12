# Wiring Google Sheets as a data backend

This lets every participant's responses land automatically in a spreadsheet.
The experiment still downloads a local CSV as a fallback in case the network request fails.

## Step 1 — Create the Google Sheet

1. Go to [sheets.new](https://sheets.new) and create a blank sheet.
2. Name it (e.g. "Noisy Recognition Study").

## Step 2 — Deploy the Apps Script

1. Inside the Sheet: **Extensions → Apps Script**.
2. Delete any existing code. Paste this exactly:

```javascript
function doPost(e) {
  const data = JSON.parse(e.postData.contents);
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  const HEADER = [
    'participant_id', 'participant_name', 'timestamp',
    'image_id', 'corruption', 'severity',
    'correct_class', 'distractor_class', 'chosen_class', 'is_correct', 'rt_ms',
  ];
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADER);
  }

  const ts = new Date().toISOString();
  data.rows.forEach(row => {
    sheet.appendRow([
      data.participant_id, data.participant_name, ts,
      row.image_id, row.corruption, row.severity,
      row.correct_class, row.distractor_class, row.chosen_class, row.is_correct, row.rt_ms,
    ]);
  });

  return ContentService
    .createTextOutput(JSON.stringify({ status: 'ok' }))
    .setMimeType(ContentService.MimeType.JSON);
}
```

3. Click **Deploy → New deployment**.
4. Type: **Web App**.
5. Settings:
   - **Execute as:** Me
   - **Who has access:** Anyone
6. Click **Deploy** → copy the **Web App URL**.

## Step 3 — Paste the URL into index.html

Open `docs/index.html` and find this line near the top of the `<script>` block:

```javascript
const SHEETS_ENDPOINT = '';   // ← paste your Apps Script Web App URL here
```

Replace the empty string with your URL:

```javascript
const SHEETS_ENDPOINT = 'https://script.google.com/macros/s/YOUR_ID/exec';
```

Commit and push — the experiment will now POST each participant's rows to the sheet automatically.

## Data columns

| Column | Description |
|---|---|
| `participant_id` | Auto-generated session ID (e.g. `M4CXA7`) |
| `participant_name` | Name entered at the login screen |
| `timestamp` | ISO-8601 server time when data arrived |
| `image_id` | Source image filename stem |
| `corruption` | `clean` or one of the 8 corruption types |
| `severity` | 0 (clean) or 3 (main experiment) |
| `correct_class` | Ground-truth Imagenette class name |
| `distractor_class` | The wrong-class foil shown alongside the correct one |
| `chosen_class` | Class name the participant clicked |
| `is_correct` | 1 if correct, 0 if not |
| `rt_ms` | Response time in milliseconds (from choice screen appearing to click) |
