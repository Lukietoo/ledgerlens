# 90-second demo video

The shot list for a screen recording. Times are targets; the story is: raw bank text → the
decode → the numbers → asking a question → why it's cheap.

## Setup (5 minutes)

1. Use a **production build**. It's faster, and it hides Next's dev badge:
   ```bash
   .venv/bin/uvicorn ledgerlens.service:app --workers 1      # with ANTHROPIC_API_KEY exported, for chat
   cd web && npm run build && npm start                        # http://localhost:3000
   ```
2. Use a clean browser window at 1440×900, 110% zoom, with no extensions or bookmarks bar.
3. Click **Try it on a demo account** once before recording, so the first real load is warm.
   Reload the page.
4. Record with QuickTime (File → New Screen Recording), or Loom for a webcam bubble.
   Voiceover reads best recorded separately and laid over the video.

## Shots

| time | on screen | say |
|---|---|---|
| 0:00–0:08 | The landing page. The hero shows `PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BTL COF…` → **Blue Bottle Coffee**. | "This is what your bank actually sends. Every fintech has to turn it into something a person can read." |
| 0:08–0:18 | Click **Try it on a demo account**. The dashboard fills in; the hero cycles through real lines from the statement. | "Ledgerlens does that for a whole year of transactions in a fraction of a second: merchant, category, and what recurs." |
| 0:18–0:32 | Scroll to **Transactions**. Hover a row: raw text in grey under the clean name. Point at an amber **Claude** tag and an **unsure** tag. | "A LightGBM model handles most of it in about 13 milliseconds. When it's unsure, or it's never seen the merchant, it asks Claude, once per merchant, then caches the answer." |
| 0:32–0:42 | Glance at **Spending by month**, then **Subscriptions** and the **Next month** forecast. | "From there it's real numbers: subscriptions, bills, what you'll have left next month." |
| 0:42–1:05 | In the chat, click **Which subscriptions cost the most?** When it answers, open **Checked N calculations** to show the tool calls. Then type *"Can I afford a $1,200 trip next month?"* | "The agent never does math itself. Every number comes from a calculation you can inspect, and the statement text is treated as data, so a sketchy Venmo memo can't hijack it." |
| 1:05–1:22 | Cut to the README results table, then the cost table. | "On merchants it's never seen, it goes from 54% for rules to 81%. The hybrid matches running Claude on everything, at about $1 instead of $26 per million transactions, and it serves 11,000 transactions a second." |
| 1:22–1:30 | Back to the dashboard hero, or the GitHub repo page. | "Ledgerlens: bank statements, decoded. Code and evals are on GitHub." |

## Don't

- Don't show a terminal or `.env` while recording: API keys show up in scrollback.
- Don't upload your real bank CSV on camera. Use the demo account.
- Don't claim real-data accuracy in the voiceover until the hand-labeled eval exists. Say "on
  merchants it's never seen", which is true of the synthetic benchmark.
