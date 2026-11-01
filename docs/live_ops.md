# Alma-Live Gas: operations

Data lead, 2026-11-01. This system does **paper trading only**. There is no broker API in `tools/live`; real money is
a separate decision for the user.

## Daily schedule (US/Eastern, trading days)

| ET time | job | command |
|---|---|---|
| 11:15, 15:20 | update storage (Thursday release 10:30 ET) and CPC (posted around 10:00-11:00 ET) | `update.py --only storage,cpc` |
| 15:30-15:50 | decide (window enforced by the wall clock; no backfill possible) | `paper.py decide --book <b> --model <m>` (one line per shadow book) |
| 18:15 | update prices (the day-d bar is published at 18:00 ET), then settle (fill at the 16:00 close, mark, risk checks) | `update.py --only px` then `paper.py settle --book <b>` |
| 18:30 | verify the chain and the git anchor | `paper.py verify --book <b>` |

- `decide` and `settle` auto-commit the ledger file only (`git commit -- runs/live/paper/<b>/ledger.jsonl`). The
  commit is the external anchor of the head hash. Nothing is ever pushed.
- `update.py` is idempotent and append-only:
  - a lock file prevents overlap;
  - an unchanged raw response is not re-stored;
  - already-held records are not re-appended;
  - each run appends to `data/live/update_log.jsonl`: what arrived, its published_at range and the new pin.
- Stale data at decision time forces the decision flat. Stale means: UNG bar older than d-1, storage older than
  9 days, or CPC older than 3 days.

## launchd / cron (NOT installed; the manager decides)

The crontab below runs on Mac local time. It assumes the Mac's clock is US/Eastern; if not, set `TZ` (cron honours
`CRON_TZ` on some systems, but macOS cron does not, so prefer launchd with the Mac set to Eastern):

```
# m  h   dom mon dow  command
15  11  *   *   1-5  cd /Users/srimanarayana/Timeseries && nice -n 10 .venv/bin/python tools/live/update.py --only storage,cpc >> runs/live/cron.log 2>&1
20  15  *   *   1-5  cd /Users/srimanarayana/Timeseries && nice -n 10 .venv/bin/python tools/live/update.py --only storage,cpc,px >> runs/live/cron.log 2>&1
31  15  *   *   1-5  cd /Users/srimanarayana/Timeseries && tools/live/shadow.sh decide >> runs/live/cron.log 2>&1   # first: tools/live/shadow.sh init (done 2026-11-01)
15  18  *   *   1-5  cd /Users/srimanarayana/Timeseries && nice -n 10 .venv/bin/python tools/live/update.py --only px && tools/live/shadow.sh settle >> runs/live/cron.log 2>&1
```

### Exact launchd agents (for the manager to install after the user approves; the data lead has NOT installed them)

StartCalendarInterval uses the Mac's **local** time zone, so these times assume the Mac is set to US/Eastern. Check
with `date +%Z` (EST or EDT). If the Mac is on another zone, convert the Hour values; the TZ variable only affects the
child process. Install each agent with `launchctl load -w ~/Library/LaunchAgents/<label>.plist`. Remove one with
`launchctl unload -w ...`.

`~/Library/LaunchAgents/com.alma.live.update_am.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.alma.live.update_am</string>
  <key>WorkingDirectory</key><string>/Users/srimanarayana/Timeseries</string>
  <key>EnvironmentVariables</key><dict><key>TZ</key><string>America/New_York</string></dict>
  <key>ProgramArguments</key><array>
    <string>/bin/zsh</string><string>-c</string><string>.venv/bin/python tools/live/update.py --only storage,cpc &gt;&gt; runs/live/cron.log 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>15</integer></dict>
  </array>
  <key>Nice</key><integer>10</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
```

`~/Library/LaunchAgents/com.alma.live.update_pre.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.alma.live.update_pre</string>
  <key>WorkingDirectory</key><string>/Users/srimanarayana/Timeseries</string>
  <key>EnvironmentVariables</key><dict><key>TZ</key><string>America/New_York</string></dict>
  <key>ProgramArguments</key><array>
    <string>/bin/zsh</string><string>-c</string><string>.venv/bin/python tools/live/update.py --only storage,cpc,px &gt;&gt; runs/live/cron.log 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>20</integer></dict>
  </array>
  <key>Nice</key><integer>10</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
```

`~/Library/LaunchAgents/com.alma.live.decide.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.alma.live.decide</string>
  <key>WorkingDirectory</key><string>/Users/srimanarayana/Timeseries</string>
  <key>EnvironmentVariables</key><dict><key>TZ</key><string>America/New_York</string></dict>
  <key>ProgramArguments</key><array>
    <string>/bin/zsh</string><string>-c</string><string>tools/live/shadow.sh decide &gt;&gt; runs/live/cron.log 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>31</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>31</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>31</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>31</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>31</integer></dict>
  </array>
  <key>Nice</key><integer>10</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
```

`~/Library/LaunchAgents/com.alma.live.settle.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.alma.live.settle</string>
  <key>WorkingDirectory</key><string>/Users/srimanarayana/Timeseries</string>
  <key>EnvironmentVariables</key><dict><key>TZ</key><string>America/New_York</string></dict>
  <key>ProgramArguments</key><array>
    <string>/bin/zsh</string><string>-c</string><string>.venv/bin/python tools/live/update.py --only px &amp;&amp; tools/live/shadow.sh settle &gt;&gt; runs/live/cron.log 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
  </array>
  <key>Nice</key><integer>10</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
```


The red-team GO conditions (2026-11-01) apply:
- `update.py` runs before 15:30 ET, so that fetched_at ≤ t;
- only `mode=live` books count; `runs/live/paper_sim` is dry-run only;
- COT, EDGAR and Open-Meteo stay off;
- WM models join the shadow loop only after the freeze audit (add them via `SHADOW_MODELS`).

Live books initialised and git-anchored on 2026-11-01: `runs/live/paper/shadow_{buy_hold, short_ung_carry, tsmom_20,
seasonal, weather_rev, storage_surprise, fc_ewma, fc_hist}` (genesis only). The first live decision is Monday
2026-11-02 at 15:30 ET. A dry run of the full chain on 2026-10-29/30 (sim books, `runs/live/paper_sim`) passed: decide,
settle, verify, with costs booked at 10 bp per side.

- Holidays: `paper.py decide` refuses non-trading days, so a holiday run exits without writing anything.
- A Mac asleep at 15:30 means a missed decision. It cannot be backfilled; the next day's decision is still allowed
  (the previous decision must be settled first). Keep the machine awake during market hours.

## Risk controls (paper book)

| control | trigger | action |
|---|---|---|
| daily-loss kill switch | day P&L ≤ -2% of start-of-day equity | `halt_day` line; the next decision is forced flat (flattened at that close) |
| drawdown halt | drawdown ≥ 10% from the equity peak | `halt_review`; flat until a human runs `paper.py resume --book <b> --note "..."` |
| stale data | inputs older than the limits above | decision forced flat, with the reason in the ledger |
| limits | UNG \|w\| ≤ 1, BOIL \|w\| ≤ 0.5, gross ≤ 1 | clipped / scaled |

## Re-scope notes (manager, 2026-11-01)

- Stopped. The code and PIT files are kept but not loaded by `Store()` by default:
  - COT (LRT-10 fixed, LRT-11 open);
  - EDGAR filings and facts;
  - Open-Meteo previous runs (rate-limited by HTTP 429; partial raw only);
  - the NG curve snapshotter (`update.py --snap-curve`);
  - BOIL work beyond the secondary weight.
- Not built: GEFS, GDELT, the IBKR/Alpaca broker.
- Baker Hughes is unavailable (HTTP 403, not bypassed).
- SEC requests need `SEC_USER_AGENT` set to a descriptive UA with operator contact (SEC fair-access policy) if EDGAR
  is re-enabled.

## Recovery

- **A PIT file fails its prefix pin.** `update.py` stops for that source. Investigate with `git diff data/live/pit`.
  Never "fix" this by editing the manifest.
- **A ledger fails `verify`.** The book is void from the first bad line. Open a new book. The red team re-verifies.
