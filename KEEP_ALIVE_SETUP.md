# Keep-Alive Setup for Render

Prevent your Render app from going to sleep (cold starts) so visitors from LinkedIn have a fast experience.

## The Problem

Render free tier apps sleep after 15 minutes of inactivity. First visitor waits 30-50 seconds for the server to wake up.

## The Solution

Use an external service to ping your `/health` endpoint every 5-10 minutes to keep it warm.

---

## Option 1: UptimeRobot (Recommended - Free)

**Best for:** Simple, reliable, free monitoring

### Setup Steps:

1. **Sign up:** Go to https://uptimerobot.com (free account)
2. **Add Monitor:**
   - Click "Add New Monitor"
   - **Monitor Type:** HTTP(s)
   - **Friendly Name:** GOATlens Keep-Alive
   - **URL:** `https://goatlens.onrender.com/health`
   - **Monitoring Interval:** 5 minutes (free tier allows this)
   - Click "Create Monitor"

3. **Done!** UptimeRobot will ping your app every 5 minutes, keeping it warm.

**Why this works:** Every 5 minutes, UptimeRobot hits your `/health` endpoint, which counts as "activity" and prevents Render from sleeping.

---

## Option 2: cron-job.org (Free)

**Best for:** More control over timing

### Setup Steps:

1. **Sign up:** Go to https://cron-job.org (free account)
2. **Create Job:**
   - **Title:** GOATlens Keep-Alive
   - **Address:** `https://goatlens.onrender.com/health`
   - **Schedule:** Every 10 minutes (`*/10 * * * *`)
   - Click "Create Cronjob"

3. **Done!** Your app will be pinged every 10 minutes.

---

## Option 3: GitHub Actions (Free, but more setup)

**Best for:** If you want it tied to your repo

### Setup Steps:

1. Create `.github/workflows/keep-alive.yml` in your repo:

```yaml
name: Keep Render Alive

on:
  schedule:
    # Run every 10 minutes
    - cron: '*/10 * * * *'
  workflow_dispatch: # Allow manual trigger

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Render
        run: |
          curl -f https://goatlens.onrender.com/health || exit 1
```

2. **Enable GitHub Actions** in your repo settings
3. **Done!** GitHub will ping your app every 10 minutes.

---

## Option 4: Simple Script (Local/Manual)

If you want to run it manually or from your computer:

```bash
# keep-alive.sh
#!/bin/bash
while true; do
  curl -s https://goatlens.onrender.com/health > /dev/null
  echo "$(date): Pinged Render"
  sleep 600  # Wait 10 minutes (600 seconds)
done
```

Run with: `chmod +x keep-alive.sh && ./keep-alive.sh`

**Note:** This only works while your computer is on and the script is running.

---

## Recommended: UptimeRobot

**Why UptimeRobot is best:**
- ✅ Free forever
- ✅ No code changes needed
- ✅ Reliable (they've been around for years)
- ✅ Easy setup (2 minutes)
- ✅ Also monitors your app (alerts if it goes down)

---

## Verify It's Working

After setting up, check:

1. **Wait 15 minutes** (Render's sleep threshold)
2. **Visit:** https://goatlens.onrender.com
3. **Should load instantly** (no 30-50 second wait)

If it still sleeps, check:
- Is the keep-alive service actually pinging? (Check UptimeRobot dashboard)
- Is the URL correct? (Must be `https://goatlens.onrender.com/health`)
- Is the interval frequent enough? (5-10 minutes is ideal)

---

## Cost

All options above are **free**. Render free tier allows external pings to keep your app warm.

---

## For Your LinkedIn Post

Once set up, your app will:
- ✅ Load instantly for visitors
- ✅ No cold start delays
- ✅ Professional experience

**Pro tip:** Mention in your LinkedIn post that the app is live and ready to use - the keep-alive ensures it actually is!

---

**Last Updated:** February 2026
