import pandas as pd
import asyncio
from telethon import TelegramClient


async def fetch_telegram_messages(
    api_id,
    api_hash,
    channel_username,
    start_date,
    end_date
):
    messages_data = []

    # --- timezone safety ---
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")

    async with TelegramClient("session_name", api_id, api_hash) as client:
        channel = await client.get_entity(channel_username)

        async for message in client.iter_messages(channel):

            if not message or not message.date:
                continue

            msg_date = message.date

            # filter time range
            if msg_date < start_date:
                continue
            if msg_date > end_date:
                continue

            messages_data.append({
                "id": message.id,
                "date": msg_date,
                "text": message.text or "",
                "views": message.views,
                "forwards": message.forwards,
                "reactions": (
                    sum(r.count for r in message.reactions.results)
                    if message.reactions else None
                )
            })

    # ---------------------------
    # ONLY HERE build dataframe
    # ---------------------------
    df = pd.DataFrame(messages_data)

    # safety check (prevents your error completely)
    if df.empty:
        print("⚠️ No messages found for this date range")
        return df

    # ensure column exists BEFORE using it
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

    return df


def run_telegram_loader(api_id, api_hash, channel, start_date, end_date):
    return asyncio.run(
        fetch_telegram_messages(
            api_id,
            api_hash,
            channel,
            start_date,
            end_date
        )
    )