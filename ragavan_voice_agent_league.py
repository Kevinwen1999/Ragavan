import asyncio
import audioop
import contextlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import wave
from collections import defaultdict
from datetime import datetime

import discord
import edge_tts
from discord.ext import commands
from dotenv import load_dotenv
import numpy as np
import requests
import speech_recognition as sr
from urllib3.exceptions import InsecureRequestWarning
import whisper

# -------------------- setup --------------------
load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Reduce sink/voice spam if any
logging.getLogger("discord").setLevel(logging.INFO)

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown")
    clean = clean.strip("._")
    return clean[:48] or "unknown"


# -------------------- Pycord live sink --------------------
class LivePCMSink(discord.sinks.Sink):
    """
    A streaming sink that receives PCM chunks and forwards them to a callback.

    In Pycord, Sink.write(self, data, user) receives:
      - data: decoded PCM bytes (48kHz stereo 16-bit) in small chunks
      - user: user_id (int) or str depending on internal mapping

    We use RMS-only VAD like your old code (no RTP extensions available).
    """

    def __init__(self, on_pcm_cb, *, filters=None):
        super().__init__(filters=filters)
        self.on_pcm_cb = on_pcm_cb

    @discord.sinks.Filters.container
    def write(self, data, user):
        try:
            user_id = int(user)
        except Exception:
            return

        # Forward to your Cog logic
        self.on_pcm_cb(user_id, data)

    def format_audio(self, audio):
        # Required by Pycord Sink.cleanup(), but this streaming sink does not format files.
        return


# -------------------- Cog --------------------
class Testing(commands.Cog):
    @staticmethod
    def _parse_llm_keywords(raw_value):
        if raw_value is None:
            return []

        if isinstance(raw_value, (list, tuple, set)):
            candidates = raw_value
        else:
            candidates = re.split(r"[,|\n]+", str(raw_value))

        keywords = []
        seen = set()
        for item in candidates:
            keyword = str(item).strip()
            if not keyword:
                continue
            lowered = keyword.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(keyword)
        return keywords

    @staticmethod
    def _format_llm_keywords(keywords) -> str:
        normalized = Testing._parse_llm_keywords(keywords)
        return ", ".join(normalized) if normalized else "(none)"

    def __init__(self, bot: commands.Bot):
        self.bot = bot

        # "stats" similar to your original, but adjusted for Pycord
        self.packet_counts = defaultdict(int)   # guild_id -> pcm chunks received
        self.decode_errors = defaultdict(int)   # kept for display; should be 0 in Pycord path
        self.out_of_order = defaultdict(int)    # not applicable in Pycord; kept for display

        self.user_states = defaultdict(dict)    # guild_id -> {user_id: state}
        self.stt_channels = {}                  # guild_id -> channel_id for transcript output

        self.stt_enabled = defaultdict(lambda: True)
        self.stt_language = defaultdict(lambda: "zh-CN")
        backend_default = os.getenv("STT_BACKEND", "whisper").strip().lower()
        if backend_default in {"sr", "speech_recognition"}:
            backend_default = "speechrecognition"
        if backend_default not in {"speechrecognition", "whisper"}:
            backend_default = "whisper"
        self.stt_backend = defaultdict(lambda: backend_default)
        self.voice_threshold = defaultdict(lambda: 12)   # kept, but now treated as "RMS threshold helper"
        self.rms_threshold = defaultdict(lambda: 260)
        self.rms_margin = defaultdict(lambda: 180)
        self.rms_release_ratio = defaultdict(lambda: 0.58)
        self.silence_timeout = defaultdict(lambda: 0.9)
        self.max_utterance_seconds = defaultdict(lambda: 15.0)
        self.min_utterance_seconds = defaultdict(lambda: 1.0)
        self.min_utterance_bytes = defaultdict(lambda: 76800)  # ~0.4s at 48k stereo 16-bit
        self.pre_roll_seconds = float(os.getenv("STT_PRE_ROLL_SECONDS", "0.25"))

        self.stt_dump_dir = os.getenv("STT_DUMP_DIR", "stt_dumps")
        dump_default = os.getenv("STT_DEBUG_DUMP", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.stt_debug_dump = defaultdict(lambda: dump_default)
        self.dump_sequence = defaultdict(int)

        self.whisper_model_name = os.getenv("WHISPER_MODEL", "large-v3")
        self.whisper_task = os.getenv("WHISPER_TASK", "transcribe")
        self._whisper_model = None
        self._whisper_lock = threading.Lock()

        self.stt_engine = os.getenv("STT_ENGINE", "google").strip().lower()
        if self.stt_engine not in {"google", "sphinx"}:
            self.stt_engine = "google"

        # LLM + TTS trigger settings
        llm_enabled_default = os.getenv("STT_LLM_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        llm_keyword_default = self._parse_llm_keywords(os.getenv("STT_LLM_KEYWORD", "alex, 石头城大胃王，石头城大卫王"))
        if not llm_keyword_default:
            llm_keyword_default = ["alex"]
        llm_timeout_default = float(os.getenv("STT_LLM_TIMEOUT_SECONDS", "15"))
        self.llm_enabled = defaultdict(lambda: llm_enabled_default)
        self.llm_keyword = defaultdict(lambda: list(llm_keyword_default))
        self.llm_timeout_seconds = defaultdict(lambda: llm_timeout_default)
        self.llm_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        self.llm_model = os.getenv("LLM_MODEL", "").strip()
        self.llm_system_prompt = os.getenv("LLM_SYSTEM_PROMPT", "").strip()
        self.llm_max_reply_chars = int(os.getenv("STT_LLM_MAX_REPLY_CHARS", "500"))
        self.tts_voice = os.getenv("TTS_VOICE", "zh-CN-XiaoyiNeural")
        self.tts_rate = os.getenv("TTS_RATE", "+0%")
        passive_enabled_default = os.getenv("STT_PASSIVE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        passive_interval_default = float(os.getenv("STT_PASSIVE_INTERVAL_SECONDS", "60"))
        self.passive_enabled = defaultdict(lambda: passive_enabled_default)
        self.passive_interval_seconds = defaultdict(lambda: passive_interval_default)
        self.passive_timeout_seconds = float(os.getenv("STT_PASSIVE_TIMEOUT_SECONDS", "15"))
        self.passive_instruction = os.getenv(
            "STT_PASSIVE_INSTRUCTION",
            "Based on the last minute of conversation, provide one short helpful comment.",
        ).strip()
        league_enabled_default = os.getenv("STT_LEAGUE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.league_enabled = defaultdict(lambda: league_enabled_default)
        self.league_poll_interval_seconds = float(os.getenv("STT_LEAGUE_POLL_INTERVAL_SECONDS", "30"))
        self.league_timeout_seconds = float(os.getenv("STT_LEAGUE_TIMEOUT_SECONDS", "5"))
        self.league_endpoint_base = os.getenv(
            "LEAGUE_LIVECLIENT_BASE_URL",
            "https://127.0.0.1:2999/liveclientdata",
        ).rstrip("/")
        self.league_verify_tls = os.getenv("LEAGUE_LIVECLIENT_VERIFY_TLS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.league_instruction = os.getenv(
            "STT_LEAGUE_INSTRUCTION",
            "Give one concise comment about the current game state for the active player.",
        ).strip()
        if not self.league_verify_tls:
            requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

        self._state_lock = threading.Lock()
        self._watchdog_task = None
        self._tts_locks = {}
        self._conversation_history = defaultdict(list)  # guild_id -> [{role, content, source}]
        self._passive_tasks = {}  # guild_id -> asyncio.Task
        self._league_tasks = {}  # guild_id -> asyncio.Task

        # keep track of voice connections per guild
        self._connections = {}  # guild_id -> VoiceClient

    async def _emit_stt_message(self, guild_id: int, message: str) -> None:
        # Console-only output as your original behavior
        print(message)

    async def _ensure_watchdog(self):
        task = self._watchdog_task
        if task is not None and not task.done():
            return

        self._watchdog_task = asyncio.create_task(self._speech_watchdog(), name="speech-watchdog")
        print("[STT] speech watchdog started")

    def cog_unload(self):
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        for guild_id, task in list(self._passive_tasks.items()):
            task.cancel()
            self._passive_tasks.pop(guild_id, None)
        for guild_id, task in list(self._league_tasks.items()):
            task.cancel()
            self._league_tasks.pop(guild_id, None)

    def _reset_guild_stats(self, guild_id: int) -> None:
        with self._state_lock:
            self.packet_counts[guild_id] = 0
            self.decode_errors[guild_id] = 0
            self.out_of_order[guild_id] = 0
            self.user_states[guild_id].clear()
        self._conversation_history[guild_id].clear()

    def _resolve_user_name(self, guild_id: int, user_id: int) -> str:
        guild = self.bot.get_guild(guild_id)
        if guild is not None:
            member = guild.get_member(user_id)
            if member is not None:
                return member.display_name or member.name or str(user_id)

        user = self.bot.get_user(user_id)
        if user is not None:
            return getattr(user, "display_name", None) or user.name or str(user_id)

        return str(user_id)

    def _get_tts_lock(self, guild_id: int) -> asyncio.Lock:
        lock = self._tts_locks.get(guild_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tts_locks[guild_id] = lock
        return lock

    def _append_conversation(self, guild_id: int, role: str, content: str, source: str):
        content = (content or "").strip()
        if not content:
            return
        self._conversation_history[guild_id].append(
            {"role": role, "content": content, "source": source}
        )

    def _drain_conversation(self, guild_id: int):
        items = self._conversation_history[guild_id]
        if not items:
            return []
        snapshot = list(items)
        items.clear()
        return snapshot

    async def _ensure_passive_task(self, guild_id: int):
        task = self._passive_tasks.get(guild_id)
        if task is not None and not task.done():
            return
        if not self.passive_enabled[guild_id]:
            return
        self._passive_tasks[guild_id] = asyncio.create_task(
            self._passive_comment_loop(guild_id), name=f"passive-comment-{guild_id}"
        )
        await self._emit_stt_message(guild_id, "[PASSIVE] minute comment loop started.")

    def _cancel_passive_task(self, guild_id: int):
        task = self._passive_tasks.pop(guild_id, None)
        if task is not None:
            task.cancel()

    async def _ensure_league_task(self, guild_id: int):
        task = self._league_tasks.get(guild_id)
        if task is not None and not task.done():
            return
        if not self.league_enabled[guild_id]:
            return
        self._league_tasks[guild_id] = asyncio.create_task(
            self._league_comment_loop(guild_id), name=f"league-comment-{guild_id}"
        )
        await self._emit_stt_message(guild_id, "[LEAGUE] 30s liveclient polling started.")

    def _cancel_league_task(self, guild_id: int):
        task = self._league_tasks.pop(guild_id, None)
        if task is not None:
            task.cancel()

    @staticmethod
    def _normalize_riot_id(player: dict) -> str:
        riot_id = str(player.get("riotId") or "").strip()
        if riot_id:
            return riot_id
        game_name = str(player.get("riotIdGameName") or "").strip()
        tag = str(player.get("riotIdTagLine") or "").strip()
        if game_name and tag:
            return f"{game_name}#{tag}"
        return ""

    @staticmethod
    def _normalize_scores(scores: dict) -> dict:
        scores = scores or {}
        return {
            "kills": scores.get("kills", 0),
            "deaths": scores.get("deaths", 0),
            "assists": scores.get("assists", 0),
            "creepScore": scores.get("creepScore", 0),
            "wardScore": scores.get("wardScore", 0),
        }

    @staticmethod
    def _normalize_team(team_value) -> str:
        team = str(team_value or "").strip().upper()
        if not team:
            return "UNKNOWN"
        if team in {"ORDER", "CHAOS"}:
            return team
        return team

    @staticmethod
    def _event_time_clock(value) -> str:
        try:
            total = max(0, int(float(value)))
        except Exception:
            return "00:00"
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _fetch_league_snapshot_sync(self):
        timeout_s = max(1.0, float(self.league_timeout_seconds))
        verify_tls = self.league_verify_tls

        active_url = f"{self.league_endpoint_base}/activeplayer"
        playerlist_url = f"{self.league_endpoint_base}/playerlist"
        eventdata_url = f"{self.league_endpoint_base}/eventdata"

        try:
            active_response = requests.get(active_url, timeout=timeout_s, verify=verify_tls)
            active_response.raise_for_status()
            active_payload = active_response.json()
        except (requests.RequestException, ValueError):
            return None

        active_riot_id = self._normalize_riot_id(active_payload if isinstance(active_payload, dict) else {})
        if not active_riot_id:
            return None
        active_game_name = str((active_payload or {}).get("riotIdGameName") or "").strip()
        active_summoner_name = str((active_payload or {}).get("summonerName") or "").strip()

        try:
            playerlist_response = requests.get(playerlist_url, timeout=timeout_s, verify=verify_tls)
            playerlist_response.raise_for_status()
            playerlist_payload = playerlist_response.json()
        except (requests.RequestException, ValueError):
            return None

        if not isinstance(playerlist_payload, list):
            return None

        events = []
        try:
            eventdata_response = requests.get(eventdata_url, timeout=timeout_s, verify=verify_tls)
            eventdata_response.raise_for_status()
            eventdata_payload = eventdata_response.json()
            raw_events = eventdata_payload.get("Events") if isinstance(eventdata_payload, dict) else None
            if isinstance(raw_events, list):
                events = [x for x in raw_events if isinstance(x, dict)]
        except (requests.RequestException, ValueError):
            events = []

        players = []
        active_player = None
        for entry in playerlist_payload:
            if not isinstance(entry, dict):
                continue

            riot_id = self._normalize_riot_id(entry)
            scores = self._normalize_scores(entry.get("scores") if isinstance(entry.get("scores"), dict) else {})
            player_summary = {
                "riotId": riot_id,
                "championName": str(entry.get("championName") or "Unknown"),
                "team": self._normalize_team(entry.get("team")),
                "scores": scores,
                "details": dict(entry),
            }
            players.append(player_summary)

            if riot_id == active_riot_id:
                active_player = player_summary

        if active_player is None:
            active_candidates = {active_riot_id, active_game_name, active_summoner_name}
            active_candidates = {x.strip().casefold() for x in active_candidates if str(x or "").strip()}
            for player in players:
                details = player.get("details") or {}
                player_candidates = {
                    str(player.get("riotId") or "").strip(),
                    str(details.get("riotIdGameName") or "").strip(),
                    str(details.get("summonerName") or "").strip(),
                }
                player_candidates = {x.casefold() for x in player_candidates if x}
                if active_candidates.intersection(player_candidates):
                    active_player = player
                    break

        if not players:
            return None

        active_team = self._normalize_team((active_player or {}).get("team"))
        for player in players:
            player_team = self._normalize_team(player.get("team"))
            player["team"] = player_team
            is_active_player = player is active_player or player.get("riotId") == active_riot_id
            if is_active_player:
                relation = "self"
            elif active_team == "UNKNOWN" or player_team == "UNKNOWN":
                relation = "unknown"
            elif player_team == active_team:
                relation = "ally"
            else:
                relation = "enemy"

            player["is_active_player"] = bool(is_active_player)
            player["relation"] = relation

            details = player.get("details")
            if isinstance(details, dict):
                details["team"] = player_team
                details["relation"] = relation
                details["isActivePlayer"] = bool(is_active_player)

        return {
            "active_riot_id": active_riot_id,
            "active_game_name": active_game_name,
            "active_summoner_name": active_summoner_name,
            "active_team": active_team,
            "active_player": active_player,
            "players": players,
            "events": events,
        }

    def _build_league_comment_query(self, snapshot: dict) -> str:
        active_riot_id = str(snapshot.get("active_riot_id") or "")
        active_game_name = str(snapshot.get("active_game_name") or "").strip()
        active_summoner_name = str(snapshot.get("active_summoner_name") or "").strip()
        active_team = self._normalize_team(snapshot.get("active_team"))
        active_player = snapshot.get("active_player") or {}
        active_scores = active_player.get("scores") or {}
        players = snapshot.get("players") or []
        events = snapshot.get("events") or []
        if not active_riot_id or not players:
            return ""

        lines = [
            "League of Legends live client update.",
            f"Active player riotId: {active_riot_id}",
            f"Active player gameName: {active_game_name or '(unknown)'}",
            f"Active player team: {active_team}",
            (
                "Active player score: "
                f"K/D/A {active_scores.get('kills', 0)}/{active_scores.get('deaths', 0)}/{active_scores.get('assists', 0)}."
            ),
            "All players score summary with relation to active player:",
        ]

        allies = []
        enemies = []
        unknown_side = []
        for player in players:
            scores = player.get("scores") or {}
            relation = str(player.get("relation") or "unknown")
            team = self._normalize_team(player.get("team"))
            marker = " [ACTIVE]" if player.get("is_active_player") else ""
            line = (
                f"- {relation.upper()} | {team} {player.get('championName', 'Unknown')} "
                f"{player.get('riotId', 'unknown')}{marker}: "
                f"K/D/A {scores.get('kills', 0)}/{scores.get('deaths', 0)}/{scores.get('assists', 0)}"
            )
            lines.append(line)
            if relation in {"self", "ally"}:
                allies.append(line[2:])
            elif relation == "enemy":
                enemies.append(line[2:])
            else:
                unknown_side.append(line[2:])

        lines.append("Team split:")
        lines.append(f"Allies+Self count: {len(allies)}")
        for line in allies:
            lines.append(f"- {line}")
        lines.append(f"Enemies count: {len(enemies)}")
        for line in enemies:
            lines.append(f"- {line}")
        if unknown_side:
            lines.append(f"Unknown-side count: {len(unknown_side)}")
            for line in unknown_side:
                lines.append(f"- {line}")

        detailed_playerlist = []
        for player in players:
            details = player.get("details")
            if not isinstance(details, dict):
                continue
            cleaned = dict(details)
            raw_scores = cleaned.get("scores")
            if isinstance(raw_scores, dict):
                cleaned_scores = dict(raw_scores)
                cleaned_scores.pop("creepScore", None)
                cleaned_scores.pop("wardScore", None)
                cleaned["scores"] = cleaned_scores
            detailed_playerlist.append(cleaned)
        if detailed_playerlist:
            lines.append("Detailed playerlist JSON (use this as primary context):")
            lines.append(json.dumps(detailed_playerlist, ensure_ascii=False, separators=(",", ":")))

        active_identifiers = {active_riot_id, active_game_name, active_summoner_name}
        active_identifiers.update(
            {
                str((active_player.get("details") or {}).get("riotIdGameName") or "").strip(),
                str((active_player.get("details") or {}).get("summonerName") or "").strip(),
            }
        )
        active_identifiers = {x for x in active_identifiers if x}

        lines.append(f"Total events captured: {len(events)}")
        if events:
            lines.append("Recent event timeline:")
            for event in events[-12:]:
                event_name = str(event.get("EventName") or "Unknown")
                event_time = self._event_time_clock(event.get("EventTime"))
                event_id = event.get("EventID", "?")

                actors = [
                    str(event.get("KillerName") or "").strip(),
                    str(event.get("VictimName") or "").strip(),
                    str(event.get("Recipient") or "").strip(),
                ]
                assisters = event.get("Assisters")
                if isinstance(assisters, list):
                    actors.extend(str(x).strip() for x in assisters if str(x).strip())
                is_active_related = any(actor in active_identifiers for actor in actors if actor)
                marker = " [ACTIVE_EVENT]" if is_active_related else ""

                if event_name == "ChampionKill":
                    killer = str(event.get("KillerName") or "Unknown")
                    victim = str(event.get("VictimName") or "Unknown")
                    assists_text = ""
                    if isinstance(assisters, list) and assisters:
                        assists_text = f", assisters={','.join(str(x) for x in assisters)}"
                    lines.append(
                        f"- t={event_time} id={event_id} ChampionKill: {killer} -> {victim}{assists_text}{marker}"
                    )
                else:
                    lines.append(
                        f"- t={event_time} id={event_id} {event_name}: "
                        f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}{marker}"
                    )

            lines.append("Detailed eventdata JSON (all events):")
            lines.append(json.dumps(events, ensure_ascii=False, separators=(",", ":")))

        lines.append(self.league_instruction or "Give detailed game-state comment.")
        lines.append("Keep it concise and practical for the active player. Focus on in-game decision-making insights.")
        return "\n".join(lines)

    # -------- PCM intake (replacement for voice_recv _on_voice_packet) --------
    def _on_pcm_chunk(self, guild_id: int, user_id: int, pcm: bytes):
        now = time.monotonic()
        if not pcm:
            return

        # PCM is 16-bit stereo in Pycord WaveSink/Sink path
        rms = audioop.rms(pcm, 2)

        with self._state_lock:
            self.packet_counts[guild_id] += 1

            state = self.user_states[guild_id].get(user_id)
            if state is None:
                state = {
                    "name": self._resolve_user_name(guild_id, user_id),
                    "speaking": False,
                    "last_voice_ts": now,
                    "pcm_buffer": bytearray(),
                    "pre_buffer": bytearray(),
                    "utterance_packets": 0,
                    "utterance_start_ts": None,
                    "last_rms": 0,
                    "noise_floor_rms": max(20.0, float(self.rms_threshold[guild_id]) * 0.45),
                    "silent_chunks": 0,
                    "adaptive_start_threshold": self.rms_threshold[guild_id],
                    "adaptive_stop_threshold": max(35, int(self.rms_threshold[guild_id] * 0.58)),
                }
                self.user_states[guild_id][user_id] = state
            else:
                # Keep name fresh from cache once member/user info is available.
                state["name"] = self._resolve_user_name(guild_id, user_id)

            state["last_rms"] = rms

            noise_floor = float(state.get("noise_floor_rms", 40.0))
            if not state["speaking"]:
                # Track room/noise baseline only while idle.
                noise_floor = (noise_floor * 0.97) + (float(rms) * 0.03)
                state["noise_floor_rms"] = noise_floor

            base_start = self.rms_threshold[guild_id]
            margin = max(40, self.rms_margin[guild_id])
            adaptive_start = max(base_start, int(noise_floor + margin))

            adaptive_stop = max(
                35,
                min(
                    adaptive_start - 10,
                    max(
                        int(adaptive_start * self.rms_release_ratio[guild_id]),
                        int(noise_floor + margin * 0.40),
                    ),
                ),
            )
            state["adaptive_start_threshold"] = adaptive_start
            state["adaptive_stop_threshold"] = adaptive_stop

            rms_start = rms >= adaptive_start
            rms_continue = rms >= adaptive_stop

            if rms_start:
                if not state["speaking"]:
                    state["speaking"] = True
                    state["pcm_buffer"].clear()
                    state["utterance_packets"] = 0
                    state["utterance_start_ts"] = now
                    state["silent_chunks"] = 0
                    if state["pre_buffer"]:
                        state["pcm_buffer"].extend(state["pre_buffer"])
                        state["pre_buffer"].clear()
                    print(
                        f"[guild={guild_id}] START user={state['name']} rms={rms} "
                        f"start>={adaptive_start} stop<{adaptive_stop}"
                    )

                state["last_voice_ts"] = now
                state["silent_chunks"] = 0
                state["pcm_buffer"].extend(pcm)
                state["utterance_packets"] += 1

            elif state["speaking"]:
                # tail keep
                state["pcm_buffer"].extend(pcm)
                state["utterance_packets"] += 1
                if rms_continue:
                    state["last_voice_ts"] = now
                    state["silent_chunks"] = 0
                else:
                    state["silent_chunks"] = int(state.get("silent_chunks", 0)) + 1

            else:
                # pre-roll buffer
                pre_roll_gate = max(40, int(noise_floor * 0.85))
                if rms >= pre_roll_gate:
                    pre = state["pre_buffer"]
                    pre.extend(pcm)
                    max_pre_bytes = int(192000 * self.pre_roll_seconds)  # 48k*2ch*2bytes = 192000 bytes/sec
                    if max_pre_bytes > 0 and len(pre) > max_pre_bytes:
                        del pre[:-max_pre_bytes]

    # -------- watchdog to decide when utterance ended --------
    async def _speech_watchdog(self):
        while True:
            jobs = []
            debug_msgs = []
            now = time.monotonic()

            with self._state_lock:
                for guild_id, states in list(self.user_states.items()):
                    timeout = self.silence_timeout[guild_id]
                    max_utt = self.max_utterance_seconds[guild_id]
                    min_seconds = self.min_utterance_seconds[guild_id]
                    min_bytes = self.min_utterance_bytes[guild_id]

                    for user_id, state in list(states.items()):
                        if not state["speaking"]:
                            continue

                        start_ts = state.get("utterance_start_ts") or state["last_voice_ts"]
                        silence_elapsed = now - state["last_voice_ts"]
                        utterance_elapsed = max(0.0, now - start_ts)
                        stop_reason = None
                        if silence_elapsed >= timeout:
                            stop_reason = "silence"
                        elif utterance_elapsed >= max_utt:
                            stop_reason = "max_utterance"

                        if stop_reason is None:
                            continue

                        state["speaking"] = False
                        pcm = bytes(state["pcm_buffer"])
                        packet_count = state.get("utterance_packets", 0)
                        state["pcm_buffer"].clear()
                        state["utterance_packets"] = 0
                        state["utterance_start_ts"] = None

                        duration_s = len(pcm) / 192000 if pcm else 0.0  # 48k stereo 16-bit
                        print(
                            f"[guild={guild_id}] STOP user={state['name']} duration={duration_s:.2f}s "
                            f"bytes={len(pcm)} chunks={packet_count} reason={stop_reason} "
                            f"silence={silence_elapsed:.2f}s"
                        )

                        if self.stt_enabled[guild_id]:
                            if duration_s > min_seconds and len(pcm) >= min_bytes:
                                dump_enabled = self.stt_debug_dump[guild_id]
                                dump_id = None
                                if dump_enabled:
                                    self.dump_sequence[guild_id] += 1
                                    dump_id = self.dump_sequence[guild_id]

                                meta = {
                                    "dump_enabled": dump_enabled,
                                    "dump_id": dump_id,
                                    "guild_id": guild_id,
                                    "user_id": user_id,
                                    "user_name": state["name"],
                                    "duration_s": round(duration_s, 4),
                                    "bytes": len(pcm),
                                    "chunks": packet_count,
                                    "utterance_start_monotonic": start_ts,
                                    "utterance_stop_monotonic": now,
                                    "stop_reason": stop_reason,
                                    "silence_elapsed_s": round(silence_elapsed, 4),
                                    "adaptive_start_threshold": state.get("adaptive_start_threshold"),
                                    "adaptive_stop_threshold": state.get("adaptive_stop_threshold"),
                                    "noise_floor_rms": round(float(state.get("noise_floor_rms", 0.0)), 2),
                                    "stt_language": self.stt_language[guild_id],
                                    "stt_backend": self.stt_backend[guild_id],
                                    "stt_engine": self.stt_engine,
                                    "whisper_model": self.whisper_model_name,
                                    "whisper_task": self.whisper_task,
                                    "pcm_layout": "stereo_48k_s16le",
                                }
                                jobs.append((guild_id, user_id, state["name"], pcm, duration_s, meta))
                            else:
                                debug_msgs.append(
                                    (
                                        guild_id,
                                        f"[STT skip {duration_s:.1f}s] {state['name']}: "
                                        f"too short (need > {min_seconds:.1f}s).",
                                    )
                                )

            for job in jobs:
                asyncio.create_task(self._transcribe_utterance(*job))
            for guild_id, message in debug_msgs:
                asyncio.create_task(self._emit_stt_message(guild_id, message))

            await asyncio.sleep(0.2)

    # -------- speech recognition helpers --------
    @staticmethod
    def _normalize_stt_language(language: str) -> str:
        if not language:
            return "zh-CN"
        lang = language.strip()
        if not lang or lang.lower() in {"auto", "detect", "none"}:
            return "zh-CN"
        return lang

    @staticmethod
    def _pcm_to_wav_16k_mono_bytes(pcm_48k_stereo: bytes) -> bytes:
        if not pcm_48k_stereo:
            return b""

        # mono mix
        mono_48k = audioop.tomono(pcm_48k_stereo, 2, 0.5, 0.5)

        # normalize
        peak = audioop.max(mono_48k, 2)
        if peak > 0:
            gain = min(4.0, 24000.0 / peak)
            mono_48k = audioop.mul(mono_48k, 2, gain)

        mono_16k, _ = audioop.ratecv(mono_48k, 2, 1, 48000, 16000, None)
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(mono_16k)
            return buf.getvalue()

    def _recognize_wav_file_sync(self, wav_path: str, language: str) -> str:
        lang = self._normalize_stt_language(language)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        if self.stt_engine == "sphinx":
            text = recognizer.recognize_sphinx(audio, language=lang)
        else:
            text = recognizer.recognize_google(audio, language=lang)
        return (text or "").strip()

    @staticmethod
    def _normalize_whisper_language(language: str):
        if not language:
            return None
        lang = language.strip().lower()
        if lang in {"auto", "detect", "none"}:
            return None
        if "-" in lang:
            lang = lang.split("-", 1)[0]
        return lang or None

    @staticmethod
    def _to_whisper_audio(pcm_48k_stereo: bytes) -> np.ndarray:
        # mono mix
        mono_48k = audioop.tomono(pcm_48k_stereo, 2, 0.5, 0.5)

        # normalize
        peak = audioop.max(mono_48k, 2)
        if peak > 0:
            gain = min(6.0, 26000.0 / peak)
            mono_48k = audioop.mul(mono_48k, 2, gain)

        mono_16k, _ = audioop.ratecv(mono_48k, 2, 1, 48000, 16000, None)
        return np.frombuffer(mono_16k, dtype=np.int16).astype(np.float32) / 32768.0

    def _get_whisper_model(self):
        with self._whisper_lock:
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def _recognize_whisper_from_pcm_sync(self, pcm_48k_stereo: bytes, language: str) -> str:
        audio = self._to_whisper_audio(pcm_48k_stereo)
        if audio.size == 0:
            return ""

        whisper_lang = self._normalize_whisper_language(language)
        kwargs = {"task": self.whisper_task}
        if whisper_lang:
            kwargs["language"] = whisper_lang

        result = self._get_whisper_model().transcribe(audio, fp16=False, **kwargs)
        return (result.get("text") or "").strip()

    def _recognize_whisper_from_wav_sync(self, wav_path: str, language: str) -> str:
        whisper_lang = self._normalize_whisper_language(language)
        kwargs = {"task": self.whisper_task}
        if whisper_lang:
            kwargs["language"] = whisper_lang
        result = self._get_whisper_model().transcribe(wav_path, fp16=False, **kwargs)
        return (result.get("text") or "").strip()

    def _recognize_sync(self, pcm_48k_stereo: bytes, language: str, backend: str) -> str:
        if backend == "whisper":
            return self._recognize_whisper_from_pcm_sync(pcm_48k_stereo, language)

        wav_bytes = self._pcm_to_wav_16k_mono_bytes(pcm_48k_stereo)
        if not wav_bytes:
            return ""

        lang = self._normalize_stt_language(language)
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)

        try:
            if self.stt_engine == "sphinx":
                text = recognizer.recognize_sphinx(audio, language=lang)
            else:
                text = recognizer.recognize_google(audio, language=lang)
            return (text or "").strip()
        except sr.UnknownValueError:
            return ""

    def _dump_utterance_sync(self, guild_id: int, user_id: int, user_name: str, pcm: bytes, meta: dict):
        base_dir = os.path.abspath(self.stt_dump_dir)
        guild_dir = os.path.join(base_dir, f"guild_{guild_id}")
        os.makedirs(guild_dir, exist_ok=True)

        t = time.time()
        sec = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
        ms = int((t - int(t)) * 1000)
        dump_id = int(meta.get("dump_id") or 0)
        stem = f"{sec}_{ms:03d}_u{user_id}_{safe_label(user_name)}_{dump_id:05d}"

        wav_path = os.path.join(guild_dir, f"{stem}.wav")
        json_path = os.path.join(guild_dir, f"{stem}.json")

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

        return wav_path, json_path

    def _extract_llm_query(self, guild_id: int, text: str):
        keywords = self._parse_llm_keywords(self.llm_keyword[guild_id])
        if not keywords:
            return None

        lower_text = text.lower()
        matched_keyword = None
        matched_idx = -1
        for keyword in keywords:
            idx = lower_text.find(keyword.lower())
            if idx < 0:
                continue
            if matched_keyword is None or idx < matched_idx or (idx == matched_idx and len(keyword) > len(matched_keyword)):
                matched_keyword = keyword
                matched_idx = idx

        if matched_keyword is None:
            return None

        # Keep words after keyword as the user query.
        query = text[matched_idx + len(matched_keyword):].lstrip(" ,.:;!?-")
        if not query:
            query = text.strip()
        return query or None

    def _request_llm_sync(self, query: str, timeout_s: float):
        messages = []
        if self.llm_system_prompt:
            messages.append({"role": "system", "content": self.llm_system_prompt})
        messages.append({"role": "user", "content": query})
        return self._request_llm_messages_sync(messages, timeout_s)

    def _request_llm_messages_sync(self, messages, timeout_s: float):
        payload = {"messages": messages}
        if self.llm_model:
            payload["model"] = self.llm_model

        response = requests.post(
            self.llm_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        response.raise_for_status()
        result = response.json()

        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = ""
        return (text or "").strip()

    async def _speak_tts(self, guild_id: int, text: str):
        if not text.strip():
            return

        vc = self._connections.get(guild_id)
        if vc is None or not vc.is_connected():
            guild = self.bot.get_guild(guild_id)
            if guild is not None:
                vc = guild.voice_client
        if vc is None or not vc.is_connected():
            await self._emit_stt_message(guild_id, "[TTS skip] bot is not connected to voice.")
            return

        speak_text = text.strip()
        if len(speak_text) > self.llm_max_reply_chars:
            speak_text = speak_text[: self.llm_max_reply_chars].rstrip() + "..."

        lock = self._get_tts_lock(guild_id)
        async with lock:
            temp_path = None
            done = asyncio.Event()
            loop = self.bot.loop

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    temp_path = tmp.name

                tts = edge_tts.Communicate(speak_text, voice=self.tts_voice, rate=self.tts_rate)
                await tts.save(temp_path)

                def after_playback(err):
                    if err:
                        print(f"[TTS error] {err}")
                    if temp_path:
                        with contextlib.suppress(Exception):
                            os.remove(temp_path)
                    loop.call_soon_threadsafe(done.set)

                if vc.is_playing() or vc.is_paused():
                    vc.stop()
                    await asyncio.sleep(0.05)

                vc.play(discord.FFmpegPCMAudio(temp_path), after=after_playback)
                await done.wait()
            except Exception as exc:
                if temp_path:
                    with contextlib.suppress(Exception):
                        os.remove(temp_path)
                await self._emit_stt_message(guild_id, f"[TTS error] {exc}")

    async def _maybe_send_to_llm(self, guild_id: int, user_name: str, transcript: str):
        if not self.llm_enabled[guild_id]:
            return

        query = self._extract_llm_query(guild_id, transcript)
        if not query:
            return

        timeout_s = max(1.0, float(self.llm_timeout_seconds[guild_id]))
        await self._emit_stt_message(guild_id, f"[LLM trigger] {user_name}: {query}")

        response_text = ""
        try:
            response_text = await asyncio.wait_for(
                asyncio.to_thread(self._request_llm_sync, query, timeout_s),
                timeout=timeout_s + 0.75,
            )
        except asyncio.TimeoutError:
            await self._emit_stt_message(guild_id, f"[LLM timeout] no response within {timeout_s:.1f}s.")
            return
        except Exception as exc:
            await self._emit_stt_message(guild_id, f"[LLM error] {exc}")
            return

        if not response_text:
            await self._emit_stt_message(guild_id, "[LLM] empty response.")
            return

        await self._emit_stt_message(guild_id, f"[LLM response] {response_text}")
        self._append_conversation(guild_id, "assistant", response_text, source="llm_response")
        await self._speak_tts(guild_id, response_text)

    async def _passive_comment_loop(self, guild_id: int):
        try:
            while True:
                interval_s = max(15.0, float(self.passive_interval_seconds[guild_id]))
                await asyncio.sleep(interval_s)

                vc = self._connections.get(guild_id)
                if vc is None or not vc.is_connected():
                    continue
                if not self.passive_enabled[guild_id]:
                    continue

                snapshot = self._drain_conversation(guild_id)
                if not snapshot:
                    continue

                # Prevent feedback loops from the previous passive output.
                convo = [x for x in snapshot if x.get("source") != "passive_comment"]
                if not convo:
                    continue

                messages = []
                if self.llm_system_prompt:
                    messages.append({"role": "system", "content": self.llm_system_prompt})
                messages.extend({"role": x["role"], "content": x["content"]} for x in convo)
                messages.append({"role": "user", "content": self.passive_instruction})
                messages.append({"role": "user", "content": "Provide one concise comment now."})

                timeout_s = max(1.0, float(self.passive_timeout_seconds))
                comment = ""
                try:
                    comment = await asyncio.wait_for(
                        asyncio.to_thread(self._request_llm_messages_sync, messages, timeout_s),
                        timeout=timeout_s + 0.75,
                    )
                except asyncio.TimeoutError:
                    await self._emit_stt_message(guild_id, f"[PASSIVE timeout] exceeded {timeout_s:.1f}s.")
                    continue
                except Exception as exc:
                    await self._emit_stt_message(guild_id, f"[PASSIVE error] {exc}")
                    continue

                comment = (comment or "").strip()
                if not comment:
                    await self._emit_stt_message(guild_id, "[PASSIVE] empty comment.")
                    continue

                await self._emit_stt_message(guild_id, f"[PASSIVE comment] {comment}")
                self._append_conversation(guild_id, "assistant", comment, source="passive_comment")
                await self._speak_tts(guild_id, comment)
        except asyncio.CancelledError:
            return

    async def _league_comment_loop(self, guild_id: int):
        try:
            while True:
                vc = self._connections.get(guild_id)
                if vc is None or not vc.is_connected():
                    interval_s = max(30.0, float(self.league_poll_interval_seconds))
                    await asyncio.sleep(interval_s)
                    continue
                if not self.league_enabled[guild_id]:
                    interval_s = max(30.0, float(self.league_poll_interval_seconds))
                    await asyncio.sleep(interval_s)
                    continue

                snapshot = await asyncio.to_thread(self._fetch_league_snapshot_sync)
                if snapshot:
                    if not snapshot.get("active_player"):
                        await self._emit_stt_message(guild_id, "[LEAGUE] active player not found in playerlist.")
                    else:
                        query = self._build_league_comment_query(snapshot)
                        if query:
                            timeout_s = max(1.0, float(self.llm_timeout_seconds[guild_id]))
                            comment = ""
                            try:
                                comment = await asyncio.wait_for(
                                    asyncio.to_thread(self._request_llm_sync, query, timeout_s),
                                    timeout=timeout_s + 0.75,
                                )
                            except asyncio.TimeoutError:
                                await self._emit_stt_message(guild_id, f"[LEAGUE timeout] exceeded {timeout_s:.1f}s.")
                            except Exception as exc:
                                await self._emit_stt_message(guild_id, f"[LEAGUE error] {exc}")
                            else:
                                comment = (comment or "").strip()
                                if not comment:
                                    await self._emit_stt_message(guild_id, "[LEAGUE] empty comment.")
                                else:
                                    await self._emit_stt_message(guild_id, f"[LEAGUE comment] {comment}")
                                    self._append_conversation(guild_id, "assistant", comment, source="league_comment")
                                    await self._speak_tts(guild_id, comment)

                interval_s = max(30.0, float(self.league_poll_interval_seconds))
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    async def _transcribe_utterance(self, guild_id: int, user_id: int, user_name: str, pcm: bytes, duration_s: float, meta: dict):
        text = ""
        error = None
        backend = self.stt_backend[guild_id]
        try:
            text = await asyncio.to_thread(self._recognize_sync, pcm, self.stt_language[guild_id], backend)
        except Exception as exc:
            error = str(exc)

        if meta.get("dump_enabled"):
            dump_meta = dict(meta)
            dump_meta["transcript"] = text
            dump_meta["error"] = error
            dump_meta["saved_at_epoch"] = time.time()
            try:
                wav_path, json_path = await asyncio.to_thread(
                    self._dump_utterance_sync, guild_id, user_id, user_name, pcm, dump_meta
                )
                await self._emit_stt_message(guild_id, f"[STT dump] wav={wav_path} meta={json_path}")
            except Exception as exc:
                await self._emit_stt_message(guild_id, f"[STT dump error] failure: {exc}")

        if error is not None:
            await self._emit_stt_message(guild_id, f"[STT error] failure: {error}")
            return

        if not text.strip():
            await self._emit_stt_message(guild_id, f"[STT {duration_s:.1f}s] {user_name}: empty transcript.")
            return

        await self._emit_stt_message(guild_id, f"[STT {duration_s:.1f}s] {user_name}: {text}")
        self._append_conversation(guild_id, "user", f"{user_name}: {text}", source="stt")
        asyncio.create_task(self._maybe_send_to_llm(guild_id, user_name, text))

    # -------------------- commands --------------------
    @commands.command()
    async def test(self, ctx: commands.Context):
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.reply("Join a voice channel first.")
            return

        await self._ensure_watchdog()
        await self._ensure_passive_task(ctx.guild.id)

        guild_id = ctx.guild.id
        self._reset_guild_stats(guild_id)
        self.stt_channels[guild_id] = ctx.channel.id

        # connect / move
        vc = ctx.voice_client
        if not vc or not vc.is_connected():
            vc = await ctx.author.voice.channel.connect()
        elif vc.channel != ctx.author.voice.channel:
            await vc.move_to(ctx.author.voice.channel)

        # stop any existing recording
        try:
            if getattr(vc, "recording", False):
                vc.stop_recording()
        except Exception:
            pass

        # Start streaming sink
        sink = LivePCMSink(lambda user_id, pcm: self._on_pcm_chunk(guild_id, user_id, pcm))
        self._connections[guild_id] = vc
        await self._ensure_league_task(guild_id)

        async def finished_cb(s: discord.sinks.Sink, channel, *args):
            # We don't use the formatted files here; live streaming is handled already.
            return

        vc.start_recording(sink, finished_cb, ctx.channel)
        await ctx.reply(
            f"Listening in `{vc.channel}`. STT={'on' if self.stt_enabled[guild_id] else 'off'} "
            f"lang={self.stt_language[guild_id]} backend={self.stt_backend[guild_id]} "
            f"engine={self.stt_engine} whisper_model={self.whisper_model_name} "
            f"rms_threshold={self.rms_threshold[guild_id]}"
        )

    @commands.command()
    async def stop(self, ctx: commands.Context):
        vc = ctx.voice_client
        if not vc or not vc.is_connected():
            await ctx.reply("Not connected.")
            return

        try:
            if getattr(vc, "recording", False):
                vc.stop_recording()
        except Exception:
            pass

        self._connections.pop(ctx.guild.id, None)
        self._cancel_passive_task(ctx.guild.id)
        self._cancel_league_task(ctx.guild.id)
        self._conversation_history[ctx.guild.id].clear()
        await vc.disconnect()
        await ctx.reply("Stopped listening and disconnected.")

    @commands.command()
    async def stt(self, ctx: commands.Context, mode: str = "status"):
        guild_id = ctx.guild.id
        mode = mode.lower()

        if mode in {"on", "enable", "enabled"}:
            self.stt_enabled[guild_id] = True
            await ctx.reply(
                f"STT enabled. Language: {self.stt_language[guild_id]}, backend: {self.stt_backend[guild_id]}"
            )
        elif mode in {"off", "disable", "disabled"}:
            self.stt_enabled[guild_id] = False
            await ctx.reply("STT disabled.")
        else:
            await ctx.reply(
                f"STT is {'on' if self.stt_enabled[guild_id] else 'off'}, "
                f"language={self.stt_language[guild_id]}, backend={self.stt_backend[guild_id]}"
            )

    @commands.command()
    async def sttlang(self, ctx: commands.Context, language: str):
        guild_id = ctx.guild.id
        self.stt_language[guild_id] = language
        await ctx.reply(f"STT language set to `{language}`.")

    @commands.command()
    async def sttbackend(self, ctx: commands.Context, backend: str = "status"):
        guild_id = ctx.guild.id
        mode = backend.strip().lower()
        if mode in {"status", "show"}:
            await ctx.reply(
                f"STT backend is `{self.stt_backend[guild_id]}` "
                f"(speechrecognition engine=`{self.stt_engine}`, whisper model=`{self.whisper_model_name}`)."
            )
            return

        if mode in {"sr", "speech_recognition"}:
            mode = "speechrecognition"
        if mode not in {"speechrecognition", "whisper"}:
            await ctx.reply("Invalid backend. Use `speechrecognition` or `whisper`.")
            return

        self.stt_backend[guild_id] = mode
        await ctx.reply(f"STT backend set to `{mode}`.")

    @commands.command()
    async def sttdump(self, ctx: commands.Context, mode: str = "status"):
        guild_id = ctx.guild.id
        mode = mode.lower()
        dump_dir = os.path.abspath(self.stt_dump_dir)

        if mode in {"on", "enable", "enabled"}:
            self.stt_debug_dump[guild_id] = True
            await ctx.reply(f"STT debug dump enabled. Directory: `{dump_dir}`")
        elif mode in {"off", "disable", "disabled"}:
            self.stt_debug_dump[guild_id] = False
            await ctx.reply("STT debug dump disabled.")
        else:
            await ctx.reply(f"STT debug dump is {'on' if self.stt_debug_dump[guild_id] else 'off'}, directory=`{dump_dir}`")

    @commands.command()
    async def llm(self, ctx: commands.Context, mode: str = "status"):
        guild_id = ctx.guild.id
        mode = mode.strip().lower()
        keywords_display = self._format_llm_keywords(self.llm_keyword[guild_id])

        if mode in {"on", "enable", "enabled"}:
            self.llm_enabled[guild_id] = True
            await ctx.reply(
                f"LLM trigger enabled. keywords=`{keywords_display}`, "
                f"timeout={self.llm_timeout_seconds[guild_id]:.1f}s"
            )
        elif mode in {"off", "disable", "disabled"}:
            self.llm_enabled[guild_id] = False
            await ctx.reply("LLM trigger disabled.")
        else:
            await ctx.reply(
                f"LLM trigger is {'on' if self.llm_enabled[guild_id] else 'off'}, "
                f"keywords=`{keywords_display}`, timeout={self.llm_timeout_seconds[guild_id]:.1f}s, "
                f"endpoint=`{self.llm_endpoint}`"
            )

    @commands.command()
    async def llmkeyword(self, ctx: commands.Context, *, keyword: str):
        guild_id = ctx.guild.id
        keywords = self._parse_llm_keywords(keyword)
        if not keywords:
            await ctx.reply("Keyword list cannot be empty.")
            return
        self.llm_keyword[guild_id] = keywords
        await ctx.reply(f"LLM trigger keywords set to `{self._format_llm_keywords(keywords)}`.")

    @commands.command()
    async def llmtimeout(self, ctx: commands.Context, seconds: float):
        guild_id = ctx.guild.id
        seconds = max(1.0, min(60.0, float(seconds)))
        self.llm_timeout_seconds[guild_id] = seconds
        await ctx.reply(f"LLM timeout set to `{seconds:.1f}` seconds.")

    @commands.command()
    async def passive(self, ctx: commands.Context, mode: str = "status"):
        guild_id = ctx.guild.id
        mode = mode.strip().lower()

        if mode in {"on", "enable", "enabled"}:
            self.passive_enabled[guild_id] = True
            await self._ensure_passive_task(guild_id)
            await ctx.reply(
                f"Passive comments enabled. interval={self.passive_interval_seconds[guild_id]:.0f}s "
                f"timeout={self.passive_timeout_seconds:.1f}s"
            )
        elif mode in {"off", "disable", "disabled"}:
            self.passive_enabled[guild_id] = False
            self._cancel_passive_task(guild_id)
            await ctx.reply("Passive comments disabled.")
        else:
            await ctx.reply(
                f"Passive comments are {'on' if self.passive_enabled[guild_id] else 'off'}, "
                f"interval={self.passive_interval_seconds[guild_id]:.0f}s, "
                f"timeout={self.passive_timeout_seconds:.1f}s, "
                f"history_items={len(self._conversation_history[guild_id])}"
            )

    @commands.command()
    async def passiveinterval(self, ctx: commands.Context, seconds: float):
        guild_id = ctx.guild.id
        seconds = max(15.0, min(600.0, float(seconds)))
        self.passive_interval_seconds[guild_id] = seconds
        await ctx.reply(f"Passive comment interval set to `{seconds:.0f}` seconds.")

    @commands.command()
    async def record_user(self, ctx: commands.Context, user_id: int, seconds: int = 10):
        """Record one user's audio to WAV (Pycord filters) and run selected STT backend."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.reply("Join a voice channel first.")
            return

        seconds = max(1, min(60, int(seconds)))

        vc = ctx.voice_client
        if not vc or not vc.is_connected():
            vc = await ctx.author.voice.channel.connect()
        elif vc.channel != ctx.author.voice.channel:
            await vc.move_to(ctx.author.voice.channel)

        member = ctx.guild.get_member(user_id)
        if member is None:
            await ctx.reply(f"User `{user_id}` is not in this guild.")
            return

        out_dir = os.path.abspath(self.stt_dump_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        wav_path = os.path.join(out_dir, f"record_user_{member.id}_{safe_label(str(member))}_{ts}.wav")

        done = asyncio.Event()
        recorded_bytes = {}

        async def finished(sink: discord.sinks.Sink, channel, *args):
            # Save only that user's audio
            audio = sink.audio_data.get(user_id)
            if audio:
                audio.file.seek(0)
                with open(wav_path, "wb") as f:
                    f.write(audio.file.read())
                recorded_bytes["ok"] = True
            done.set()

        # filter to just this user + auto-stop after N seconds
        filters = {"users": [user_id], "time": seconds, "max_size": 0}
        sink = discord.sinks.WaveSink(filters=filters)

        # stop any existing recording
        try:
            if getattr(vc, "recording", False):
                vc.stop_recording()
        except Exception:
            pass

        vc.start_recording(sink, finished, ctx.channel)
        await ctx.reply(f"Recording `{member}` for {seconds}s -> `{wav_path}`")
        await done.wait()

        # STT on the saved wav
        text = ""
        error = None
        backend = self.stt_backend[ctx.guild.id]
        try:
            if backend == "whisper":
                text = await asyncio.to_thread(
                    self._recognize_whisper_from_wav_sync,
                    wav_path,
                    self.stt_language[ctx.guild.id],
                )
            else:
                text = await asyncio.to_thread(
                    self._recognize_wav_file_sync,
                    wav_path,
                    self.stt_language[ctx.guild.id],
                )
        except sr.UnknownValueError:
            text = ""
        except Exception as exc:
            error = str(exc)

        if error:
            await ctx.reply(f"Saved recording to `{wav_path}` but STT (`{backend}`) failed: {error}")
        else:
            preview = text[:200] if text else "(empty)"
            await ctx.reply(f"Saved `{wav_path}`. STT (`{backend}`) preview: `{preview}`")

    @commands.command()
    async def threshold(self, ctx: commands.Context, value: int):
        # kept for compatibility; in Pycord version we primarily use RMS threshold
        guild_id = ctx.guild.id
        value = max(1, min(127, value))
        self.voice_threshold[guild_id] = value
        await ctx.reply(f"Voice threshold set to `{value}` (note: Pycord path uses RMS threshold primarily).")

    @commands.command()
    async def rmsthreshold(self, ctx: commands.Context, value: int):
        guild_id = ctx.guild.id
        value = max(50, min(5000, value))
        self.rms_threshold[guild_id] = value
        await ctx.reply(
            f"Base RMS start threshold set to `{value}` "
            f"(adaptive start is `max(base, noise_floor + margin)`)."
        )

    @commands.command()
    async def rmsmargin(self, ctx: commands.Context, value: int):
        guild_id = ctx.guild.id
        value = max(40, min(4000, value))
        self.rms_margin[guild_id] = value
        await ctx.reply(
            f"Adaptive RMS margin set to `{value}` "
            f"(higher = harder to trigger speech in noisy rooms)."
        )

    @commands.command()
    async def silence(self, ctx: commands.Context, seconds: float):
        guild_id = ctx.guild.id
        seconds = max(0.2, min(3.0, float(seconds)))
        self.silence_timeout[guild_id] = seconds
        await ctx.reply(f"Silence timeout set to `{seconds:.2f}`s.")

    @commands.command()
    async def maxutt(self, ctx: commands.Context, seconds: float):
        guild_id = ctx.guild.id
        seconds = max(2.0, min(60.0, float(seconds)))
        self.max_utterance_seconds[guild_id] = seconds
        await ctx.reply(f"Max utterance length set to `{seconds:.1f}`s.")

    @commands.command()
    async def stats(self, ctx: commands.Context):
        guild_id = ctx.guild.id
        packets = self.packet_counts[guild_id]
        await ctx.reply(
            f"PCM_chunks={packets}, "
            f"STT={'on' if self.stt_enabled[guild_id] else 'off'}, lang={self.stt_language[guild_id]}, "
            f"backend={self.stt_backend[guild_id]}, engine={self.stt_engine}, whisper_model={self.whisper_model_name}, "
            f"rms_base={self.rms_threshold[guild_id]}, rms_margin={self.rms_margin[guild_id]}, "
            f"silence={self.silence_timeout[guild_id]:.2f}s, maxutt={self.max_utterance_seconds[guild_id]:.1f}s, "
            f"dump={'on' if self.stt_debug_dump[guild_id] else 'off'}, "
            f"llm={'on' if self.llm_enabled[guild_id] else 'off'} keywords=`{self._format_llm_keywords(self.llm_keyword[guild_id])}` "
            f"llm_timeout={self.llm_timeout_seconds[guild_id]:.1f}s, "
            f"passive={'on' if self.passive_enabled[guild_id] else 'off'} "
            f"passive_interval={self.passive_interval_seconds[guild_id]:.0f}s "
            f"league={'on' if self.league_enabled[guild_id] else 'off'} "
            f"league_interval={max(30.0, float(self.league_poll_interval_seconds)):.0f}s "
            f"history_items={len(self._conversation_history[guild_id])}"
        )

    @commands.command()
    async def die(self, ctx: commands.Context):
        with contextlib.suppress(Exception):
            if ctx.voice_client:
                try:
                    if getattr(ctx.voice_client, "recording", False):
                        ctx.voice_client.stop_recording()
                except Exception:
                    pass
                self._cancel_passive_task(ctx.guild.id)
                self._cancel_league_task(ctx.guild.id)
                self._connections.pop(ctx.guild.id, None)
                self._conversation_history[ctx.guild.id].clear()
                await ctx.voice_client.disconnect()
        await ctx.bot.close()


@bot.event
async def on_ready():
    if bot.get_cog("Testing") is None:
        maybe_coro = bot.add_cog(Testing(bot))
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro
    testing_cog = bot.get_cog("Testing")
    if testing_cog is not None:
        await testing_cog._ensure_watchdog()

    print(f"Logged in as {bot.user.id}/{bot.user}")
    print("------")


bot.run(BOT_TOKEN)
