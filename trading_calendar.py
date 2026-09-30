"""Scheduled TWSE sessions for DAILY-data freshness, not an order calendar.

Unknown years/ambiguous responses never become five-day grace periods. Annual
schedules cannot certify typhoon closures; dated, sourced local overrides are
supported. Latest daily data and validity of a next-open plan are separate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import uuid
import hashlib
from urllib.parse import urlparse
from typing import Optional
import pandas as pd
import requests
from market_data import _taipei_timestamp, _daily_cutoff

CALENDAR_URL = 'https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule'
BUNDLED_DIR = Path(__file__).resolve().parent / 'resources'


def _date(value) -> pd.Timestamp:
    if not isinstance(value, (str, pd.Timestamp)):
        raise ValueError('An explicit date is required')
    t = _taipei_timestamp(value)
    return t.normalize()


def _validate_schedule(obj, year):
    if not isinstance(obj, dict) or obj.get('year') != year:
        raise ValueError('Calendar year mismatch')
    if not obj.get('source') or not obj.get('verified_on'):
        raise ValueError('Calendar provenance missing')
    closed = {_date(v).strftime('%Y-%m-%d') for v in obj['closed_dates']}
    opened = {_date(v).strftime('%Y-%m-%d') for v in obj.get('open_dates', [])}
    if len(closed) < 5 or any(pd.Timestamp(d).year != year for d in closed | opened):
        raise ValueError('Incomplete or cross-year calendar')
    if closed & opened:
        raise ValueError('Conflicting calendar dates')
    return {**obj, 'closed_dates': sorted(closed), 'open_dates': sorted(opened)}


def parse_twse_schedule(payload, year, checked_at):
    """Never assume every row is a closure: last/first trading days are open."""
    if not isinstance(payload, dict) or str(payload.get('stat', '')).lower() != 'ok':
        raise ValueError('TWSE calendar response is not OK')
    try:
        qyear = int(payload['queryYear'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Calendar year missing') from exc
    if qyear not in (year, year - 1911):
        raise ValueError('TWSE returned a different year')
    closed, opened = [], []
    for row in payload.get('data', []):
        if not isinstance(row, list) or len(row) < 3 or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(row[0])):
            raise ValueError('Unsupported calendar row')
        day = _date(row[0]).strftime('%Y-%m-%d')
        text = str(row[1]) + ' ' + str(row[2])
        if any(w in text for w in ('\u7121\u4ea4\u6613', '\u4f11\u5e02', '\u653e\u5047', '\u88dc\u5047')):
            closed.append(day)
        elif any(w in text for w in ('\u958b\u59cb\u4ea4\u6613', '\u6700\u5f8c\u4ea4\u6613', '\u958b\u5e02')):
            opened.append(day)
        else:
            raise ValueError('Unknown calendar event: ' + text[:100])
    return _validate_schedule({'year': year, 'closed_dates': closed, 'open_dates': opened,
                              'source': CALENDAR_URL, 'verified_on': str(checked_at),
                              'kind': 'official_scheduled_sessions'}, year)


class CalendarStore:
    def __init__(self, data_dir, allow_fetch=False, fetcher=None):
        self.root = Path(data_dir) / 'calendars'
        self.allow_fetch = allow_fetch
        self.fetcher = fetcher or requests.get
        self.memo = {}
        self.errors = []

    def year(self, year, now):
        if year in self.memo:
            return self.memo[year]
        cache = self.root / f'twse_{year}.json'
        fallback = None
        for path in (cache, BUNDLED_DIR / f'twse_{year}.json'):
            if path.exists():
                try:
                    item = _validate_schedule(json.loads(path.read_text(encoding='utf-8')), year)
                    if fallback is None:
                        fallback = item
                    # A successful same-day fetch needs no second request.
                    if path == cache and _date(item['verified_on']) == now.normalize():
                        self.memo[year] = item
                        return item
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    self.errors.append('Invalid calendar cache: ' + type(exc).__name__)
        if self.allow_fetch:
            try:
                response = self.fetcher(CALENDAR_URL, params={'response':'json','queryYear':year-1911}, timeout=6)
                response.raise_for_status()
                item = parse_twse_schedule(response.json(), year, now.isoformat())
                self.root.mkdir(parents=True, exist_ok=True)
                temp = cache.with_name(cache.name + '.' + uuid.uuid4().hex + '.tmp')
                try:
                    temp.write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
                    temp.replace(cache)
                finally:
                    temp.unlink(missing_ok=True)
                self.memo[year] = item
                return item
            except (requests.RequestException, ValueError, TypeError, KeyError, OSError) as exc:
                self.errors.append('Calendar fetch unavailable: ' + type(exc).__name__)
        self.memo[year] = fallback
        return fallback

    def overrides(self, now):
        """Optional dated exchange-closure notices; no future-available overrides."""
        path = self.root / 'session_overrides.json'
        records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        for notice in sorted((self.root / 'notices').glob('*.json')):
            records.append(json.loads(notice.read_text(encoding='utf-8')))
        result = {}
        for item in records:
            if (not isinstance(item, dict) or type(item.get('is_open')) is not bool
                    or not item.get('source') or not item.get('available_from')):
                raise ValueError('Override requires date,is_open,source,available_from')
            known = _taipei_timestamp(item['available_from'])
            if known > now:
                continue
            day = _date(item['date']).strftime('%Y-%m-%d')
            if day not in result or known > result[day][0]:
                result[day] = (known, item['is_open'])
        return {k:v[1] for k,v in result.items()}

    def reference(self, now=None):
        now = _taipei_timestamp(now)
        try:
            overrides = self.overrides(now)
            used = {}
            def is_open(day):
                day = day.normalize(); key = day.strftime('%Y-%m-%d')
                if key in overrides:
                    return overrides[key]
                sched = self.year(day.year, now)
                if sched is None:
                    raise ValueError('No verified calendar for ' + str(day.year))
                used[str(day.year)] = {'source':sched['source'], 'verified_on':sched['verified_on']}
                if key in sched['open_dates']:
                    return True
                return day.weekday() < 5 and key not in sched['closed_dates']
            last = _daily_cutoff(now)
            for _ in range(45):
                if is_open(last):
                    break
                last -= pd.Timedelta(days=1)
            else:
                raise ValueError('No completed session within calendar window')
            next_day = last + pd.Timedelta(days=1)
            for _ in range(45):
                if is_open(next_day):
                    break
                next_day += pd.Timedelta(days=1)
            else:
                raise ValueError('No next session within calendar window')
            next_open = next_day + pd.Timedelta(hours=9)
            return {'operator_notice_count':len(overrides), 'status':'KNOWN_SCHEDULE', 'now':now.isoformat(),
                    'expected_date':last.strftime('%Y-%m-%d'), 'next_open':next_open.isoformat(),
                    'next_open_passed':bool(now >= next_open), 'sources':used,
                    'scope':'TWSE scheduled sessions; ad-hoc closures require sourced overrides',
                    'warnings':list(self.errors)}
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return {'status':'UNKNOWN_CALENDAR', 'now':now.isoformat(), 'expected_date':None,
                    'next_open':None, 'next_open_passed':False, 'sources':{},
                    'warnings':self.errors+[str(exc)]}


def calendar_reference(data_dir, now=None, allow_fetch=False):
    return CalendarStore(data_dir, allow_fetch=allow_fetch).reference(now)


def daily_freshness(price_date, reference):
    """Tri-state freshness: a missing calendar is not a fresh/stale assertion."""
    try:
        price = _date(price_date)
        if reference.get('status') != 'KNOWN_SCHEDULE':
            return {'status':'UNKNOWN', 'current_daily':False, 'allow_entry_review':False}
        expected = _date(reference['expected_date'])
        status = 'CURRENT_DAILY' if price == expected else 'STALE' if price < expected else 'FUTURE_OR_UNFINISHED'
        return {'status':status, 'current_daily':status=='CURRENT_DAILY',
                'allow_entry_review':status=='CURRENT_DAILY' and not reference['next_open_passed'],
                'expected_date':expected.strftime('%Y-%m-%d')}
    except (ValueError, TypeError, KeyError):
        return {'status':'INVALID_DATE', 'current_daily':False, 'allow_entry_review':False}


def entry_review_allowed(price_date, plan, reference):
    result = daily_freshness(price_date, reference)
    if not result['allow_entry_review']:
        return False
    if not plan or not plan.get('expires_at'):
        return False
    try:
        now = _taipei_timestamp(reference['now'])
        return _taipei_timestamp(plan['signal_at']) <= now <= _taipei_timestamp(plan['expires_at'])
    except (ValueError, TypeError, KeyError):
        return False


def save_closure_notice(data_dir, day, source, available_from, *, now=None, confirmed=False):
    """Explicit operator-sourced closure, NOT a force-fresh switch.

    Only closure notices are accepted in this UI API. One immutable file per
    notice avoids lost updates across sessions. Source is operator-declared and
    domain checked, not independently verified by this local routine.
    """
    checked = _taipei_timestamp(now)
    if confirmed is not True:
        raise ValueError('Explicit official-notice confirmation is required')
    d = _date(day)
    if available_from is None or not str(available_from).strip():
        raise ValueError('Explicit notice publication time is required')
    known = _taipei_timestamp(available_from)
    if known > checked:
        raise ValueError('A future notice cannot be applied now')
    parsed = urlparse(str(source).strip())
    host=(parsed.hostname or '').lower()
    if parsed.scheme != 'https' or parsed.username or parsed.password or not any(
            host == base or host.endswith('.'+base) for base in ('twse.com.tw','tpex.org.tw')):
        raise ValueError('Provide an HTTPS official TWSE/TPEx notice URL')
    notice={'date':d.strftime('%Y-%m-%d'),'is_open':False,'source':str(source).strip(),
            'available_from':known.isoformat(),'recorded_at':checked.isoformat(),
            'verification':'OPERATOR_CONFIRMED_OFFICIAL_NOTICE_NOT_AUTOMATICALLY_FETCHED'}
    key=hashlib.sha256(json.dumps(notice,sort_keys=True).encode()).hexdigest()
    root=Path(data_dir)/'calendars'/'notices'; root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.json'); tmp=root/(key+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(notice,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)
    return notice
