// Same-schema NBA Stats proxy for the Render backend.
// The proxy preserves NBA game IDs and response bodies; it only changes the
// network path so Render does not call stats.nba.com directly.
module.exports = async function handler(req, res) {
  const requestUrl = new URL(req.url, `https://${req.headers.host || 'localhost'}`);
  const marker = '/api/nba/stats/';
  const rawPath = requestUrl.pathname.split(marker)[1] || '';
  const endpoint = rawPath.split('/').filter(Boolean).at(-1);
  const allowedEndpoints = new Set([
    'scoreboardv3', 'playbyplayv3', 'boxscoresummaryv2',
    'boxscoretraditionalv3', 'commonteamroster', 'leaguegamelog',
    'scheduleleaguev2',
  ]);
  if (!endpoint || !allowedEndpoints.has(endpoint)) {
    return res.status(400).json({ detail: 'Invalid NBA Stats endpoint' });
  }

  const expectedToken = process.env.NBA_STATS_PROXY_TOKEN || '';
  const suppliedToken = req.headers['x-swoosh-proxy-token'] || '';
  if (!expectedToken || suppliedToken !== expectedToken) {
    return res.status(401).json({
      detail: 'NBA Stats proxy authorization required',
      token_present: Boolean(suppliedToken),
    });
  }

  const upstream = new URL(`https://stats.nba.com/stats/${endpoint}`);
  requestUrl.searchParams.forEach((value, key) => upstream.searchParams.append(key, value));

  try {
    if (endpoint === 'scoreboardv3') {
      const gameDate = requestUrl.searchParams.get('GameDate');
      const espnUrl = new URL(
        'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
      );
      if (gameDate) espnUrl.searchParams.set('dates', gameDate.replaceAll('-', ''));
      const espnResponse = await fetch(espnUrl, {
        signal: AbortSignal.timeout(10000),
        headers: {
          Accept: 'application/json',
          'User-Agent': 'SwooshAI/1.0',
        },
      });
      if (!espnResponse.ok) {
        const body = await espnResponse.text();
        return res.status(502).json({
          detail: `ESPN scoreboard returned HTTP ${espnResponse.status}: ${body.slice(0, 240)}`,
        });
      }
      const espnPayload = await espnResponse.json();
      const games = (espnPayload.events || []).map((event) => {
        const competition = event.competitions?.[0] || {};
        const competitors = competition.competitors || [];
        const home = competitors.find((team) => team.homeAway === 'home') || {};
        const away = competitors.find((team) => team.homeAway === 'away') || {};
        const status = competition.status || {};
        const state = status.type?.state || 'pre';
        const statusId = state === 'in' ? 2 : state === 'post' ? 3 : 1;
        const seasonType = event.season?.slug || event.season?.type?.abbreviation || '';
        const gameType = seasonType.includes('post') ? 'playoffs' : 'regular';
        return {
          gameId: `espn:${event.id}`,
          provider: 'espn',
          providerGameId: String(event.id),
          gameStatus: statusId,
          gameStatusText: status.type?.shortDetail || status.type?.detail || '',
          period: Number(status.period || 0),
          gameClock: status.displayClock || '',
          gameDate: event.date || gameDate || '',
          gameDateTimeUTC: event.date || '',
          gameLabel: event.name || '',
          gameSubtype: gameType === 'playoffs' ? 'Playoffs' : '',
          homeTeam: {
            teamTricode: home.team?.abbreviation || '',
            teamName: home.team?.shortDisplayName || home.team?.displayName || '',
            score: home.score || '0',
          },
          awayTeam: {
            teamTricode: away.team?.abbreviation || '',
            teamName: away.team?.shortDisplayName || away.team?.displayName || '',
            score: away.score || '0',
          },
        };
      });
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'public, s-maxage=30, stale-while-revalidate=120');
      return res.status(200).send(JSON.stringify({ scoreboard: { games } }));
    }

    // The live scoreboard is also published as a small CDN artifact. Prefer
    // it for today's slate because stats.nba.com can hang from cloud hosts.
    // Keep the Stats API fallback for historical date queries and if the CDN
    // is unavailable in a given region.
    if (endpoint === 'scoreboardv3') {
      const requestedDate = requestUrl.searchParams.get('GameDate');
      const today = new Intl.DateTimeFormat('en-CA', {
        timeZone: 'America/New_York',
      }).format(new Date());
      if (!requestedDate || requestedDate === today) {
        try {
          const cdnResponse = await fetch(
            'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json',
            {
              signal: AbortSignal.timeout(8000),
              headers: {
                Accept: 'application/json, text/plain, */*',
                Referer: 'https://www.nba.com/',
                'User-Agent': 'Mozilla/5.0 (compatible; SwooshAI/1.0)',
              },
            },
          );
          if (cdnResponse.ok) {
            const cdnBody = await cdnResponse.text();
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Cache-Control', 'public, s-maxage=30, stale-while-revalidate=120');
            return res.status(200).send(cdnBody);
          }
        } catch (cdnError) {
          // Try stats.nba.com below; the CDN is an optimization, not a hard dependency.
        }
      }
    }

    const response = await fetch(upstream, {
      signal: AbortSignal.timeout(12000),
      headers: {
        Accept: 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        Referer: 'https://www.nba.com/',
        'User-Agent': 'Mozilla/5.0 (compatible; SwooshAI/1.0)',
      },
    });
    const body = await response.text();
    if (!response.ok) {
      return res.status(502).json({
        detail: `NBA upstream returned HTTP ${response.status}: ${body.slice(0, 240)}`,
      });
    }
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('json')) {
      return res.status(502).json({
        detail: `NBA upstream returned non-JSON content (${contentType || 'unknown content type'})`,
      });
    }
    res.setHeader('Content-Type', contentType || 'application/json');
    res.setHeader('Cache-Control', endpoint.toLowerCase().includes('scoreboard')
      ? 'public, s-maxage=30, stale-while-revalidate=120'
      : 'no-store');
    return res.status(response.status).send(body);
  } catch (error) {
    return res.status(502).json({ detail: `NBA Stats proxy request failed: ${error.message}` });
  }
};
