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
    const response = await fetch(upstream, {
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
