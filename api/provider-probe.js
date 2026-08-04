module.exports = async function handler(req, res) {
  const url = new URL('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard');
  const date = req.query?.date;
  if (date) url.searchParams.set('dates', String(date).replaceAll('-', ''));
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(10000),
      headers: { Accept: 'application/json', 'User-Agent': 'SwooshAI/1.0' },
    });
    const body = await response.text();
    return res.status(response.status).json({
      ok: response.ok,
      status: response.status,
      provider: 'espn',
      body_prefix: body.slice(0, 1000),
    });
  } catch (error) {
    return res.status(502).json({ ok: false, provider: 'espn', error: error.message });
  }
};
