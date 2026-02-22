/* ── BLACK-SCHOLES ── */

/**
 * Cumulative Standard Normal Distribution approximation
 */
function CND(x) {
  const a1=0.31938153, a2=-0.356563782, a3=1.781477937, a4=-1.821255978, a5=1.330274429;
  const L=Math.abs(x), K=1/(1+0.2316419*L);
  const w=1-1/Math.sqrt(2*Math.PI)*Math.exp(-L*L/2)*(a1*K+a2*K**2+a3*K**3+a4*K**4+a5*K**5);
  return x>=0?w:1-w;
}

/**
 * Black-Scholes call and put pricing
 */
function calcBS(S, K, T, r, v) {
  const d1 = (Math.log(S/K) + (r + v*v/2)*T) / (v*Math.sqrt(T));
  const d2 = d1 - v*Math.sqrt(T);
  return {
    call: S*CND(d1) - K*Math.exp(-r*T)*CND(d2),
    put:  K*Math.exp(-r*T)*CND(-d2) - S*CND(-d1)
  };
}

/**
 * Read inputs, run Black-Scholes, update result display.
 * Returns false if inputs are invalid.
 */
function calculate() {
  const S = parseFloat(document.getElementById('stock').value);
  const K = parseFloat(document.getElementById('strike').value);
  const T = parseFloat(document.getElementById('time').value);
  const r = parseFloat(document.getElementById('rate').value);
  const v = parseFloat(document.getElementById('vol').value);

  if ([S, K, T, r, v].some(isNaN)) {
    alert('Please fill all option parameter fields.');
    return false;
  }

  const { call, put } = calcBS(S, K, T, r, v);
  document.getElementById('call-val').textContent = '$' + call.toFixed(2);
  document.getElementById('put-val').textContent  = '$' + put.toFixed(2);
  return true;
}


/* ── GEOCODING (OpenStreetMap Nominatim — no key needed) ── */

/**
 * Convert a place name string to { lat, lon, display }
 */
async function geocode(query) {
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`;
  const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } });
  const data = await res.json();

  if (!data.length) throw new Error('Location not found. Try a different city name.');

  return {
    lat:     parseFloat(data[0].lat),
    lon:     parseFloat(data[0].lon),
    display: data[0].display_name.split(',').slice(0, 2).join(', ')
  };
}


/* ── OPEN-METEO (free, no API key required) ── */

/**
 * Fetch current atmospheric conditions for the given coordinates.
 */
async function fetchClimate(lat, lon) {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
    `&current=temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation,surface_pressure,cloud_cover,soil_temperature_6cm` +
    `&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=auto&precipitation_unit=inch`;

  const res = await fetch(url);
  if (!res.ok) throw new Error('Climate API error');
  return res.json();
}


/* ── CLIMATE RISK SCORING ── */

/**
 * Derive four risk scores (0–100) and a composite from live weather data.
 */
function computeScores(data) {
  const c        = data.current;
  const wind     = c.wind_speed_10m      ?? 0;
  const gust     = c.wind_gusts_10m      ?? 0;
  const precip   = c.precipitation       ?? 0;
  const temp     = c.temperature_2m      ?? 60;
  const pressure = c.surface_pressure    ?? 1013;
  const soil     = c.soil_temperature_6cm ?? 55;
  const cloud    = c.cloud_cover         ?? 50;

  // Extreme Weather: wind, gusts, precipitation (hurricane threshold ~74 mph)
  const wScore      = Math.min(100, wind   / 74   * 100);
  const gScore      = Math.min(100, gust   / 90   * 100);
  const pScore      = Math.min(100, precip / 1.0  * 100);
  const weatherRisk = Math.round(wScore*0.35 + gScore*0.35 + pScore*0.30);

  // Carbon / Heat: deviation from temperate baseline (65 °F)
  const tempDev   = Math.abs(temp - 65);
  const carbonRisk = Math.round(Math.min(100, tempDev / 45 * 100));

  // Agricultural: soil temperature stress outside optimal range (50–75 °F) + cloud stress
  const soilOpt    = soil >= 50 && soil <= 75
    ? 0
    : Math.min(100, (Math.abs(soil - 62.5) - 12.5) / 30 * 100);
  const cloudStress = cloud > 90 ? 25 : cloud < 15 ? 20 : 0;
  const agriRisk    = Math.round(Math.min(100, soilOpt*0.65 + cloudStress + pScore*0.15));

  // Sea Level / Coastal: pressure drop below 1013 hPa is a key storm-surge indicator
  const pressureDrop = Math.max(0, 1013 - pressure);
  const pressScore   = Math.min(100, pressureDrop / 30 * 100);
  const seaRisk      = Math.round(Math.min(100, pressScore*0.45 + wScore*0.35 + pScore*0.20));

  const composite = Math.round((weatherRisk + carbonRisk + agriRisk + seaRisk) / 4);

  return { weatherRisk, carbonRisk, agriRisk, seaRisk, composite,
           raw: { wind, gust, precip, temp, pressure, soil, cloud } };
}


/* ── UI HELPERS ── */

function badgeHTML(score) {
  if (score < 35) return '<span class="status-badge badge-low">Low</span>';
  if (score < 65) return '<span class="status-badge badge-mod">Moderate</span>';
  return '<span class="status-badge badge-high">High</span>';
}

/**
 * Update a risk card's numeric display, progress bar, and badge.
 */
function setScore(scoreId, fillId, badgeId, score) {
  document.getElementById(scoreId).textContent     = score;
  document.getElementById(fillId).style.width      = score + '%';
  document.getElementById(badgeId).innerHTML       = badgeHTML(score);
}

/**
 * Update the composite ring, verdict label, and description.
 */
function setComposite(score) {
  const ring    = document.getElementById('composite-ring');
  const verdict = document.getElementById('composite-verdict');
  const sub     = document.getElementById('composite-sub');

  ring.textContent = score;

  const color = score < 35 ? '#3dffa0' : score < 65 ? '#f5c518' : '#ff4d4d';
  ring.style.borderColor = color;
  ring.style.color       = color;

  const messages = [
    [0,   25, 'Minimal Climate Stress',    '#3dffa0', 'Benign atmospheric conditions — minimal disruption risk across all four climate dimensions.'],
    [25,  45, 'Low-to-Moderate Risk',      '#3dffa0', 'Mild weather anomalies detected. Climate-sensitive sectors may see minor volatility impact.'],
    [45,  65, 'Elevated Climate Exposure', '#f5c518', 'Meaningful atmospheric stress across multiple dimensions. Relevant for energy, agriculture & insurance sectors.'],
    [65,  80, 'High Climate Risk',         '#f5c518', 'Significant weather events or anomalies present. Option pricing in climate-sensitive stocks may need adjustment.'],
    [80, 101, 'Extreme Climate Conditions','#ff4d4d', 'Severe atmospheric readings. High potential for supply disruption, physical asset damage & volatility spikes.'],
  ];

  for (const [lo, hi, label, col, desc] of messages) {
    if (score >= lo && score < hi) {
      verdict.textContent  = label;
      verdict.style.color  = col;
      sub.textContent      = desc;
      break;
    }
  }
}

/**
 * Reset the dashboard to a loading / pending state.
 */
function resetDashboard() {
  document.getElementById('loc-dot').className  = 'dot';
  document.getElementById('loc-text').textContent = 'Fetching climate data…';
  document.getElementById('loc-sub').textContent  = '';

  ['weather', 'carbon', 'agri', 'sea'].forEach(k => {
    document.getElementById('score-' + k).textContent = '…';
    document.getElementById('fill-'  + k).style.width = '0%';
    document.getElementById('badge-' + k).innerHTML   = '';
  });

  const ring = document.getElementById('composite-ring');
  ring.textContent       = '…';
  ring.style.borderColor = 'var(--border)';
  ring.style.color       = 'var(--muted)';

  document.getElementById('composite-verdict').textContent  = 'Loading…';
  document.getElementById('composite-verdict').style.color  = 'var(--muted)';
  document.getElementById('composite-sub').textContent      = '';
}


/* ── MAIN ORCHESTRATOR ── */

async function runAll() {
  if (!calculate()) return;

  const locInput = document.getElementById('location').value.trim();
  if (!locInput) { alert('Please enter a city or region for climate data.'); return; }

  resetDashboard();

  try {
    const geo  = await geocode(locInput);
    const data = await fetchClimate(geo.lat, geo.lon);
    const { weatherRisk, carbonRisk, agriRisk, seaRisk, composite, raw } = computeScores(data);

    // Location bar
    document.getElementById('loc-dot').className        = 'dot live';
    document.getElementById('loc-text').textContent     = geo.display;
    document.getElementById('loc-sub').textContent      =
      'Updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Risk cards
    setScore('score-weather', 'fill-weather', 'badge-weather', weatherRisk);
    setScore('score-carbon',  'fill-carbon',  'badge-carbon',  carbonRisk);
    setScore('score-agri',    'fill-agri',    'badge-agri',    agriRisk);
    setScore('score-sea',     'fill-sea',     'badge-sea',     seaRisk);
    setComposite(composite);

    // Raw atmospheric readings
    const hiTemp = raw.temp > 90, loTemp = raw.temp < 32;
    document.getElementById('d-temp').className     = 'dv' + (hiTemp ? ' hi' : loTemp ? ' lo' : '');
    document.getElementById('d-temp').textContent   = raw.temp.toFixed(1) + '°F';

    document.getElementById('d-wind').className     = 'dv' + (raw.wind >= 50 ? ' hi' : raw.wind < 5 ? ' lo' : '');
    document.getElementById('d-wind').textContent   = raw.wind.toFixed(1) + ' mph';

    document.getElementById('d-gust').className     = 'dv' + (raw.gust >= 65 ? ' hi' : '');
    document.getElementById('d-gust').textContent   = raw.gust.toFixed(1) + ' mph';

    document.getElementById('d-precip').className   = 'dv' + (raw.precip > 0.5 ? ' hi' : raw.precip === 0 ? ' lo' : '');
    document.getElementById('d-precip').textContent = raw.precip.toFixed(2) + ' in/hr';

    const pDrop = 1013 - raw.pressure;
    document.getElementById('d-pressure').className   = 'dv' + (pDrop > 15 ? ' hi' : pDrop < -5 ? ' lo' : '');
    document.getElementById('d-pressure').textContent = raw.pressure.toFixed(1) + ' hPa';

    document.getElementById('d-soil').className   = 'dv' + (raw.soil > 80 ? ' hi' : raw.soil < 35 ? ' lo' : '');
    document.getElementById('d-soil').textContent = raw.soil.toFixed(1) + '°F';

    document.getElementById('d-cloud').className   = 'dv';
    document.getElementById('d-cloud').textContent = raw.cloud + '%';

  } catch (e) {
    document.getElementById('loc-dot').className    = 'dot';
    document.getElementById('loc-text').textContent = 'Error: ' + e.message;
  }
}


/* ── GEOLOCATION ── */

/**
 * Use the browser's Geolocation API to auto-fill the location field.
 */
function geoLocate() {
  if (!navigator.geolocation) {
    alert('Geolocation is not available in this browser.');
    return;
  }

  navigator.geolocation.getCurrentPosition(async pos => {
    const { latitude: lat, longitude: lon } = pos.coords;
    try {
      const res  = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
      const data = await res.json();
      const city = data.address?.city || data.address?.town || data.address?.county || 'My Location';
      const cc   = (data.address?.country_code || '').toUpperCase();
      document.getElementById('location').value = `${city}${cc ? ', ' + cc : ''}`;
    } catch {
      document.getElementById('location').value = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
    }
  }, () => alert('Could not retrieve your location.'));
}


/* ── SAMPLE DATA ── */

function loadSamples() {
  document.getElementById('stock').value    = 100;
  document.getElementById('strike').value   = 100;
  document.getElementById('time').value     = 1;
  document.getElementById('rate').value     = 0.05;
  document.getElementById('vol').value      = 0.2;
  document.getElementById('location').value = 'Miami, FL';
  runAll();
}