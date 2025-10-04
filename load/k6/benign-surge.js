import http from 'k6/http';
import { sleep } from 'k6';

function envNum(name, def, min = 0, max = Number.POSITIVE_INFINITY) {
  const raw = __ENV[name];
  const v = raw ? Number(raw) : def;
  if (!Number.isFinite(v) || v < min || v > max) throw new Error(`ENV ${name} invalid`);
  return v;
}
function envStr(name, def) { return (__ENV[name] && __ENV[name].trim()) || def }
function parseList(name, defArr) {
  const raw = __ENV[name];
  if (!raw) return defArr;
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)] }

const CF_DOMAIN = envStr('CF_DOMAIN', '');
const BASE_URL = envStr('BASE_URL', CF_DOMAIN ? `https://${CF_DOMAIN}` : 'https://example.com');
if (!BASE_URL.startsWith('http')) throw new Error(`BASE_URL must include scheme`);

const SURGE_RAMP_MINS = envNum('SURGE_RAMP_MINS', 3, 0.1, 120);
const SURGE_PEAK_MINS = envNum('SURGE_PEAK_MINS', 30, 0.1, 480);
const SURGE_COOL_MINS = envNum('SURGE_COOL_MINS', 7, 0.1, 240);

const SURGE_START_RPS = envNum('SURGE_START_RPS', 3, 0, 10000);
const SURGE_PEAK_RPS  = envNum('SURGE_PEAK_RPS', 10, 0, 10000);
const SURGE_END_RPS   = envNum('SURGE_END_RPS', Math.max(2, Math.floor(SURGE_PEAK_RPS/3)), 0, 10000);

const EST_LAT_MS       = envNum('EST_LAT_MS',       400, 10, 60000);
const THINK_MS         = envNum('THINK_MS',         50,  0,  60000);
const VU_SAFETY_FACTOR = envNum('VU_SAFETY_FACTOR', 4,   1,  20);
const VU_MAX_CAP       = envNum('VU_MAX_CAP',       1000,1,  20000);

const peakRps = Math.max(SURGE_START_RPS, SURGE_PEAK_RPS, SURGE_END_RPS);
const estIterSec = Math.max((EST_LAT_MS + THINK_MS) / 1000, 0.001);
const preAllocatedVUs = Math.min(VU_MAX_CAP, Math.max(1, Math.ceil(peakRps * estIterSec * VU_SAFETY_FACTOR)));
const maxVUs = Math.min(VU_MAX_CAP, Math.max(preAllocatedVUs, preAllocatedVUs * 2));

const GRACEFUL_STOP = envStr('GRACEFUL_STOP', '30s');

const BENIGN_ROUTES = parseList('BENIGN_ROUTES', [
  '/', '/catalog', '/item/42', '/search?q=watch', '/api/status',
  '/assets/app.css', '/images/hero.jpg'
]);

const STATIC_RE = /\.(css|js|png|jpe?g|gif|ico|svg|webp|woff2?)$/i;

function do_request() {
  const path = pick(BENIGN_ROUTES);
  const url = `${BASE_URL}${path}`;
  const isStatic = STATIC_RE.test(path) || path.startsWith('/assets') || path.startsWith('/static');
  const tags = {
    route: path,
    scenario_kind: 'benign',
    traffic: 'surge',
    static_asset: isStatic ? 'true' : 'false',
  };
  http.get(url, { tags });
  sleep(THINK_MS / 1000);
}

export const options = {
  thresholds: {
    http_req_failed: [
      'rate<0.35',
      { threshold: 'rate<0.25', delayAbortEval: '2m' },
    ],
    'http_req_duration': [
      'p(95)<3000',
      { threshold: 'p(99)<5000', delayAbortEval: '2m' },
    ],
    'http_req_duration{static_asset:true}': [
      { threshold: 'p(95)<1500', delayAbortEval: '2m' },
    ],
  },
  scenarios: {
    surge: {
      executor: 'ramping-arrival-rate',
      startRate: SURGE_START_RPS,
      timeUnit: '1s',
      preAllocatedVUs,
      maxVUs,
      gracefulStop: GRACEFUL_STOP,
      stages: [
        { duration: `${SURGE_RAMP_MINS}m`, target: SURGE_PEAK_RPS }, // ramp up
        { duration: `${SURGE_PEAK_MINS}m`, target: SURGE_PEAK_RPS }, // hold
        { duration: `${SURGE_COOL_MINS}m`, target: SURGE_END_RPS },  // ramp down
      ],
      exec: 'run',
    },
  },
};

export function run() { do_request(); }
export function setup() {
  console.log(`[C] benign-surge base=${BASE_URL}, peakRps=${peakRps}, preAllocatedVUs=${preAllocatedVUs}, maxVUs=${maxVUs}`);
}
