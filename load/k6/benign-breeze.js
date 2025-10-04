import http from 'k6/http';
import { sleep } from 'k6';
import exec from 'k6/execution';

/** =======================
 *   Helpers & Env parsing
 *  ======================= */
function envNum(name, def, min = 0, max = Number.POSITIVE_INFINITY) {
  const raw = __ENV[name];
  const v = raw ? Number(raw) : def;
  if (!Number.isFinite(v) || v < min || v > max) {
    throw new Error(`ENV ${name} invalid: got "${raw}", expected number in [${min}, ${max}]`);
  }
  return v;
}
function envStr(name, def) {
  return (__ENV[name] && __ENV[name].trim()) || def;
}
function parseList(name, defArr) {
  const raw = __ENV[name];
  if (!raw) return defArr;
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}

const CF_DOMAIN = envStr('CF_DOMAIN', '');
const BASE_URL = envStr('BASE_URL', CF_DOMAIN ? `https://${CF_DOMAIN}` : 'https://example.com');
if (!BASE_URL.startsWith('http')) throw new Error(`BASE_URL must include scheme, got: ${BASE_URL}`);

const WARM_MINS   = envNum('WARM_MINS',   5, 0.1, 240);
const BREEZE_MINS = envNum('BREEZE_MINS', 30, 0.1, 480);
const COOL_MINS   = envNum('COOL_MINS',   5, 0.1, 240);

const WARM_RPS    = envNum('WARM_RPS',    1,  0,  5000);
const BREEZE_RPS  = envNum('BREEZE_RPS',  2,  0,  5000);
const COOL_RPS    = envNum('COOL_RPS',    1,  0,  5000);

// Capacity planning for arrival-rate executors:
const EST_LAT_MS       = envNum('EST_LAT_MS',       400, 10, 60000);
const THINK_MS         = envNum('THINK_MS',         50,  0,  60000);
const VU_SAFETY_FACTOR = envNum('VU_SAFETY_FACTOR', 4,   1,  20);
const VU_MAX_CAP       = envNum('VU_MAX_CAP',       10, 1,  20000);

const peakRps = Math.max(WARM_RPS, BREEZE_RPS, COOL_RPS);
const estIterSec = Math.max((EST_LAT_MS + THINK_MS) / 1000, 0.001);
const preAllocatedVUs = Math.min(VU_MAX_CAP, Math.max(1, Math.ceil(peakRps * estIterSec * VU_SAFETY_FACTOR)));
const maxVUs = Math.min(VU_MAX_CAP, Math.max(preAllocatedVUs, preAllocatedVUs * 2));

const GRACEFUL_STOP = envStr('GRACEFUL_STOP', '30s');

const BENIGN_ROUTES = parseList('BENIGN_ROUTES', [
  '/', '/health', '/products', '/products/1', '/search?q=shoes',
  '/assets/app.css', '/assets/app.js', '/images/logo.png'
]);

const STATIC_RE = /\.(css|js|png|jpe?g|gif|ico|svg|webp|woff2?)$/i;

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)] }

function do_request() {
  const path = pick(BENIGN_ROUTES);
  const url = `${BASE_URL}${path}`;
  const isStatic = STATIC_RE.test(path) || path.startsWith('/assets') || path.startsWith('/static');
  const tags = {
    route: path,
    scenario_kind: 'benign',
    traffic: 'breeze',
    static_asset: isStatic ? 'true' : 'false',
  };
  http.get(url, { tags });
  sleep(THINK_MS / 1000);
}

/** =======================
 *   Scenarios (A)
 *  ======================= */
const tWarm  = `${WARM_MINS}m`;
const tMain  = `${BREEZE_MINS}m`;
const tCool  = `${COOL_MINS}m`;

export const options = {
  thresholds: {
    // Global availability & latency SLOs with safe auto-abort:
    http_req_failed: [
      'rate<0.20',
      { threshold: 'rate<0.10', delayAbortEval: '2m' },
    ],
    // Overall latency
    'http_req_duration': [
      'p(95)<3000',
      { threshold: 'p(99)<5000', delayAbortEval: '2m' },
    ],
    // Static assets stricter:
    'http_req_duration{static_asset:true}': [
      { threshold: 'p(95)<1500', delayAbortEval: '2m' },
    ],
  },
  scenarios: {
    breeze_warm: {
      executor: 'constant-arrival-rate',
      rate: WARM_RPS,
      timeUnit: '1s',
      duration: tWarm,
      preAllocatedVUs,
      maxVUs,
      startTime: '0s',
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
    breeze_main: {
      executor: 'constant-arrival-rate',
      rate: BREEZE_RPS,
      timeUnit: '1s',
      duration: tMain,
      preAllocatedVUs,
      maxVUs,
      startTime: tWarm,
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
    breeze_cool: {
      executor: 'constant-arrival-rate',
      rate: COOL_RPS,
      timeUnit: '1s',
      duration: tCool,
      preAllocatedVUs,
      maxVUs,
      startTime: `${WARM_MINS + BREEZE_MINS}m`,
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
  },
};

export function run() {
  do_request();
}

// Optional: print where we run
export function setup() {
  console.log(`[A] benign-breeze base=${BASE_URL}, peakRps=${peakRps}, preAllocatedVUs=${preAllocatedVUs}, maxVUs=${maxVUs}`);
}
