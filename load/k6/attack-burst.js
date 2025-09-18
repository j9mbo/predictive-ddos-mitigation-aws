import http from 'k6/http';
import { sleep } from 'k6';

/** ============ Helpers ============ */
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

/** ============ Env ============ */
const CF_DOMAIN = envStr('CF_DOMAIN', '');
const BASE_URL = envStr('BASE_URL', CF_DOMAIN ? `https://${CF_DOMAIN}` : 'https://example.com');
if (!BASE_URL.startsWith('http')) throw new Error(`BASE_URL must include scheme`);

const ATT_RAMP_MINS = envNum('ATT_RAMP_MINS', 2, 0.1, 120);
const ATT_PEAK_MINS = envNum('ATT_PEAK_MINS', 8, 0.1, 120);
const ATT_COOL_MINS = envNum('ATT_COOL_MINS', 2, 0.1, 120);

const ATT_TARGET_RPS  = envNum('ATT_TARGET_RPS', 6, 0, 10000);
const ATT_WARM_RPS    = __ENV.ATT_WARM_RPS ? envNum('ATT_WARM_RPS', 0) : Math.max(1, Math.floor(ATT_TARGET_RPS * envNum('ATT_WARM_FRAC', 0.5, 0, 1)));
const ATT_COOL_RPS    = __ENV.ATT_COOL_RPS ? envNum('ATT_COOL_RPS', 0) : Math.max(1, Math.floor(ATT_TARGET_RPS * envNum('ATT_COOL_FRAC', 0.5, 0, 1)));

const ATT_ATTACKY_RATIO = envNum('ATT_ATTACKY_RATIO', 0.8, 0, 1);

const EST_LAT_MS       = envNum('EST_LAT_MS',       400, 10, 60000);
const THINK_MS         = envNum('THINK_MS',         10,  0,  60000);
const VU_SAFETY_FACTOR = envNum('VU_SAFETY_FACTOR', 4,   1,  20);
const VU_MAX_CAP       = envNum('VU_MAX_CAP',       1000,1,  20000);

const peakRps = Math.max(ATT_TARGET_RPS, ATT_WARM_RPS, ATT_COOL_RPS);
const estIterSec = Math.max((EST_LAT_MS + THINK_MS) / 1000, 0.001);
const preAllocatedVUs = Math.min(VU_MAX_CAP, Math.max(1, Math.ceil(peakRps * estIterSec * VU_SAFETY_FACTOR)));
const maxVUs = Math.min(VU_MAX_CAP, Math.max(preAllocatedVUs, preAllocatedVUs * 2));

const GRACEFUL_STOP = envStr('GRACEFUL_STOP', '30s');

const BENIGN_ROUTES  = parseList('BENIGN_ROUTES',  ['/', '/products', '/search?q=bag', '/api/status']);
const ATTACKY_ROUTES = parseList('ATTACKY_ROUTES', [
  '/?a=1', '/?a=2', '/?a=3', '/?a=4', '/?a=5',
  '/login', '/wp-login.php', '/.env', '/admin', '/api'
]);

const STATIC_RE = /\.(css|js|png|jpe?g|gif|ico|svg|webp|woff2?)$/i;

function do_request_attack_mix() {
  // weighted 80% attacky, 20% benign (configurable)
  const useAttack = Math.random() < ATT_ATTACKY_RATIO;
  const path = useAttack ? pick(ATTACKY_ROUTES) : pick(BENIGN_ROUTES);
  const url = `${BASE_URL}${path}`;
  const isStatic = STATIC_RE.test(path) || path.startsWith('/assets') || path.startsWith('/static');
  const tags = {
    route: path,
    scenario_kind: 'attack',
    traffic: 'burst',
    pool: useAttack ? 'attacky' : 'benign',
    static_asset: isStatic ? 'true' : 'false',
  };
  http.get(url, { tags });
  sleep(THINK_MS / 1000);
}

/** ============ Scenarios (B) ============ */
const tWarm = `${ATT_RAMP_MINS}m`;
const tPeak = `${ATT_PEAK_MINS}m`;
const tCool = `${ATT_COOL_MINS}m`;

export const options = {
  thresholds: {
    http_req_failed: [
      'rate<0.20',
      { threshold: 'rate<0.10', abortOnFail: true, delayAbortEval: '1m' },
    ],
    'http_req_duration': [
      'p(95)<3000',
      { threshold: 'p(99)<5000', abortOnFail: true, delayAbortEval: '1m' },
    ],
    'http_req_duration{static_asset:true}': [
      { threshold: 'p(95)<1500', abortOnFail: true, delayAbortEval: '1m' },
    ],
  },
  scenarios: {
    attack_ramp: {
      executor: 'constant-arrival-rate',
      rate: ATT_WARM_RPS,
      timeUnit: '1s',
      duration: tWarm,
      preAllocatedVUs,
      maxVUs,
      startTime: '0s',
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
    attack_peak: {
      executor: 'constant-arrival-rate',
      rate: ATT_TARGET_RPS,
      timeUnit: '1s',
      duration: tPeak,
      preAllocatedVUs,
      maxVUs,
      startTime: tWarm,
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
    attack_cool: {
      executor: 'constant-arrival-rate',
      rate: ATT_COOL_RPS,
      timeUnit: '1s',
      duration: tCool,
      preAllocatedVUs,
      maxVUs,
      startTime: `${ATT_RAMP_MINS + ATT_PEAK_MINS}m`,
      gracefulStop: GRACEFUL_STOP,
      exec: 'run',
    },
  },
};

export function run() {
  do_request_attack_mix();
}

export function setup() {
  console.log(`[B] attack-burst base=${BASE_URL}, peakRps=${peakRps}, preAllocatedVUs=${preAllocatedVUs}, maxVUs=${maxVUs}`);
}
