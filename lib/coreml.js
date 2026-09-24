const fs = require('fs')
const fsp = require('fs/promises')
const path = require('path')
const { Tokenizer } = require('@huggingface/tokenizers')
const { cacheDir, downloadFile } = require('./bundle.js')

const REPO = 'FluidInference/laya-coreml'
const MODEL_BASE = `https://huggingface.co/${REPO}/resolve/main`
const TREE_API = `https://huggingface.co/api/models/${REPO}/tree/main`
const NATIVE_URLS = [
  'https://unpkg.com/coreml@0.0.1/dist/coreml.node',
  'https://cdn.jsdelivr.net/npm/coreml@0.0.1/dist/coreml.node'
]
const CACHE_DIR = path.join(cacheDir('laya'), 'coreml')
const PRECISIONS = ['fp16', 'e8']
const AVAILABLE_LENGTHS = [128, 256, 512, 1024]
const MAX_OPTIONS = 32
const DTYPES = { i32: 1, f32: 4 }

// laya-multilingual maps its special markers onto the Gemma vocabulary
// (upstream tokenizer_config: cls <bos>, sep <eos>, pad <pad>, mask <mask>)
const SPECIAL_IDS = { cls: 2, sep: 1, pad: 0, mask: 4, maskToken: '[MASK]' }

/* eslint-disable no-loss-of-precision -- float32 calibration values from the laya checkpoint */
const AGENT_CONFIG = {
  max_len: 512,
  head_max_len: 192,
  temperature: [1.6369030475616455, 1.2514300345373412, 1.9833999103864437],
  temperature_by_options: {
    'choice:2': 1.9063563346862793,
    'choice:3-5': 1.7601518630981445,
    'choice:6-10': 1.0000158548355103,
    'choice:11+': 0.10058288276100159,
    'score:3-5': 1.2514300345373412,
    'noul:2': 1.9833999103864437
  }
}
/* eslint-enable no-loss-of-precision */

exports.load = async function (opts = {}) {
  if (process.platform !== 'darwin') throw new Error('coreml backend requires macOS')

  const native = await ensureNative()
  const precision = opts.precision || 'fp16'
  if (!PRECISIONS.includes(precision)) throw new Error(`precision must be one of ${PRECISIONS}`)

  const lengths = (opts.lengths || [128, 512]).slice().sort((a, b) => a - b)
  if (!lengths.length) throw new Error('lengths must list at least one bucket')
  if (!lengths.every(length => AVAILABLE_LENGTHS.includes(length))) {
    throw new Error(`lengths must be a subset of ${AVAILABLE_LENGTHS}`)
  }

  const modelDir = await ensureBundle(lengths, precision)
  const { tokenizer } = await loadTokenizer(path.join(modelDir, 'tokenizer.json'))

  const computeUnits = opts.computeUnits || {}
  const buckets = []
  for (const length of lengths) {
    // 'all' by default: on some machines the ANE path returns non-finite
    // values for a minority of prompts, and the GPU is within ~1 ms
    const units = computeUnits[length] || 'all'
    if (!['all', 'cpu', 'gpu', 'ane'].includes(units)) {
      throw new Error(`computeUnits[${length}] must be one of all, cpu, gpu, ane`)
    }

    const model = native.open(path.join(modelDir, bucketFile(length, precision)), { units, lpaog: true })
    buckets.push({ length, model })
  }

  const config = { ...AGENT_CONFIG, max_len: lengths[lengths.length - 1] }

  const engine = {
    name: 'laya-coreml',
    padToMultiple: null,
    tokenizer,
    ids: SPECIAL_IDS,
    config,
    modelDir,

    async forward (batch) {
      if (batch.options > MAX_OPTIONS) {
        throw new Error(`coreml backend supports up to ${MAX_OPTIONS} options, got ${batch.options}`)
      }

      const logits = []
      const actProbabilities = []

      for (let row = 0; row < batch.size; row++) {
        const output = predictRow(native, buckets, batch, row)
        logits.push(output.logits)
        actProbabilities.push(output.action)
      }

      return { logits, actProbabilities }
    },

    async close () {}
  }

  await warmUp(native, buckets)

  return engine
}

function predictRow (native, buckets, batch, row) {
  const realLength = countReal(batch.attention[row])
  const bucket = buckets.find(bucket => realLength <= bucket.length) || buckets[buckets.length - 1]
  const length = bucket.length
  const features = native.features
  const input = features.new(4)

  const ids = features.multiarray.new([1, length], DTYPES.i32)
  const attention = features.multiarray.new([1, length], DTYPES.i32)
  const idsView = new Int32Array(ids.buffer)
  const attentionView = new Int32Array(attention.buffer)
  // MLMultiArray memory is uninitialized: every element must be written
  idsView.fill(SPECIAL_IDS.pad)
  attentionView.fill(0)
  for (let i = 0; i < Math.min(realLength, length); i++) {
    idsView[i] = batch.inputIds[row * batch.length + i]
    attentionView[i] = 1
  }
  features.set(input, 'input_ids', ids)
  features.set(input, 'attention_mask', attention)

  const markerMap = features.multiarray.new([1, MAX_OPTIONS, length], DTYPES.f32)
  const markerView = new Float32Array(markerMap.buffer)
  markerView.fill(0)
  for (let slot = 0; slot < batch.options; slot++) {
    const position = batch.markerPos[row * batch.options + slot]
    if (batch.markerMask[row][slot] && position < length) {
      markerView[slot * length + position] = 1
    }
  }
  features.set(input, 'marker_map', markerMap)

  const questionType = features.multiarray.new([1, 3], DTYPES.f32)
  const typeView = new Float32Array(questionType.buffer)
  typeView.fill(0)
  typeView[batch.qtype[row]] = 1
  features.set(input, 'question_type', questionType)

  const output = native.predict(bucket.model, input)

  return {
    logits: readOutput(features, output, 'logits'),
    action: readOutput(features, output, 'action_probabilities')
  }
}

function readOutput (features, output, name) {
  const array = features.multiarray.get(features.get(output, name))
  const values = Array.from(new Float32Array(array.buffer))
  if (!values.every(value => Number.isFinite(value))) {
    throw new Error(`coreml backend returned non-finite ${name}; try computeUnits "all" or "gpu"`)
  }

  return values
}

function countReal (attention) {
  let count = 0
  while (count < attention.length && attention[count]) count++
  return count
}

async function warmUp (native, buckets) {
  for (const bucket of buckets) {
    const realLength = 8
    const batch = {
      size: 1,
      length: bucket.length,
      options: 2,
      inputIds: new Int32Array(bucket.length).fill(SPECIAL_IDS.pad),
      attention: [Array.from({ length: realLength }, () => 1)],
      markerPos: new Int32Array([2, 4]),
      markerMask: [[1, 1]],
      qtype: new Int32Array([0])
    }

    predictRow(native, [bucket], batch, 0)
  }
}

function bucketFile (length, precision) {
  return `laya_multilingual_${precision}_L${length}_options32.mlmodelc`
}

async function loadTokenizer (file) {
  const raw = await fsp.readFile(file, 'utf8')
  return { tokenizer: new Tokenizer(JSON.parse(raw), {}) }
}

async function ensureBundle (lengths, precision) {
  const dir = path.join(CACHE_DIR, precision)

  await Promise.all([
    ensureFile('tokenizer.json', path.join(dir, 'tokenizer.json')),
    ...lengths.map(length => ensureBucket(dir, bucketFile(length, precision)))
  ])

  return dir
}

async function ensureBucket (dir, name) {
  const files = await listTree(name)
  await Promise.all(files.map(file => ensureFile(file.path, path.join(dir, file.path))))
}

async function ensureFile (repoPath, target) {
  if (fs.existsSync(target)) return
  fs.mkdirSync(path.dirname(target), { recursive: true })
  await downloadFile(`${MODEL_BASE}/${repoPath}`, target)
}

async function listTree (prefix) {
  const response = await fetch(`${TREE_API}/${prefix}?recursive=true`)
  if (!response.ok) throw new Error(`Could not list ${prefix}: ${response.status} ${response.statusText}`)

  const entries = await response.json()
  return entries.filter(entry => entry.type === 'file')
}

async function ensureNative () {
  const target = path.join(CACHE_DIR, 'native', 'coreml.node')
  if (fs.existsSync(target)) return require(target)

  fs.mkdirSync(path.dirname(target), { recursive: true })

  let lastError = null
  for (const url of NATIVE_URLS) {
    try {
      await downloadFile(url, target)
      return require(target)
    } catch (err) {
      lastError = err
    }
  }

  throw lastError
}
