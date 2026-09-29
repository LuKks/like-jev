const fs = require('fs')
const fsp = require('fs/promises')
const path = require('path')
const { Tokenizer } = require('@huggingface/tokenizers')
const { cacheDir, downloadFile } = require('./bundle.js')

const REPO = 'FluidInference/laya-english-coreml'
const REVISION = '3a5df0d0f257e790b79d1fc7414a0a74e8a103ff'
const MODEL_BASE = `https://huggingface.co/${REPO}/resolve/${REVISION}`
const TREE_API = `https://huggingface.co/api/models/${REPO}/tree/${REVISION}`
const NATIVE_URLS = [
  'https://unpkg.com/coreml@0.0.1/dist/coreml.node',
  'https://cdn.jsdelivr.net/npm/coreml@0.0.1/dist/coreml.node'
]
const CACHE_DIR = path.join(cacheDir('laya'), 'coreml', 'english')
const DTYPES = { i32: 1, f32: 4 }
const LENGTHS = [128, 512]
const MAX_OPTIONS = 32
const IDS = { cls: 50281, sep: 50282, pad: 50283, mask: 50284, maskToken: '[MASK]' }
const CONFIG = {
  max_len: 512,
  head_max_len: 192,
  temperature: [Number('1.6369030475616455'), Number('1.2514300345373412'), Number('1.9833999103864437')],
  temperature_by_options: {
    'choice:2': Number('1.9063563346862793'),
    'choice:3-5': Number('1.7601518630981445'),
    'choice:6-10': Number('1.0000158548355103'),
    'choice:11+': Number('0.10058288276100159'),
    'score:3-5': Number('1.2514300345373412'),
    'noul:2': Number('1.9833999103864437')
  }
}

exports.load = async function (opts = {}) {
  if (process.platform !== 'darwin') throw new Error('Core ML requires macOS')
  if (opts.model && opts.model !== 'english') throw new Error('only the English model is supported')
  if (opts.precision && opts.precision !== 'fp16') throw new Error('English Core ML supports only fp16')

  const lengths = (opts.lengths || LENGTHS).slice().sort((a, b) => a - b)
  if (!lengths.length || !lengths.every(length => LENGTHS.includes(length))) {
    throw new Error(`English Core ML lengths must be selected from ${LENGTHS}`)
  }

  const native = await ensureNative()
  const modelDir = await ensureBundle(lengths)
  const rawTokenizer = await fsp.readFile(path.join(modelDir, 'tokenizer', 'tokenizer.json'), 'utf8')
  const tokenizer = new Tokenizer(JSON.parse(rawTokenizer), {})
  const buckets = []

  for (const length of lengths) {
    const modelPath = path.join(modelDir, modelFile(length))
    buckets.push({ length, model: native.open(modelPath, { units: 'all', lpaog: true }) })
  }

  const engine = {
    name: 'laya-coreml-english',
    padToMultiple: null,
    tokenizer,
    ids: IDS,
    config: { ...CONFIG, max_len: lengths[lengths.length - 1] },
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
  idsView.fill(IDS.pad)
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
    if (batch.markerMask[row][slot] && position < length) markerView[slot * length + position] = 1
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
  if (!values.every(value => Number.isFinite(value))) throw new Error(`coreml returned non-finite ${name}`)
  return values
}

function countReal (attention) {
  let count = 0
  while (count < attention.length && attention[count]) count++
  return count
}

async function warmUp (native, buckets) {
  for (const bucket of buckets) {
    const batch = {
      size: 1,
      length: bucket.length,
      options: 2,
      inputIds: new Int32Array(bucket.length).fill(IDS.pad),
      attention: [Array.from({ length: 8 }, () => 1)],
      markerPos: new Int32Array([2, 4]),
      markerMask: [[1, 1]],
      qtype: new Int32Array([0])
    }
    predictRow(native, [bucket], batch, 0)
  }
}

function modelFile (length) {
  return `laya_english_fp16_L${length}_options32.mlpackage`
}

async function ensureBundle (lengths) {
  const tokenizerPath = 'tokenizer/tokenizer.json'
  await Promise.all([
    ensureFile(tokenizerPath, path.join(CACHE_DIR, tokenizerPath)),
    ...lengths.map(length => ensureModel(modelFile(length)))
  ])
  return CACHE_DIR
}

async function ensureModel (name) {
  const files = await listTree(name)
  await Promise.all(files.map(file => ensureFile(file.path, path.join(CACHE_DIR, file.path))))
}

async function ensureFile (repoPath, target) {
  if (fs.existsSync(target)) return
  fs.mkdirSync(path.dirname(target), { recursive: true })
  await downloadFile(`${MODEL_BASE}/${repoPath}`, target)
}

async function listTree (prefix) {
  const response = await fetch(`${TREE_API}/${prefix}?recursive=true`)
  if (!response.ok) throw new Error(`Could not list ${prefix}: ${response.status} ${response.statusText}`)
  return (await response.json()).filter(entry => entry.type === 'file')
}

async function ensureNative () {
  const target = path.join(cacheDir('laya'), 'coreml', 'native', 'coreml.node')
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
