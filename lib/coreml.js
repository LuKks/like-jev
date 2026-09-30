const fs = require('fs')
const fsp = require('fs/promises')
const path = require('path')
const { Tokenizer } = require('@huggingface/tokenizers')
const { cacheDir, downloadFile } = require('./bundle.js')

const NATIVE_URLS = [
  'https://unpkg.com/coreml@0.0.1/dist/coreml.node',
  'https://cdn.jsdelivr.net/npm/coreml@0.0.1/dist/coreml.node'
]
const CACHE_ROOT = path.join(cacheDir('laya'), 'coreml')
const DTYPES = { i32: 1, f32: 4 }
const MAX_OPTIONS = 32

/* eslint-disable no-loss-of-precision -- float32 calibration values from the English checkpoint */
const ENGLISH_TEMPERATURE = [1.6369030475616455, 1.2514300345373412, 1.983399510383606]
const ENGLISH_TEMPERATURE_BY_OPTIONS = {
  'choice:2': 1.9063563346862793,
  'choice:3-5': 1.7601518630981445,
  'choice:6-10': 1.0000158548355103,
  'choice:11+': 0.10058288276100159,
  'score:3-5': 1.2514300345373412,
  'noul:2': 1.9833999103864437
}
/* eslint-enable no-loss-of-precision */

const MODEL_PROFILES = {
  english: {
    repo: 'FluidInference/laya-english-coreml',
    revision: '3a5df0d0f257e790b79d1fc7414a0a74e8a103ff',
    tokenizerPath: 'tokenizer/tokenizer.json',
    lengths: [128, 512],
    defaultLengths: [128, 512],
    precisions: ['fp16'],
    // the English cache predates the precision split and stays flat
    cacheDir: path.join(CACHE_ROOT, 'english'),
    maxLength: 512,
    headMaxLength: 192,
    ids: { cls: 50281, sep: 50282, pad: 50283, mask: 50284, maskToken: '[MASK]' },
    temperature: ENGLISH_TEMPERATURE,
    temperatureByOptions: ENGLISH_TEMPERATURE_BY_OPTIONS,
    modelFile (length) {
      return `laya_english_fp16_L${length}_options32.mlpackage`
    }
  },
  multilingual: {
    repo: 'FluidInference/laya-coreml',
    revision: '7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc',
    tokenizerPath: 'tokenizer.json',
    lengths: [128, 256, 512, 1024],
    defaultLengths: [128, 512],
    precisions: ['fp16', 'e8'],
    cacheDir: null,
    maxLength: 1024,
    headMaxLength: 256,
    ids: { cls: 2, sep: 1, pad: 0, mask: 4, maskToken: '<mask>' },
    temperature: [1, 1, 1],
    temperatureByOptions: {},
    modelFile (length, precision) {
      return `laya_multilingual_${precision}_L${length}_options32.mlmodelc`
    }
  }
}

exports.load = async function (opts = {}) {
  if (process.platform !== 'darwin') throw new Error('Core ML requires macOS')

  const profileName = resolveProfileName(opts.model)
  const profile = MODEL_PROFILES[profileName]
  const precision = opts.precision || 'fp16'
  if (!profile.precisions.includes(precision)) {
    throw new Error(`${profileName} model precision must be one of ${profile.precisions}`)
  }

  const lengths = (opts.lengths || profile.defaultLengths).slice().sort((a, b) => a - b)
  if (!lengths.length || !lengths.every(length => profile.lengths.includes(length))) {
    throw new Error(`${profileName} model lengths must be selected from ${profile.lengths}`)
  }

  const native = await ensureNative()
  const modelDir = await ensureBundle(profile, lengths, precision)
  const rawTokenizer = await fsp.readFile(path.join(modelDir, profile.tokenizerPath), 'utf8')
  const tokenizer = new Tokenizer(JSON.parse(rawTokenizer), {})
  const buckets = []

  for (const length of lengths) {
    const modelPath = path.join(modelDir, profile.modelFile(length, precision))
    buckets.push({ length, model: native.open(modelPath, { units: 'all', lpaog: true }) })
  }

  const engine = {
    backend: 'coreml',
    model: profileName,
    padToMultiple: null,
    tokenizer,
    ids: profile.ids,
    config: {
      max_len: Math.min(profile.maxLength, lengths[lengths.length - 1]),
      head_max_len: profile.headMaxLength,
      temperature: profile.temperature,
      temperature_by_options: profile.temperatureByOptions
    },
    modelDir,

    async forward (batch) {
      if (batch.options > MAX_OPTIONS) {
        throw new Error(`coreml backend supports up to ${MAX_OPTIONS} options, got ${batch.options}`)
      }

      const logits = []
      const actProbabilities = []
      for (let row = 0; row < batch.size; row++) {
        const output = predictRow(native, profile, buckets, batch, row)
        logits.push(output.logits)
        actProbabilities.push(output.action)
      }

      return { logits, actProbabilities }
    },

    async close () {}
  }

  await warmUp(native, profile, buckets)
  return engine
}

function resolveProfileName (model) {
  if (model === undefined || model === 'english') return 'english'
  if (model === 'multilingual' || model === 'base') return 'multilingual'
  throw new Error('model must be english or multilingual')
}

function predictRow (native, profile, buckets, batch, row) {
  const realLength = countReal(batch.attention[row])
  const bucket = buckets.find(bucket => realLength <= bucket.length) || buckets[buckets.length - 1]
  const length = bucket.length
  const features = native.features
  const input = features.new(4)

  const ids = features.multiarray.new([1, length], DTYPES.i32)
  const attention = features.multiarray.new([1, length], DTYPES.i32)
  const idsView = new Int32Array(ids.buffer)
  const attentionView = new Int32Array(attention.buffer)
  idsView.fill(profile.ids.pad)
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

async function warmUp (native, profile, buckets) {
  for (const bucket of buckets) {
    const batch = {
      size: 1,
      length: bucket.length,
      options: 2,
      inputIds: new Int32Array(bucket.length).fill(profile.ids.pad),
      attention: [Array.from({ length: 8 }, () => 1)],
      markerPos: new Int32Array([2, 4]),
      markerMask: [[1, 1]],
      qtype: new Int32Array([0])
    }
    predictRow(native, profile, [bucket], batch, 0)
  }
}

function modelBase (profile) {
  return `https://huggingface.co/${profile.repo}/resolve/${profile.revision}`
}

function treeApi (profile) {
  return `https://huggingface.co/api/models/${profile.repo}/tree/${profile.revision}`
}

async function ensureBundle (profile, lengths, precision) {
  const dir = profile.cacheDir || path.join(CACHE_ROOT, precision)
  const tokenizerTarget = path.join(dir, profile.tokenizerPath)

  await Promise.all([
    ensureFile(profile, profile.tokenizerPath, tokenizerTarget),
    ...lengths.map(length => ensureModel(profile, dir, profile.modelFile(length, precision)))
  ])
  return dir
}

async function ensureModel (profile, dir, name) {
  const files = await listTree(profile, name)
  await Promise.all(files.map(file => ensureFile(profile, file.path, path.join(dir, file.path))))
}

async function ensureFile (profile, repoPath, target) {
  if (fs.existsSync(target)) return
  fs.mkdirSync(path.dirname(target), { recursive: true })
  await downloadFile(`${modelBase(profile)}/${repoPath}`, target)
}

async function listTree (profile, prefix) {
  const response = await fetch(`${treeApi(profile)}/${prefix}?recursive=true`)
  if (!response.ok) throw new Error(`Could not list ${prefix}: ${response.status} ${response.statusText}`)
  return (await response.json()).filter(entry => entry.type === 'file')
}

async function ensureNative () {
  const target = path.join(CACHE_ROOT, 'native', 'coreml.node')
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
