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

const MODEL_PROFILES = {
  'jev-multilingual-base': {
    repo: 'FluidInference/laya-coreml',
    revision: '7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc',
    tokenizerPath: 'tokenizer.json',
    lengths: [128, 256, 512, 1024],
    defaultLengths: [128, 512],
    cacheDir: null,
    maxLength: 1024,
    headMaxLength: 256,
    ids: { cls: 2, sep: 1, pad: 0, mask: 4, maskToken: '<mask>' },
    temperature: [1, 1, 1],
    temperatureByOptions: {},
    modelFile (length) {
      return `laya_multilingual_fp16_L${length}_options32.mlmodelc`
    }
  }
}

exports.load = async function (opts = {}) {
  if (process.platform !== 'darwin') throw new Error('Core ML requires macOS')

  const profileName = resolveProfileName(opts.model)
  const profile = MODEL_PROFILES[profileName]
  const precision = opts.precision || 'fp16'
  if (precision !== 'fp16') {
    throw new Error(`${profileName} model precision must be fp16`)
  }

  const lengths = (opts.lengths || profile.defaultLengths).slice().sort((a, b) => a - b)
  if (!lengths.length || !lengths.every(length => profile.lengths.includes(length))) {
    throw new Error(`${profileName} model lengths must be selected from ${profile.lengths}`)
  }

  const native = await ensureNative()
  const modelDir = await ensureBundle(profile, lengths)
  const rawTokenizer = await fsp.readFile(path.join(modelDir, profile.tokenizerPath), 'utf8')
  const tokenizer = new Tokenizer(JSON.parse(rawTokenizer), {})
  const buckets = []

  for (const length of lengths) {
    const modelPath = path.join(modelDir, profile.modelFile(length))
    const model = native.open(modelPath, { units: 'all', lpaog: true })
    buckets.push(createBucket(native, length, model))
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
      for (let row = 0; row < batch.size; row++) {
        const output = predictRow(native, profile, buckets, batch, row)
        logits.push(output.logits)
      }

      return { logits }
    },

    async close () {}
  }

  await warmUp(native, profile, buckets)
  return engine
}

function resolveProfileName (model) {
  if (model === undefined || model === 'jev-multilingual-base') return 'jev-multilingual-base'
  throw new Error('model must be jev-multilingual-base')
}

function predictRow (native, profile, buckets, batch, row) {
  const realLength = countReal(batch.attention[row])
  const bucket = buckets.find(bucket => realLength <= bucket.length) || buckets[buckets.length - 1]
  const length = bucket.length
  const { input, idsView, attentionView, markerView, typeView } = bucket

  idsView.fill(profile.ids.pad)
  attentionView.fill(0)
  for (let i = 0; i < Math.min(realLength, length); i++) {
    idsView[i] = batch.inputIds[row * batch.length + i]
    attentionView[i] = 1
  }

  markerView.fill(0)
  for (let slot = 0; slot < batch.options; slot++) {
    const position = batch.markerPos[row * batch.options + slot]
    if (batch.markerMask[row][slot] && position < length) markerView[slot * length + position] = 1
  }

  typeView.fill(0)
  typeView[batch.qtype[row]] = 1

  const output = native.predict(bucket.model, input)
  return {
    logits: readOutput(native.features, output, 'logits')
  }
}

function createBucket (native, length, model) {
  const features = native.features
  const input = features.new(4)
  const ids = features.multiarray.new([1, length], DTYPES.i32)
  const attention = features.multiarray.new([1, length], DTYPES.i32)
  const markerMap = features.multiarray.new([1, MAX_OPTIONS, length], DTYPES.f32)
  const questionType = features.multiarray.new([1, 3], DTYPES.f32)

  features.set(input, 'input_ids', ids)
  features.set(input, 'attention_mask', attention)
  features.set(input, 'marker_map', markerMap)
  features.set(input, 'question_type', questionType)

  return {
    length,
    model,
    input,
    idsView: new Int32Array(ids.buffer),
    attentionView: new Int32Array(attention.buffer),
    markerView: new Float32Array(markerMap.buffer),
    typeView: new Float32Array(questionType.buffer)
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

async function ensureBundle (profile, lengths) {
  const dir = profile.cacheDir || path.join(CACHE_ROOT, 'fp16')
  const tokenizerTarget = path.join(dir, profile.tokenizerPath)

  await Promise.all([
    ensureFile(profile, profile.tokenizerPath, tokenizerTarget),
    ...lengths.map(length => ensureModel(profile, dir, profile.modelFile(length)))
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
