const fsp = require('fs/promises')
const path = require('path')
const ort = require('onnxruntime-node')
const { Tokenizer } = require('@huggingface/tokenizers')
const { ensureBundle, cacheDir } = require('./bundle.js')

const MODEL = {
  ids: { cls: 2, sep: 1, pad: 0, mask: 4, maskToken: '<mask>' },
  base: 'https://huggingface.co/killkli/open-jev-laya-multilingual-onnx/resolve/643219b2b3a112ff1c40fb1143f050aa871f76f7',
  files: [
    'onnx/laya_fp16.onnx',
    'laya_config.json',
    'tokenizer.json',
    'tokenizer_config.json'
  ],
  modelFile: 'onnx/laya_fp16.onnx',
  configFile: 'laya_config.json',
  tokenizerFile: 'tokenizer.json',
  tokenizerConfigFile: null
}
const CACHE_ROOT = path.join(cacheDir('laya'), 'onnx')
const DEVICES = {
  webgpu: ['webgpu', 'cpu'],
  cpu: ['cpu'],
  cuda: ['cuda', 'cpu']
}

exports.load = async function (opts = {}) {
  const name = resolveModelName(opts.model)
  const precision = opts.precision || 'fp16'
  if (precision !== 'fp16') {
    throw new Error(`${name} model precision must be fp16`)
  }

  const build = MODEL

  ort.env.logLevel = opts.logLevel || 'error'

  const modelDir = await resolveModelDir(opts, name, precision, build)
  const config = JSON.parse(await fsp.readFile(path.join(modelDir, build.configFile), 'utf8'))
  const tokenizer = new Tokenizer(
    JSON.parse(await fsp.readFile(path.join(modelDir, build.tokenizerFile), 'utf8')),
    build.tokenizerConfigFile
      ? JSON.parse(await fsp.readFile(path.join(modelDir, build.tokenizerConfigFile), 'utf8'))
      : {}
  )
  const session = await createSession(opts, modelDir, build)

  return {
    backend: 'onnx',
    model: name,
    padToMultiple: null,
    tokenizer,
    ids: MODEL.ids,
    config,
    modelDir,

    async forward (batch) {
      const output = await session.run(buildInputs(batch))
      return readOutputs(output, batch)
    },

    async close () {
      await session.release()
    }
  }
}

function resolveModelName (model) {
  if (model === undefined || model === 'jev-multilingual-base') return 'jev-multilingual-base'
  throw new Error('model must be jev-multilingual-base')
}

async function resolveModelDir (opts, name, precision, build) {
  if (opts.modelDir) return path.resolve(opts.modelDir)

  return ensureBundle({
    files: build.files,
    base: build.base,
    cacheDir: path.join(CACHE_ROOT, 'multilingual', precision)
  })
}

async function createSession (opts, modelDir, build) {
  const providers = DEVICES[opts.device || 'webgpu']
  if (!providers) throw new Error(`device must be one of ${Object.keys(DEVICES)}`)

  return ort.InferenceSession.create(path.join(modelDir, build.modelFile), {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
    ...opts.sessionOptions
  })
}

function buildInputs (batch) {
  return {
    input_ids: new ort.Tensor('int64', toBigInt64(batch.inputIds), [batch.size, batch.length]),
    attention_mask: new ort.Tensor('int64', toBigInt64(batch.attention), [batch.size, batch.length]),
    marker_pos: new ort.Tensor('int64', toBigInt64(batch.markerPos), [batch.size, batch.options]),
    marker_mask: new ort.Tensor('bool', toUint8(batch.markerMask), [batch.size, batch.options]),
    qtype: new ort.Tensor('int64', toBigInt64(batch.qtype), [batch.size])
  }
}

function readOutputs (output, batch) {
  return {
    logits: sliceRows(output.logits.data, batch.size, batch.options)
  }
}

function sliceRows (flat, size, width) {
  const rows = []

  for (let row = 0; row < size; row++) {
    rows.push(Array.from(flat.subarray(row * width, (row + 1) * width)))
  }

  return rows
}

function toBigInt64 (values) {
  const flat = Array.isArray(values) ? values.flat() : Array.from(values)
  return BigInt64Array.from(flat, value => BigInt(value))
}

function toUint8 (nested) {
  return Uint8Array.from(nested.flat())
}
