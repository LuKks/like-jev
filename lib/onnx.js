const fsp = require('fs/promises')
const os = require('os')
const path = require('path')
const ort = require('onnxruntime-node')
const { Tokenizer } = require('@huggingface/tokenizers')
const { ensureBundle, cacheDir } = require('./bundle.js')

const MODELS = {
  english: {
    base: 'https://huggingface.co/receptron/laya-onnx/resolve/main',
    files: [
      'laya.onnx',
      'laya.onnx.data',
      'laya_config.json',
      'tokenizer/tokenizer.json',
      'tokenizer/tokenizer_config.json'
    ],
    modelFile: 'laya.onnx',
    tokenizerFile: 'tokenizer/tokenizer.json',
    tokenizerConfigFile: 'tokenizer/tokenizer_config.json',
    ids: { cls: 50281, sep: 50282, pad: 50283, mask: 50284, maskToken: '[MASK]' }
  },
  multilingual: {
    base: 'https://huggingface.co/killkli/open-jev-laya-multilingual-onnx/resolve/643219b2b3a112ff1c40fb1143f050aa871f76f7',
    files: [
      'onnx/laya.onnx',
      'laya_config.json',
      'tokenizer.json',
      'tokenizer_config.json'
    ],
    modelFile: 'onnx/laya.onnx',
    tokenizerFile: 'tokenizer.json',
    tokenizerConfigFile: null,
    ids: { cls: 2, sep: 1, pad: 0, mask: 4, maskToken: '<mask>' }
  }
}
const CACHE_ROOT = path.join(cacheDir('laya'), 'onnx')
const LEGACY_DIR = path.join(os.homedir(), '.cache', 'receptron-laya')

exports.load = async function (opts = {}) {
  const name = resolveModelName(opts.model)
  const model = MODELS[name]

  ort.env.logLevel = opts.logLevel || 'error'

  const modelDir = await resolveModelDir(opts, name, model)
  const config = JSON.parse(await fsp.readFile(path.join(modelDir, 'laya_config.json'), 'utf8'))
  const tokenizer = new Tokenizer(
    JSON.parse(await fsp.readFile(path.join(modelDir, model.tokenizerFile), 'utf8')),
    model.tokenizerConfigFile
      ? JSON.parse(await fsp.readFile(path.join(modelDir, model.tokenizerConfigFile), 'utf8'))
      : {}
  )
  const session = await createSession(opts, modelDir, model)

  return {
    backend: 'onnx',
    model: name,
    padToMultiple: null,
    tokenizer,
    ids: model.ids,
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
  if (model === undefined || model === 'english') return 'english'
  if (model === 'multilingual' || model === 'base') return 'multilingual'
  throw new Error('model must be english or multilingual')
}

async function resolveModelDir (opts, name, model) {
  if (opts.modelDir) return path.resolve(opts.modelDir)

  return ensureBundle({
    files: model.files,
    base: model.base,
    cacheDir: path.join(CACHE_ROOT, name),
    fallbackDir: name === 'english' ? LEGACY_DIR : null
  })
}

async function createSession (opts, modelDir, model) {
  return ort.InferenceSession.create(path.join(modelDir, model.modelFile), {
    executionProviders: opts.executionProviders || ['webgpu', 'cpu'],
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
  const act = output.act_probs.data
  const actWidth = output.act_probs.dims[1] || 1

  return {
    logits: sliceRows(output.logits.data, batch.size, batch.options),
    actProbabilities: sliceRows(act, batch.size, actWidth)
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
